"""
Hardware Metrics Collection for Model Experiments

Measures:
- GPU VRAM usage (peak allocated, peak reserved)
- Model disk size (HuggingFace cache)
- GPU power draw and energy consumption (Watt-hours)
- System memory usage

Uses nvidia-smi for power monitoring (works on any NVIDIA GPU).
Falls back gracefully when GPU is not available.
"""

import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Optional torch import (for VRAM measurement)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class HardwareMetrics:
    """Hardware resource metrics for an experiment run."""

    # GPU memory (bytes)
    gpu_vram_allocated_bytes: int = 0       # Peak allocated by PyTorch
    gpu_vram_reserved_bytes: int = 0        # Peak reserved by PyTorch (includes fragmentation)
    gpu_vram_total_bytes: int = 0           # Total VRAM on device

    # Friendly versions (GB)
    gpu_vram_allocated_gb: float = 0.0
    gpu_vram_reserved_gb: float = 0.0
    gpu_vram_total_gb: float = 0.0

    # Model disk size
    model_disk_size_bytes: int = 0
    model_disk_size_gb: float = 0.0
    model_cache_path: str = ""

    # Energy / power
    gpu_energy_wh: float = 0.0              # Total energy consumed during experiment
    gpu_avg_power_watts: float = 0.0        # Average power draw
    gpu_peak_power_watts: float = 0.0       # Peak power draw observed
    gpu_power_samples: int = 0              # Number of power readings taken

    # Timing
    model_load_time_seconds: float = 0.0    # Time to load model into VRAM

    # System
    gpu_name: str = ""
    gpu_device_id: int = 0
    cuda_version: str = ""


def _bytes_to_gb(b: int) -> float:
    return round(b / (1024 ** 3), 3)


# =============================================================================
# GPU VRAM Measurement
# =============================================================================

def get_gpu_vram_snapshot(device_id: int = 0) -> dict:
    """Take a snapshot of current GPU VRAM usage via PyTorch."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return {}

    device = torch.device(f"cuda:{device_id}")
    return {
        "allocated": torch.cuda.memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
        "max_allocated": torch.cuda.max_memory_allocated(device),
        "max_reserved": torch.cuda.max_memory_reserved(device),
        "total": torch.cuda.get_device_properties(device).total_mem,
    }


def reset_gpu_peak_stats(device_id: int = 0) -> None:
    """Reset peak memory tracking. Call BEFORE the experiment starts."""
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_id)


def get_gpu_info(device_id: int = 0) -> dict:
    """Get GPU name, CUDA version, total VRAM."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return {"name": "N/A", "cuda_version": "N/A", "total_gb": 0}

    props = torch.cuda.get_device_properties(device_id)
    return {
        "name": props.name,
        "cuda_version": torch.version.cuda or "unknown",
        "total_bytes": props.total_mem,
        "total_gb": _bytes_to_gb(props.total_mem),
    }


# =============================================================================
# Model Disk Size
# =============================================================================

def get_model_disk_size(model_id: str) -> dict:
    """Measure total disk size of a HuggingFace model in the local cache.

    Checks both the HF_HOME/hub cache and the default ~/.cache/huggingface/hub.
    """
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = Path(hf_home) / "hub"

    # HF cache structure: models--org--name
    safe_name = f"models--{model_id.replace('/', '--')}"
    model_cache = hub_dir / safe_name

    if not model_cache.exists():
        # Try alternate cache locations
        for alt in [
            hub_dir / model_id.replace("/", "--"),
            hub_dir / model_id.split("/")[-1],
        ]:
            if alt.exists():
                model_cache = alt
                break

    if not model_cache.exists():
        return {"path": str(model_cache), "size_bytes": 0, "size_gb": 0.0, "found": False}

    total = 0
    for f in model_cache.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass

    return {
        "path": str(model_cache),
        "size_bytes": total,
        "size_gb": _bytes_to_gb(total),
        "found": True,
    }


# =============================================================================
# GPU Power Monitoring (nvidia-smi based)
# =============================================================================

class GPUPowerMonitor:
    """Background thread that polls nvidia-smi for GPU power draw.

    Usage:
        monitor = GPUPowerMonitor(device_id=0, interval=1.0)
        monitor.start()
        # ... run experiment ...
        monitor.stop()
        print(f"Energy: {monitor.energy_wh:.3f} Wh")
    """

    def __init__(self, device_id: int = 0, interval: float = 1.0):
        self.device_id = device_id
        self.interval = interval
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._power_readings: list[float] = []  # watts
        self._timestamps: list[float] = []
        self._available = shutil.which("nvidia-smi") is not None

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        """Start background power monitoring."""
        if not self._available:
            return
        self._stop_event.clear()
        self._power_readings = []
        self._timestamps = []
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop monitoring and compute totals."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None

    def _poll_loop(self) -> None:
        """Poll nvidia-smi in a loop."""
        while not self._stop_event.is_set():
            power = self._read_power()
            if power is not None:
                self._power_readings.append(power)
                self._timestamps.append(time.time())
            self._stop_event.wait(self.interval)

    def _read_power(self) -> float | None:
        """Read current GPU power draw in watts from nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.device_id}",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                val = result.stdout.strip()
                if val and val != "[N/A]":
                    return float(val)
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            pass
        return None

    @property
    def energy_wh(self) -> float:
        """Total energy consumed in Watt-hours (trapezoidal integration)."""
        if len(self._timestamps) < 2:
            return 0.0

        total_joules = 0.0
        for i in range(1, len(self._timestamps)):
            dt = self._timestamps[i] - self._timestamps[i - 1]
            avg_power = (self._power_readings[i] + self._power_readings[i - 1]) / 2
            total_joules += avg_power * dt

        return total_joules / 3600  # joules -> Wh

    @property
    def avg_power_watts(self) -> float:
        if not self._power_readings:
            return 0.0
        return sum(self._power_readings) / len(self._power_readings)

    @property
    def peak_power_watts(self) -> float:
        if not self._power_readings:
            return 0.0
        return max(self._power_readings)

    @property
    def num_samples(self) -> int:
        return len(self._power_readings)


# =============================================================================
# Convenience: Collect All Metrics
# =============================================================================

def collect_pre_experiment_metrics(
    model_id: str,
    device_id: int = 0,
) -> dict:
    """Collect metrics BEFORE the experiment (model disk size, GPU info).

    Call this after model is loaded but before running questions.
    """
    gpu_info = get_gpu_info(device_id)
    disk_info = get_model_disk_size(model_id)
    vram = get_gpu_vram_snapshot(device_id)

    return {
        "gpu_info": gpu_info,
        "disk_info": disk_info,
        "vram_after_load": vram,
    }


def collect_post_experiment_metrics(
    model_id: str,
    device_id: int = 0,
    power_monitor: GPUPowerMonitor | None = None,
    model_load_time: float = 0.0,
) -> HardwareMetrics:
    """Collect all metrics AFTER the experiment completes."""
    gpu_info = get_gpu_info(device_id)
    disk_info = get_model_disk_size(model_id)
    vram = get_gpu_vram_snapshot(device_id)

    metrics = HardwareMetrics(
        # VRAM
        gpu_vram_allocated_bytes=vram.get("max_allocated", 0),
        gpu_vram_reserved_bytes=vram.get("max_reserved", 0),
        gpu_vram_total_bytes=vram.get("total", 0),
        gpu_vram_allocated_gb=_bytes_to_gb(vram.get("max_allocated", 0)),
        gpu_vram_reserved_gb=_bytes_to_gb(vram.get("max_reserved", 0)),
        gpu_vram_total_gb=_bytes_to_gb(vram.get("total", 0)),
        # Disk
        model_disk_size_bytes=disk_info.get("size_bytes", 0),
        model_disk_size_gb=disk_info.get("size_gb", 0.0),
        model_cache_path=disk_info.get("path", ""),
        # Power
        gpu_energy_wh=power_monitor.energy_wh if power_monitor else 0.0,
        gpu_avg_power_watts=power_monitor.avg_power_watts if power_monitor else 0.0,
        gpu_peak_power_watts=power_monitor.peak_power_watts if power_monitor else 0.0,
        gpu_power_samples=power_monitor.num_samples if power_monitor else 0,
        # Timing
        model_load_time_seconds=model_load_time,
        # System
        gpu_name=gpu_info.get("name", ""),
        gpu_device_id=device_id,
        cuda_version=gpu_info.get("cuda_version", ""),
    )

    return metrics
