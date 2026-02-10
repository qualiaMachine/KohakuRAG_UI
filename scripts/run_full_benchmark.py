#!/usr/bin/env python3
"""
Full Benchmark Runner

Runs ALL models on the training set with tagged inference profiles.
Includes a smoke test (1 question) before each full run to catch
failures early without burning through the whole question set.

Usage:
    # Smoke test only (1 question per model, fast)
    python scripts/run_full_benchmark.py --smoke-test

    # Full benchmark (all questions, all models)
    python scripts/run_full_benchmark.py

    # Single model only
    python scripts/run_full_benchmark.py --model haiku

Estimated cost: ~$2-3 for all 9 models on 41 questions.
Estimated time: ~15-30 minutes total.
"""

import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path


# Models to benchmark (config filename -> experiment name)
MODELS = {
    "bedrock_haiku": "haiku-bench",
    "bedrock_claude35_haiku": "claude35-haiku-bench",
    "bedrock_sonnet": "sonnet-bench",
    "bedrock_claude37_sonnet": "claude37-sonnet-bench",
    "bedrock_llama3_70b": "llama3-70b-bench",
    "bedrock_llama4_scout": "llama4-scout-bench",
    "bedrock_llama4_maverick": "llama4-maverick-bench",
    "bedrock_deepseek_v3": "deepseek-r1-bench",
    "bedrock_gpt_oss_20b": "gpt-oss-20b-bench",
    "bedrock_gpt_oss_120b": "gpt-oss-120b-bench",
    # NOTE: bedrock_nova_pro dropped per Chris (slow + worst performer)
    # NOTE: bedrock_mistral_small excluded -- model not available in this account
}


def run_experiment(config_name: str, experiment_name: str,
                   questions_path: str | None = None) -> tuple[bool, str]:
    """Run a single experiment. Returns (success, output_summary)."""
    config_path = f"configs/{config_name}.py"

    if not Path(config_path).exists():
        return False, f"Config not found: {config_path}"

    cmd = [
        sys.executable, "scripts/run_experiment.py",
        "--config", config_path,
        "--name", experiment_name,
    ]

    if questions_path:
        cmd.extend(["--questions", questions_path])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout per model (needed for 282-question test set)
        )

        # Extract key metrics from output
        output = result.stdout + result.stderr
        success = result.returncode == 0

        # Look for the score line
        score_line = ""
        for line in output.split("\n"):
            if "OVERALL SCORE" in line:
                score_line = line.strip()
            elif "Error" in line and "Traceback" not in line:
                score_line = line.strip()[:120]

        if not success:
            # Get the error
            error_lines = [l for l in output.split("\n") if "Error" in l or "error" in l]
            error_summary = error_lines[-1][:150] if error_lines else "Unknown error"
            return False, error_summary

        return True, score_line

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT (>10 min)"
    except Exception as e:
        return False, str(e)[:150]


def main():
    parser = argparse.ArgumentParser(description="Run full benchmark across all models")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Only run 1 question per model to verify they work",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Run only a specific model (e.g., 'haiku', 'sonnet')",
    )
    parser.add_argument(
        "--questions", "-q", type=str, default=None,
        help="Override questions CSV path (e.g., path/to/test_solutions.csv)",
    )
    parser.add_argument(
        "--suffix", type=str, default="bench",
        help="Experiment name suffix (default: 'bench'). Use 'test' for test set runs.",
    )
    args = parser.parse_args()

    # Filter models if specific one requested
    models = MODELS
    if args.model:
        models = {k: v for k, v in MODELS.items() if args.model.lower() in k.lower() or args.model.lower() in v.lower()}
        if not models:
            print(f"No model matching '{args.model}'. Available:")
            for k in MODELS:
                print(f"  {k}")
            return

    # Override experiment names with custom suffix
    if args.suffix != "bench":
        models = {k: v.replace("-bench", f"-{args.suffix}") for k, v in models.items()}

    mode = "SMOKE TEST" if args.smoke_test else "FULL BENCHMARK"
    q_info = f" | questions: {args.questions}" if args.questions else ""
    print(f"{'='*70}")
    print(f"{mode}: {len(models)} models{q_info}")
    print(f"{'='*70}\n")

    results = {}
    total_start = time.time()

    for i, (config_name, exp_name) in enumerate(models.items(), 1):
        if args.smoke_test:
            exp_name = f"{exp_name}-smoke"

        print(f"[{i}/{len(models)}] {config_name} -> {exp_name}")
        start = time.time()

        success, summary = run_experiment(config_name, exp_name, args.questions)
        elapsed = time.time() - start

        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {elapsed:.1f}s - {summary}")
        print()

        results[config_name] = {
            "experiment": exp_name,
            "success": success,
            "summary": summary,
            "time": elapsed,
        }

    # Final summary
    total_time = time.time() - total_start
    passed = sum(1 for r in results.values() if r["success"])
    failed = sum(1 for r in results.values() if not r["success"])

    print(f"\n{'='*70}")
    print(f"RESULTS: {passed} passed, {failed} failed ({total_time:.0f}s total)")
    print(f"{'='*70}")

    for config_name, r in results.items():
        status = "PASS" if r["success"] else "FAIL"
        print(f"  [{status}] {config_name:<35s} {r['time']:>6.1f}s  {r['summary'][:60]}")

    if failed > 0:
        print(f"\n!!! {failed} model(s) FAILED -- fix before running full benchmark !!!")
        sys.exit(1)
    else:
        print(f"\nAll {passed} models passed!")
        if args.smoke_test:
            print("\nRun without --smoke-test for the full benchmark.")


if __name__ == "__main__":
    main()
