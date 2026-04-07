#!/usr/bin/env python3
"""FastAPI OCR server using Vision Language Models.

Serves OCR over HTTP using Qwen2.5-VL (primary) or GOT-OCR2.0 (lightweight).
Accepts images via multipart upload or base64, returns extracted text with
optional structured output (JSON, markdown tables).

Launch:
    python ocr_app/scripts/ocr_server.py
    # or with uvicorn directly:
    uvicorn ocr_app.scripts.ocr_server:app --host 0.0.0.0 --port 8090

Environment variables:
    OCR_MODEL       - Model ID (default: Qwen/Qwen2.5-VL-7B-Instruct)
    OCR_PORT        - Server port (default: 8090)
    OCR_HOST        - Server host (default: 0.0.0.0)
    OCR_MAX_PIXELS  - Max image pixels before resize (default: 1280*28*28)
    OCR_DEVICE      - Device: cuda, cpu, auto (default: auto)
    VLLM_BASE_URL   - If set, use vLLM backend instead of local transformers
"""

import base64
import io
import os
import time
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("OCR_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
HOST = os.environ.get("OCR_HOST", "0.0.0.0")
PORT = int(os.environ.get("OCR_PORT", "8090"))
MAX_PIXELS = int(os.environ.get("OCR_MAX_PIXELS", str(1280 * 28 * 28)))
MIN_PIXELS = 256 * 28 * 28
DEVICE = os.environ.get("OCR_DEVICE", "auto")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "")

# ---------------------------------------------------------------------------
# Model backends
# ---------------------------------------------------------------------------

_model = None
_processor = None
_backend = None  # "transformers" or "vllm"


def _load_transformers_model():
    """Load Qwen2.5-VL via HuggingFace transformers."""
    global _model, _processor, _backend
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = DEVICE
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[ocr_server] Loading {MODEL_NAME} on {device}...", flush=True)
    t0 = time.time()

    _processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=device if device == "auto" else {"": device},
        attn_implementation="flash_attention_2" if device == "cuda" else "sdpa",
    )

    elapsed = time.time() - t0
    print(f"[ocr_server] Model loaded in {elapsed:.1f}s", flush=True)
    _backend = "transformers"


def _init_vllm_backend():
    """Configure vLLM remote backend (no local model loading)."""
    global _backend
    print(f"[ocr_server] Using vLLM backend at {VLLM_BASE_URL}", flush=True)
    _backend = "vllm"


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------

def _build_messages(image: Image.Image, prompt: str) -> list[dict]:
    """Build Qwen2.5-VL chat messages with an image."""
    # Convert image to base64 for the message
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image/png;base64,{b64}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


async def _infer_transformers(image: Image.Image, prompt: str, max_tokens: int) -> str:
    """Run inference using local transformers model."""
    from qwen_vl_utils import process_vision_info

    messages = _build_messages(image, prompt)

    text_input = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(_model.device)

    with torch.no_grad():
        generated_ids = _model.generate(**inputs, max_new_tokens=max_tokens)

    # Trim input tokens from output
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    result = _processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return result[0]


async def _infer_vllm(image: Image.Image, prompt: str, max_tokens: int) -> str:
    """Run inference via remote vLLM OpenAI-compatible API."""
    import httpx

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{VLLM_BASE_URL}/chat/completions", json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def _run_ocr(image: Image.Image, prompt: str, max_tokens: int) -> str:
    """Route to the active backend."""
    if _backend == "vllm":
        return await _infer_vllm(image, prompt, max_tokens)
    return await _infer_transformers(image, prompt, max_tokens)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    award = "award"
    budget = "budget"
    terms = "terms"
    table = "table"
    key_values = "key_values"
    markdown = "markdown"
    json = "json"
    text = "text"


PROMPTS = {
    OutputFormat.award: (
        "Extract all information from this grant award notice / notice of award. "
        "This is a research and sponsored programs document.\n\n"
        "Return a JSON object with these fields (omit any that are not present):\n"
        '  "document_type": type of document (e.g. "Notice of Award", "Award Letter", "Subaward Agreement"),\n'
        '  "award_number": grant/award/contract number,\n'
        '  "sponsor": funding agency or organization,\n'
        '  "pi": principal investigator name(s),\n'
        '  "co_pis": array of co-PI names if listed,\n'
        '  "institution": awardee institution/organization,\n'
        '  "department": department or unit,\n'
        '  "project_title": title of the funded project,\n'
        '  "award_amount": total award amount (preserve exact formatting),\n'
        '  "current_period_amount": current period funding if shown,\n'
        '  "project_start": project/budget period start date,\n'
        '  "project_end": project/budget period end date,\n'
        '  "budget_periods": array of {period, start, end, amount} if shown,\n'
        '  "fa_rate": F&A / indirect cost rate and base if shown,\n'
        '  "cfda_number": CFDA number if shown,\n'
        '  "award_type": e.g. "New", "Continuation", "Supplement", "No-Cost Extension",\n'
        '  "special_conditions": array of any special terms or conditions,\n'
        '  "contacts": array of {role, name, email, phone} for program officers or admin contacts,\n'
        '  "additional_fields": object with any other labeled fields not covered above\n\n'
        "Preserve ALL dollar amounts, dates, and reference numbers exactly as printed. "
        "Output only valid JSON."
    ),
    OutputFormat.budget: (
        "Extract the budget information from this research grant document. "
        "This may be a budget page, budget justification, or financial summary "
        "from a grant award, proposal, or sponsored program.\n\n"
        "Return a JSON object with:\n"
        '  "award_number": grant/award number if shown,\n'
        '  "budget_period": period or fiscal year if shown,\n'
        '  "categories": array of objects, each with:\n'
        '    "category": budget category (e.g. "Senior Personnel", "Fringe Benefits", '
        '"Equipment", "Travel", "Participant Support", "Other Direct Costs", "Indirect Costs"),\n'
        '    "items": array of {description, amount} line items,\n'
        '    "subtotal": category subtotal if shown\n'
        '  "total_direct": total direct costs,\n'
        '  "fa_rate": F&A / indirect cost rate,\n'
        '  "fa_base": F&A base (MTDC, TDC, etc.),\n'
        '  "total_indirect": total indirect/F&A costs,\n'
        '  "total": total project costs,\n'
        '  "cost_sharing": cost sharing amount if any,\n'
        '  "notes": any footnotes or annotations\n\n'
        "CRITICAL: Preserve ALL dollar amounts exactly as printed — do not "
        "round, drop commas, or reformat. Output only valid JSON."
    ),
    OutputFormat.terms: (
        "Extract the terms and conditions from this research and sponsored "
        "programs document. This may be award terms, RSP policies, compliance "
        "requirements, subaward terms, or similar.\n\n"
        "Return a JSON object with:\n"
        '  "document_title": title of the document,\n'
        '  "effective_date": effective date if shown,\n'
        '  "sections": array of objects, each with:\n'
        '    "number": section number if present,\n'
        '    "title": section heading,\n'
        '    "text": full text of the section,\n'
        '    "subsections": array of {number, title, text} if nested\n'
        '  "definitions": object of defined terms if present,\n'
        '  "references": array of referenced regulations, OMB circulars, CFR citations\n\n'
        "Preserve exact wording — do not paraphrase or summarize. "
        "Include all regulatory citations (2 CFR 200, etc.) exactly. "
        "Output only valid JSON."
    ),
    OutputFormat.table: (
        "Extract the table(s) from this image. Return each table as a "
        "Markdown table with proper column alignment. If there are multiple "
        "tables, separate them with a blank line. "
        "Preserve ALL numeric values exactly — do not round, truncate, or "
        "reformat numbers. Include currency symbols, percentages, and units. "
        "Output only the table(s)."
    ),
    OutputFormat.key_values: (
        "Extract all labeled data points from this image as key-value pairs. "
        "Look for field labels, line items, metrics, reference numbers, dates, "
        "names, and their corresponding values. "
        "Return a JSON object where keys are the field/label names and "
        "values are their corresponding values.\n\n"
        "Preserve ALL values exactly as printed — do not round or reformat. "
        "Include units, currency symbols, and percentages. "
        "For nested sections, use nested objects. Output only valid JSON."
    ),
    OutputFormat.markdown: (
        "Extract all text from this image and format it as clean Markdown. "
        "Use headings, lists, bold/italic, and code blocks where appropriate. "
        "Preserve tables as Markdown tables. Output only the Markdown."
    ),
    OutputFormat.json: (
        "Extract all text from this image and return it as a JSON object. "
        "Structure the content logically with appropriate keys. "
        "For forms, use field names as keys and field values as values. "
        "For documents, use sections as keys. Output only valid JSON."
    ),
    OutputFormat.text: (
        "Extract all text from this image exactly as it appears. "
        "Preserve the original reading order, line breaks, and structure. "
        "Output only the extracted text, nothing else."
    ),
}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if VLLM_BASE_URL:
        _init_vllm_backend()
    else:
        _load_transformers_model()
    yield
    # Shutdown (nothing to clean up)


app = FastAPI(
    title="VLM OCR Server",
    version="1.0.0",
    description="OCR service powered by Qwen2.5-VL",
    lifespan=lifespan,
)


class OCRRequest(BaseModel):
    """Request body for base64 image OCR."""
    image_base64: str
    format: OutputFormat = OutputFormat.text
    prompt: Optional[str] = None
    max_tokens: int = 4096


class OCRResponse(BaseModel):
    """OCR result."""
    text: str
    format: str
    model: str
    elapsed_ms: float
    image_width: int
    image_height: int


class BatchOCRRequest(BaseModel):
    """Request body for batch OCR of multiple base64 images."""
    images_base64: list[str]
    format: OutputFormat = OutputFormat.text
    prompt: Optional[str] = None
    max_tokens: int = 4096


class BatchOCRResponse(BaseModel):
    """Batch OCR results."""
    results: list[OCRResponse]
    total_elapsed_ms: float
    count: int


def _decode_image(data: bytes) -> Image.Image:
    """Decode image bytes to PIL Image, converting to RGB."""
    img = Image.open(io.BytesIO(data))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


@app.get("/health")
async def health():
    if _backend is None:
        return {"status": "loading"}
    return {"status": "ok", "model": MODEL_NAME, "backend": _backend}


@app.get("/info")
async def info():
    if _backend is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {
        "model": MODEL_NAME,
        "backend": _backend,
        "device": DEVICE,
        "max_pixels": MAX_PIXELS,
        "formats": [f.value for f in OutputFormat],
    }


@app.post("/ocr", response_model=OCRResponse)
async def ocr_base64(request: OCRRequest):
    """OCR from a base64-encoded image."""
    if _backend is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    image_bytes = base64.b64decode(request.image_base64)
    image = _decode_image(image_bytes)
    prompt = request.prompt or PROMPTS[request.format]

    t0 = time.time()
    text = await _run_ocr(image, prompt, request.max_tokens)
    elapsed_ms = (time.time() - t0) * 1000

    return OCRResponse(
        text=text,
        format=request.format.value,
        model=MODEL_NAME,
        elapsed_ms=round(elapsed_ms, 2),
        image_width=image.width,
        image_height=image.height,
    )


@app.post("/ocr/upload", response_model=OCRResponse)
async def ocr_upload(
    file: UploadFile = File(...),
    format: OutputFormat = Form(OutputFormat.text),
    prompt: Optional[str] = Form(None),
    max_tokens: int = Form(4096),
):
    """OCR from an uploaded image file."""
    if _backend is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    contents = await file.read()
    image = _decode_image(contents)
    actual_prompt = prompt or PROMPTS[format]

    t0 = time.time()
    text = await _run_ocr(image, actual_prompt, max_tokens)
    elapsed_ms = (time.time() - t0) * 1000

    return OCRResponse(
        text=text,
        format=format.value,
        model=MODEL_NAME,
        elapsed_ms=round(elapsed_ms, 2),
        image_width=image.width,
        image_height=image.height,
    )


@app.post("/ocr/batch", response_model=BatchOCRResponse)
async def ocr_batch(request: BatchOCRRequest):
    """OCR multiple base64-encoded images sequentially."""
    if _backend is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    prompt = request.prompt or PROMPTS[request.format]
    results = []
    total_t0 = time.time()

    for b64 in request.images_base64:
        image_bytes = base64.b64decode(b64)
        image = _decode_image(image_bytes)

        t0 = time.time()
        text = await _run_ocr(image, prompt, request.max_tokens)
        elapsed_ms = (time.time() - t0) * 1000

        results.append(OCRResponse(
            text=text,
            format=request.format.value,
            model=MODEL_NAME,
            elapsed_ms=round(elapsed_ms, 2),
            image_width=image.width,
            image_height=image.height,
        ))

    total_elapsed = (time.time() - total_t0) * 1000

    return BatchOCRResponse(
        results=results,
        total_elapsed_ms=round(total_elapsed, 2),
        count=len(results),
    )


@app.post("/ocr/pdf", response_model=BatchOCRResponse)
async def ocr_pdf(
    file: UploadFile = File(...),
    format: OutputFormat = Form(OutputFormat.text),
    prompt: Optional[str] = Form(None),
    max_tokens: int = Form(4096),
    pages: Optional[str] = Form(None),
):
    """OCR a PDF by rendering each page as an image.

    Args:
        pages: Page range like "1-5", "1,3,5", or None for all pages.
    """
    if _backend is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="PyMuPDF not installed. Install with: pip install pymupdf",
        )

    contents = await file.read()
    doc = fitz.open(stream=contents, filetype="pdf")

    # Parse page selection
    page_indices = _parse_pages(pages, len(doc))

    actual_prompt = prompt or PROMPTS[format]
    results = []
    total_t0 = time.time()

    for page_idx in page_indices:
        page = doc[page_idx]
        # Render at 2x for better OCR quality
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        t0 = time.time()
        text = await _run_ocr(img, actual_prompt, max_tokens)
        elapsed_ms = (time.time() - t0) * 1000

        results.append(OCRResponse(
            text=text,
            format=format.value,
            model=MODEL_NAME,
            elapsed_ms=round(elapsed_ms, 2),
            image_width=img.width,
            image_height=img.height,
        ))

    doc.close()
    total_elapsed = (time.time() - total_t0) * 1000

    return BatchOCRResponse(
        results=results,
        total_elapsed_ms=round(total_elapsed, 2),
        count=len(results),
    )


def _parse_pages(pages_str: Optional[str], total_pages: int) -> list[int]:
    """Parse a page range string like '1-5' or '1,3,5' into 0-indexed list."""
    if not pages_str:
        return list(range(total_pages))

    indices = set()
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start = max(1, int(start))
            end = min(total_pages, int(end))
            indices.update(range(start - 1, end))
        else:
            idx = int(part) - 1
            if 0 <= idx < total_pages:
                indices.add(idx)

    return sorted(indices)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
