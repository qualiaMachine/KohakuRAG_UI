#!/usr/bin/env python3
"""Streamlit UI for document extraction.

Upload PDFs, TIFFs, or images to extract structured data.
- Digital PDFs: text extraction + LLM parsing (fast, no GPU)
- Scans / TIFFs: VLM OCR + structuring (uses GPU)

Launch:
    streamlit run ocr_app/app.py

Environment variables:
    OCR_SERVICE_URL  - URL of the extraction server (default: http://localhost:8090)
"""

import io
import os
from pathlib import Path

import httpx
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OCR_SERVICE_URL = os.environ.get("OCR_SERVICE_URL", "http://localhost:8090")

st.set_page_config(
    page_title="Document Extraction",
    page_icon="\U0001F4C4",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=5)
def _check_server() -> dict | None:
    try:
        resp = httpx.get(f"{OCR_SERVICE_URL}/health", timeout=5.0)
        return resp.json()
    except Exception:
        return None


def _extract_pdf(file_bytes: bytes, filename: str, fmt: str,
                 prompt: str | None, max_tokens: int, pages: str | None) -> dict:
    files = {"file": (filename, file_bytes)}
    data = {"format": fmt, "max_tokens": str(max_tokens)}
    if prompt:
        data["prompt"] = prompt
    if pages:
        data["pages"] = pages
    resp = httpx.post(
        f"{OCR_SERVICE_URL}/extract/pdf", files=files, data=data, timeout=300.0,
    )
    resp.raise_for_status()
    return resp.json()


def _extract_image(file_bytes: bytes, filename: str, fmt: str,
                   prompt: str | None, max_tokens: int) -> dict:
    files = {"file": (filename, file_bytes)}
    data = {"format": fmt, "max_tokens": str(max_tokens)}
    if prompt:
        data["prompt"] = prompt
    resp = httpx.post(
        f"{OCR_SERVICE_URL}/extract/image", files=files, data=data, timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()


def _format_elapsed(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def _render_result(text: str, fmt: str):
    if fmt in ("json", "award", "budget", "terms", "key_values"):
        st.code(text, language="json")
    elif fmt in ("markdown", "table"):
        st.markdown(text)
    else:
        st.text(text)


def _method_badge(method: str) -> str:
    if method == "text_extraction":
        return "\u26A1 text extraction + LLM"
    return "\U0001F50D VLM OCR"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("\U0001F4C4 Document Extraction")
    st.caption("Document Processing — Grants, Archives & More")

    status = _check_server()
    if status and status.get("status") == "ok":
        llm = status.get("llm_model", "unknown")
        st.success(f"LLM: {llm}")
    else:
        st.error(f"Server unreachable at {OCR_SERVICE_URL}")

    st.divider()

    FORMAT_OPTIONS = {
        "award": "Award Notice (structured JSON)",
        "budget": "Budget / Financial (structured JSON)",
        "terms": "Terms & Conditions (structured JSON)",
        "table": "Tables (Markdown)",
        "key_values": "Key-Value Pairs (JSON)",
        "markdown": "Markdown",
        "json": "JSON (generic)",
        "text": "Plain Text",
    }
    output_format = st.selectbox(
        "Output format",
        list(FORMAT_OPTIONS.keys()),
        index=0,
        format_func=lambda x: FORMAT_OPTIONS[x],
    )

    with st.expander("Advanced options"):
        max_tokens = st.slider(
            "Max output tokens", 256, 8192, 4096, 256,
        )
        custom_prompt = st.text_area(
            "Custom prompt (optional)",
            placeholder="Override the default extraction prompt...",
        )
        pdf_pages = st.text_input(
            "PDF pages (optional)",
            placeholder="e.g., 1-5 or 1,3,5",
        )

    st.divider()
    st.caption(f"Server: `{OCR_SERVICE_URL}`")
    st.caption(
        "Digital PDFs: text extract + LLM (fast)\n\n"
        "Scans / TIFFs: VLM OCR (GPU)"
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("Document Extraction")
st.markdown(
    "Upload grant award notices, budgets, terms & conditions, archival "
    "documents, or other institutional records. All pages are processed "
    "via VLM for structured extraction."
)

uploaded_files = st.file_uploader(
    "Upload PDFs, TIFFs, or images",
    type=["pdf", "tiff", "tif", "png", "jpg", "jpeg", "webp", "bmp", "gif"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more files to get started.")
    st.stop()

run = st.button("Extract", type="primary", use_container_width=True)

if not run:
    cols = st.columns(min(len(uploaded_files), 4))
    for i, f in enumerate(uploaded_files):
        with cols[i % len(cols)]:
            if f.type == "application/pdf":
                st.markdown(f"**{f.name}** (PDF, {len(f.getvalue()) / 1024:.0f} KB)")
            else:
                img = Image.open(io.BytesIO(f.getvalue()))
                st.image(img, caption=f.name, use_container_width=True)
    st.stop()

# ---------------------------------------------------------------------------
# Process files
# ---------------------------------------------------------------------------
prompt = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else None

for uploaded_file in uploaded_files:
    file_bytes = uploaded_file.getvalue()
    is_pdf = uploaded_file.type == "application/pdf"
    is_image = not is_pdf

    st.divider()
    st.subheader(uploaded_file.name)

    with st.spinner(f"Processing {uploaded_file.name}..."):
        try:
            if is_pdf:
                result = _extract_pdf(
                    file_bytes, uploaded_file.name, output_format,
                    prompt, max_tokens, pdf_pages or None,
                )

                # Summary metrics
                total_ms = result.get("total_elapsed_ms", 0)
                digital = result.get("digital_pages", 0)
                scanned = result.get("scanned_pages", 0)
                total = result.get("total_pages", 0)

                col_stats = st.columns(4)
                col_stats[0].metric("Pages", total)
                col_stats[1].metric("Digital", digital)
                col_stats[2].metric("Scanned", scanned)
                col_stats[3].metric("Time", _format_elapsed(total_ms))

                # Per-page results
                for page_result in result.get("pages", []):
                    page_num = page_result["page"]
                    method = page_result["method"]
                    elapsed = page_result["elapsed_ms"]
                    label = f"Page {page_num} \u2014 {_method_badge(method)} ({_format_elapsed(elapsed)})"

                    with st.expander(label, expanded=(page_num == 1)):
                        _render_result(page_result["text"], output_format)

                # Download all pages
                all_text = "\n\n---\n\n".join(
                    p["text"] for p in result.get("pages", [])
                )

            else:
                # Image / TIFF
                col_img, col_result = st.columns([1, 1])
                with col_img:
                    img = Image.open(io.BytesIO(file_bytes))
                    st.image(img, caption=f"{img.width}x{img.height}", use_container_width=True)

                result = _extract_image(
                    file_bytes, uploaded_file.name, output_format,
                    prompt, max_tokens,
                )

                with col_result:
                    elapsed = result.get("elapsed_ms", 0)
                    method = result.get("method", "vlm_ocr")
                    st.caption(f"{_method_badge(method)} \u2014 {_format_elapsed(elapsed)}")
                    _render_result(result["text"], output_format)

                all_text = result["text"]

            # Download button
            ext = {"json": "json", "award": "json", "budget": "json",
                   "terms": "json", "key_values": "json",
                   "markdown": "md", "table": "md"}.get(output_format, "txt")
            mime = "application/json" if ext == "json" else "text/plain"
            st.download_button(
                f"Download {uploaded_file.name} result",
                data=all_text,
                file_name=f"{Path(uploaded_file.name).stem}_extracted.{ext}",
                mime=mime,
            )

        except httpx.HTTPStatusError as e:
            st.error(f"Server error: {e.response.status_code} - {e.response.text}")
        except httpx.ConnectError:
            st.error(f"Cannot connect to server at {OCR_SERVICE_URL}")
        except Exception as e:
            st.error(f"Error: {e}")
