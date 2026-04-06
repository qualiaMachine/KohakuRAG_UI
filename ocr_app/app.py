#!/usr/bin/env python3
"""Streamlit OCR application powered by VLMs.

Upload images or PDFs, extract text using Qwen2.5-VL or other VLMs,
and get structured output (plain text, markdown, JSON, tables).

Launch:
    streamlit run ocr_app/app.py

Environment variables:
    OCR_SERVICE_URL  - URL of the OCR server (default: http://localhost:8090)
"""

import base64
import io
import os
import time
from pathlib import Path

import httpx
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OCR_SERVICE_URL = os.environ.get("OCR_SERVICE_URL", "http://localhost:8090")
PAGE_TITLE = "VLM OCR"
PAGE_ICON = "📄"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=5)
def _check_server() -> dict | None:
    """Check if the OCR server is reachable."""
    try:
        resp = httpx.get(f"{OCR_SERVICE_URL}/health", timeout=5.0)
        return resp.json()
    except Exception:
        return None


def _ocr_upload(file_bytes: bytes, filename: str, fmt: str, prompt: str | None, max_tokens: int) -> dict:
    """Send an image to the OCR server via multipart upload."""
    files = {"file": (filename, file_bytes)}
    data = {"format": fmt, "max_tokens": str(max_tokens)}
    if prompt:
        data["prompt"] = prompt

    resp = httpx.post(
        f"{OCR_SERVICE_URL}/ocr/upload",
        files=files,
        data=data,
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()


def _ocr_pdf(file_bytes: bytes, filename: str, fmt: str, prompt: str | None, max_tokens: int, pages: str | None) -> dict:
    """Send a PDF to the OCR server."""
    files = {"file": (filename, file_bytes)}
    data = {"format": fmt, "max_tokens": str(max_tokens)}
    if prompt:
        data["prompt"] = prompt
    if pages:
        data["pages"] = pages

    resp = httpx.post(
        f"{OCR_SERVICE_URL}/ocr/pdf",
        files=files,
        data=data,
        timeout=300.0,
    )
    resp.raise_for_status()
    return resp.json()


def _format_elapsed(ms: float) -> str:
    """Format milliseconds as a human-readable string."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title(f"{PAGE_ICON} VLM OCR")
    st.caption("Powered by Qwen2.5-VL")

    # Server status
    status = _check_server()
    if status and status.get("status") == "ok":
        st.success(f"Server: {status.get('model', 'unknown')}")
    elif status and status.get("status") == "loading":
        st.warning("Server: model loading...")
    else:
        st.error(f"Server unreachable at {OCR_SERVICE_URL}")

    st.divider()

    # Output format
    output_format = st.selectbox(
        "Output format",
        ["text", "markdown", "json", "table"],
        index=0,
        help="How to structure the extracted text",
    )

    # Advanced options
    with st.expander("Advanced options"):
        max_tokens = st.slider(
            "Max output tokens",
            min_value=256,
            max_value=8192,
            value=4096,
            step=256,
            help="Maximum tokens in the OCR output",
        )

        custom_prompt = st.text_area(
            "Custom prompt (optional)",
            placeholder="Override the default OCR prompt...",
            help="Leave empty to use the default prompt for the selected format",
        )

        pdf_pages = st.text_input(
            "PDF pages (optional)",
            placeholder="e.g., 1-5 or 1,3,5",
            help="Which pages to OCR. Leave empty for all pages.",
        )

    st.divider()
    st.caption(f"Server: `{OCR_SERVICE_URL}`")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("Document OCR")
st.markdown("Upload images or PDFs to extract text using a Vision Language Model.")

# File upload
uploaded_files = st.file_uploader(
    "Upload images or PDFs",
    type=["png", "jpg", "jpeg", "webp", "tiff", "bmp", "gif", "pdf"],
    accept_multiple_files=True,
    help="Supported: PNG, JPG, WebP, TIFF, BMP, GIF, PDF",
)

if not uploaded_files:
    st.info("Upload one or more files to get started.")
    st.stop()

# Process button
run_ocr = st.button("Extract Text", type="primary", use_container_width=True)

if not run_ocr:
    # Show previews of uploaded files
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
# Run OCR
# ---------------------------------------------------------------------------
prompt = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else None

for uploaded_file in uploaded_files:
    file_bytes = uploaded_file.getvalue()
    is_pdf = uploaded_file.type == "application/pdf"

    st.divider()
    st.subheader(uploaded_file.name)

    # Show preview
    if not is_pdf:
        col_img, col_result = st.columns([1, 1])
        with col_img:
            img = Image.open(io.BytesIO(file_bytes))
            st.image(img, caption=f"{img.width}x{img.height}", use_container_width=True)
    else:
        col_result = st.container()
        st.caption(f"PDF ({len(file_bytes) / 1024:.0f} KB)")

    # Call OCR server
    with st.spinner(f"Processing {uploaded_file.name}..."):
        try:
            if is_pdf:
                result = _ocr_pdf(
                    file_bytes, uploaded_file.name, output_format,
                    prompt, max_tokens, pdf_pages or None,
                )
                # Display each page result
                with col_result:
                    total_ms = result.get("total_elapsed_ms", 0)
                    count = result.get("count", 0)
                    st.caption(f"{count} page(s) in {_format_elapsed(total_ms)}")

                    for i, page_result in enumerate(result.get("results", [])):
                        with st.expander(f"Page {i + 1} ({_format_elapsed(page_result['elapsed_ms'])})", expanded=(i == 0)):
                            if output_format == "markdown":
                                st.markdown(page_result["text"])
                            elif output_format == "json":
                                st.code(page_result["text"], language="json")
                            else:
                                st.text(page_result["text"])
            else:
                result = _ocr_upload(
                    file_bytes, uploaded_file.name, output_format,
                    prompt, max_tokens,
                )

                with col_result:
                    elapsed = result.get("elapsed_ms", 0)
                    st.caption(f"Extracted in {_format_elapsed(elapsed)}")

                    if output_format == "markdown":
                        st.markdown(result["text"])
                    elif output_format == "json":
                        st.code(result["text"], language="json")
                    elif output_format == "table":
                        st.markdown(result["text"])
                    else:
                        st.text(result["text"])

            # Copy button
            all_text = result.get("text", "") or "\n\n---\n\n".join(
                r["text"] for r in result.get("results", [])
            )
            st.download_button(
                f"Download {uploaded_file.name} result",
                data=all_text,
                file_name=f"{Path(uploaded_file.name).stem}_ocr.{'md' if output_format == 'markdown' else 'txt'}",
                mime="text/plain",
            )

        except httpx.HTTPStatusError as e:
            st.error(f"Server error: {e.response.status_code} - {e.response.text}")
        except httpx.ConnectError:
            st.error(f"Cannot connect to OCR server at {OCR_SERVICE_URL}")
        except Exception as e:
            st.error(f"Error: {e}")
