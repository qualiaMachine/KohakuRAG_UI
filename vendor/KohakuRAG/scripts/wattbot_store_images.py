"""Render PDF pages as images and store them for multimodal retrieval.

Academic PDFs contain figures as vector graphics that can't be extracted
as embedded images. Instead, this script renders each page as a high-quality
image using PyMuPDF (fitz), then stores the page images in ImageStore.

JinaV4's multimodal embeddings handle the rest — text queries find
relevant page images via cross-modal search in the shared vector space.

Usage:
    cd vendor/KohakuRAG
    kogine run scripts/wattbot_store_images.py --config configs/jinav4/index.py

After this, build the image index for retrieval:
    kogine run scripts/wattbot_build_image_index.py --config configs/jinav4/image_index.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import fitz  # PyMuPDF
except ImportError:
    print(
        "ERROR: PyMuPDF (fitz) is required for page rendering.\n"
        "  Install with: pip install pymupdf",
        file=sys.stderr,
    )
    sys.exit(1)

from kohakurag.datastore import ImageStore
from kohakurag.parsers import dict_to_payload, payload_to_dict
from kohakurag.types import ParagraphPayload, SentencePayload

# ============================================================================
# GLOBAL CONFIGURATION (overridden by kogine config injection)
# ============================================================================

docs_dir = "../../data/corpus"
pdf_dir = "../../data/pdfs"
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"
limit = 0  # 0 = all documents

# Rendering settings
dpi = 150  # Resolution for page rendering (150 = good balance of quality vs size)
max_page_dim = 1536  # Max width/height in pixels after rendering
image_format = "jpeg"  # jpeg or png
jpeg_quality = 90


def render_pdf_pages(pdf_path: Path, doc_id: str) -> list[dict]:
    """Render each page of a PDF as a JPEG image.

    Returns list of dicts: {page_num, image_bytes, width, height}
    """
    doc = fitz.open(str(pdf_path))
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Render at target DPI
        zoom = dpi / 72.0  # PDF default is 72 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Scale down if too large
        w, h = pix.width, pix.height
        if max(w, h) > max_page_dim:
            scale = max_page_dim / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            # Re-render at lower zoom
            adjusted_zoom = zoom * scale
            mat = fitz.Matrix(adjusted_zoom, adjusted_zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            w, h = pix.width, pix.height

        # Convert to JPEG bytes
        if image_format == "jpeg":
            img_bytes = pix.tobytes(output="jpeg", jpg_quality=jpeg_quality)
        else:
            img_bytes = pix.tobytes(output="png")

        pages.append({
            "page_num": page_num + 1,  # 1-indexed
            "image_bytes": img_bytes,
            "width": w,
            "height": h,
        })

    doc.close()
    return pages


def process_document(
    json_path: Path, pdf_dir_path: Path, db_path: Path, idx: int, total: int
) -> dict:
    """Render PDF pages and store as images."""
    doc_id = json_path.stem
    pdf_path = pdf_dir_path / f"{doc_id}.pdf"
    stats = {"doc_id": doc_id, "pages_rendered": 0, "pages_stored": 0, "errors": 0}

    print(f"[{idx}/{total}] {doc_id}... ", end="", flush=True)

    if not pdf_path.exists():
        print("SKIP (no PDF)")
        return stats

    try:
        # Render all pages
        pages = render_pdf_pages(pdf_path, doc_id)
        stats["pages_rendered"] = len(pages)

        if not pages:
            print("no pages")
            return stats

        # Open image store
        image_store = ImageStore(db_path, table="image_blobs")

        # Load existing JSON payload to update metadata
        payload = dict_to_payload(json.loads(json_path.read_text(encoding="utf-8")))
        updated = False

        for page_info in pages:
            page_num = page_info["page_num"]
            img_bytes = page_info["image_bytes"]
            w, h = page_info["width"], page_info["height"]

            # Storage key: one image per page
            storage_key = f"img:{doc_id}:p{page_num}:full"

            # Skip if already stored
            existing = image_store._sync_get(storage_key)
            if existing:
                stats["pages_stored"] += 1
                continue

            # Store compressed page image
            image_store._kv[storage_key] = img_bytes
            stats["pages_stored"] += 1

            # Find or create image node in the document payload for this page
            for section in payload.sections or []:
                if section.metadata.get("page") != page_num:
                    continue

                # Check if a page-image paragraph already exists
                has_page_img = any(
                    p.metadata.get("attachment_type") == "page_image"
                    for p in section.paragraphs
                )
                if has_page_img:
                    # Update storage key if needed
                    for p in section.paragraphs:
                        if p.metadata.get("attachment_type") == "page_image":
                            if "image_storage_key" not in p.metadata:
                                p.metadata["image_storage_key"] = storage_key
                                updated = True
                    break

                # Add a page-image paragraph
                caption = f"[page_image:{doc_id} p{page_num} {w}x{h}] Full page render of {doc_id} page {page_num}"
                section.paragraphs.append(
                    ParagraphPayload(
                        text=caption,
                        sentences=[SentencePayload(text=caption)],
                        metadata={
                            "page": page_num,
                            "image_index": 0,
                            "image_name": f"page_{page_num}",
                            "image_width": w,
                            "image_height": h,
                            "attachment_type": "page_image",
                            "image_storage_key": storage_key,
                        },
                    )
                )
                updated = True
                break

        # Write updated JSON
        if updated:
            output_data = payload_to_dict(payload)
            json_path.write_text(
                json.dumps(output_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        stored = stats["pages_stored"]
        rendered = stats["pages_rendered"]
        size_mb = sum(len(p["image_bytes"]) for p in pages) / 1024 / 1024
        print(f"{stored}/{rendered} pages ({size_mb:.1f} MB)")

    except Exception as e:
        print(f"ERROR: {e}")
        stats["errors"] += 1

    return stats


async def main() -> None:
    docs_dir_path = Path(docs_dir)
    pdf_dir_path = Path(pdf_dir)
    db_path = Path(db)

    if not docs_dir_path.exists():
        print(f"ERROR: {docs_dir_path} not found")
        sys.exit(1)
    if not pdf_dir_path.exists():
        print(f"ERROR: {pdf_dir_path} not found")
        sys.exit(1)

    json_files = sorted(docs_dir_path.glob("*.json"))
    if limit > 0:
        json_files = json_files[:limit]

    print("=" * 60)
    print("KohakuRAG — Render & Store PDF Page Images")
    print("=" * 60)
    print(f"Documents:  {len(json_files)}")
    print(f"PDF dir:    {pdf_dir_path}")
    print(f"Database:   {db_path}")
    print(f"Resolution: {dpi} DPI (max {max_page_dim}px)")
    print(f"Format:     {image_format} (quality={jpeg_quality})")
    print("=" * 60)

    t0 = time.time()

    # Process sequentially (PyMuPDF isn't thread-safe for writes to same DB)
    results = []
    for i, jp in enumerate(json_files):
        result = process_document(jp, pdf_dir_path, db_path, i + 1, len(json_files))
        results.append(result)

    total_rendered = sum(r["pages_rendered"] for r in results)
    total_stored = sum(r["pages_stored"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Pages rendered: {total_rendered}")
    print(f"Pages stored:   {total_stored}")
    print(f"Errors:         {total_errors}")
    print(f"{'=' * 60}")

    if total_stored > 0:
        print(
            f"\nNext: rebuild text index to pick up page image nodes,\n"
            f"then build the image search index:\n\n"
            f"  kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py\n"
            f"  kogine run scripts/wattbot_build_image_index.py --config configs/jinav4/image_index.py\n"
        )


if __name__ == "__main__":
    asyncio.run(main())
