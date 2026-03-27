"""Extract figures from PDF pages for multimodal retrieval.

Detects actual figures (charts, plots, diagrams) in PDFs and crops them
with their captions — instead of storing full pages. Uses two strategies:

  1. Raster images: PyMuPDF's get_images() finds embedded images, then
     we expand the crop region to include nearby caption text.
  2. Vector figures: Finds "Figure N" / "Fig. N" caption text and crops
     the non-text region above it (for plots drawn as vector graphics).

Optionally runs a VLM verification pass on each extracted crop to:
  - Confirm it's actually a figure/table (not a logo, equation, etc.)
  - Generate a rich description for better retrieval
  - Classify the figure type (chart, diagram, table, photo, etc.)

Set vlm_verify=True and provide an OPENAI_API_KEY (for OpenRouter) to enable.

JinaV4's multimodal embeddings handle the rest — text queries find
relevant figures via cross-modal search in the shared vector space.

Usage:
    cd vendor/KohakuRAG
    kogine run scripts/wattbot_store_images.py --config configs/jinav4/index.py

After this, rebuild the text index and build the image index:
    kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
    kogine run scripts/wattbot_build_image_index.py --config configs/jinav4/image_index.py
"""

import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import fitz  # PyMuPDF
except ImportError:
    print(
        "ERROR: PyMuPDF (fitz) is required for figure extraction.\n"
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
dpi = 200  # Higher DPI for cropped figures (they're smaller than full pages)
max_figure_dim = 1536  # Max width/height in pixels for a cropped figure
image_format = "jpeg"  # jpeg or png
jpeg_quality = 92

# Figure detection settings
min_image_area_ratio = 0.03  # Embedded images must cover > 3% of page to count as figure
caption_expand_pt = 60  # Points to expand below image rect to capture caption
margin_pt = 10  # Points of padding around the crop region

# VLM verification settings
vlm_verify = False  # Set True to enable VLM-based verification of extracted crops
vlm_provider = "openrouter"  # "openrouter" (API) or "hf_local" (load from HuggingFace)
vlm_model = "qwen/qwen3-vl-235b-a22b-instruct"  # OpenRouter model ID
vlm_local_model = "Qwen/Qwen2.5-VL-72B-Instruct"  # HuggingFace model ID (for hf_local)
#   Options:  "Qwen/Qwen2.5-VL-72B-Instruct"  (~145 GB bf16, ~40 GB 4bit) — best quality
#             "Qwen/Qwen2.5-VL-32B-Instruct"  (~65 GB bf16, ~20 GB 4bit)  — strong, less VRAM
#             "Qwen/Qwen2.5-VL-7B-Instruct"   (~16 GB bf16, ~8 GB 4bit)   — lightweight fallback
vlm_local_dtype = "4bit"  # "bf16", "fp16", "4bit" (for hf_local)
vlm_max_concurrent = 5  # API concurrency (openrouter) or GPU concurrency (hf_local, recommend 1)

# Caption pattern for "Figure N" / "Fig. N" / "Table N" labels
_CAPTION_RE = re.compile(
    r"^(Fig(?:ure|\.)\s*\d+|Table\s*\d+)",
    re.IGNORECASE,
)

# VLM verification prompt — returns structured JSON
_VLM_VERIFY_PROMPT = """\
Analyze this cropped region from a scientific PDF. Respond ONLY with valid JSON matching this schema:

{
  "is_figure": true/false,
  "figure_type": "chart"|"diagram"|"table"|"photo"|"screenshot"|"other"|"not_a_figure",
  "caption": "the original caption text visible in the image, if any",
  "description": "2-3 sentence description of what this figure shows, focusing on data/content"
}

Rules:
- is_figure=true for charts, plots, tables, diagrams, architecture diagrams, photos, screenshots
- is_figure=false for logos, icons, decorative elements, page headers/footers, equations, author photos, watermarks
- For "caption": extract the exact caption text (e.g. "Figure 3: Comparison of...") if visible
- For "description": describe the actual content, data, trends, labels, axes — not the visual style
- If it's a table, list the column headers and key data points
"""


async def _vlm_verify_figure(
    image_bytes: bytes,
    vision_model,
) -> dict | None:
    """Use a VLM to verify if a crop is a real figure and generate a description.

    Returns a dict with {is_figure, figure_type, caption, description} or None on failure.
    """
    try:
        raw = await vision_model.caption(
            image_bytes,
            prompt=_VLM_VERIFY_PROMPT,
            max_tokens=400,
        )
        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        result = json.loads(raw)
        # Validate required fields
        if "is_figure" not in result:
            return None
        return result
    except (json.JSONDecodeError, Exception) as e:
        print(f"    VLM verify failed: {e}")
        return None


async def _vlm_verify_batch(
    figures: list[dict],
    vision_model,
) -> list[dict]:
    """Verify a batch of figure crops via VLM, filtering out non-figures.

    Returns only the figures that pass verification, with enriched metadata.
    """
    tasks = [_vlm_verify_figure(fig["image_bytes"], vision_model) for fig in figures]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    verified = []
    for fig, result in zip(figures, results):
        if isinstance(result, Exception):
            print(f"    VLM error for p{fig['page_num']}:fig{fig['fig_idx']}: {result}")
            # Keep figure on VLM failure (fail-open)
            verified.append(fig)
            continue

        if result is None:
            # VLM returned unparseable response — keep figure (fail-open)
            verified.append(fig)
            continue

        if not result.get("is_figure", True):
            fig_label = fig.get("fig_label", f"p{fig['page_num']}:fig{fig['fig_idx']}")
            print(f"    VLM rejected: {fig_label} ({result.get('figure_type', '?')})")
            continue

        # Enrich figure metadata with VLM results
        fig["vlm_verified"] = True
        fig["figure_type"] = result.get("figure_type", "")
        fig["vlm_description"] = result.get("description", "")
        # Use VLM-extracted caption if we didn't have one from regex
        if result.get("caption") and not fig.get("caption"):
            fig["caption"] = result["caption"]
        # If VLM found a caption and ours was just a placeholder, prefer VLM's
        if result.get("caption") and fig.get("caption", "").startswith("Image on page"):
            fig["caption"] = result["caption"]

        verified.append(fig)

    return verified


def _find_figure_regions(page) -> list[dict]:
    """Find figure regions on a page by combining image rects and caption detection.

    Returns list of dicts: {rect, caption, fig_label, source}
    where rect is a fitz.Rect and source is 'raster' or 'caption'.
    """
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height
    if page_area == 0:
        return []

    figures = []
    used_image_xrefs = set()

    # --- Strategy 1: Find text captions ("Figure N", "Fig. N", "Table N") ---
    text_blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, type)
    text_blocks_sorted = sorted(
        [b for b in text_blocks if b[6] == 0],  # type 0 = text
        key=lambda b: b[1],  # sort by y0 (top to bottom)
    )

    caption_blocks = []
    for block in text_blocks_sorted:
        text = block[4].strip()
        m = _CAPTION_RE.match(text)
        if m:
            caption_blocks.append({
                "rect": fitz.Rect(block[0], block[1], block[2], block[3]),
                "text": text,
                "label": m.group(1),
            })

    # For each caption, find the figure region above it
    for i, cap in enumerate(caption_blocks):
        cap_rect = cap["rect"]

        # Figure region: from some boundary above the caption to the caption bottom.
        # The top boundary is either:
        #   - The bottom of the previous text block that ISN'T part of a figure
        #   - Or the top of the page (for the first figure)
        # We look for the nearest text block above the caption that's NOT another caption.
        top_y = page_rect.y0
        for block in reversed(text_blocks_sorted):
            block_bottom = block[3]
            block_text = block[4].strip()
            # Must be above the caption with some gap (at least 20pt of figure space)
            if block_bottom < cap_rect.y0 - 20 and not _CAPTION_RE.match(block_text):
                top_y = block_bottom
                break

        # Build figure rect: full page width, from top boundary to caption bottom
        fig_rect = fitz.Rect(
            page_rect.x0,
            top_y,
            page_rect.x1,
            cap_rect.y1 + margin_pt,
        )

        # Check if this region contains a raster image
        images = page.get_images(full=True)
        for img_info in images:
            xref = img_info[0]
            try:
                for img_rect in page.get_image_rects(xref):
                    if fig_rect.intersects(img_rect):
                        used_image_xrefs.add(xref)
            except Exception:
                pass

        # Only keep if the figure region is tall enough to be real content
        fig_height = fig_rect.y1 - fig_rect.y0
        if fig_height > 50:  # at least ~50pt tall
            # Truncate caption text for storage
            caption_text = cap["text"]
            if len(caption_text) > 300:
                caption_text = caption_text[:300] + "..."
            figures.append({
                "rect": fig_rect,
                "caption": caption_text,
                "fig_label": cap["label"],
                "source": "caption",
            })

    # --- Strategy 2: Large raster images not already claimed by a caption ---
    images = page.get_images(full=True)
    for img_info in images:
        xref = img_info[0]
        if xref in used_image_xrefs:
            continue

        try:
            img_rects = page.get_image_rects(xref)
        except Exception:
            continue

        for img_rect in img_rects:
            img_area = img_rect.width * img_rect.height
            if img_area / page_area < min_image_area_ratio:
                continue  # Too small (icon, logo, etc.)

            # Expand downward to capture potential caption
            expanded = fitz.Rect(
                img_rect.x0 - margin_pt,
                img_rect.y0 - margin_pt,
                img_rect.x1 + margin_pt,
                img_rect.y1 + caption_expand_pt,
            )

            # Find caption text in the expanded region
            caption = ""
            for block in text_blocks_sorted:
                block_rect = fitz.Rect(block[0], block[1], block[2], block[3])
                if expanded.intersects(block_rect):
                    block_text = block[4].strip()
                    if _CAPTION_RE.match(block_text):
                        caption = block_text[:300]
                        # Extend crop to include full caption block
                        expanded.y1 = max(expanded.y1, block[3] + margin_pt)
                        break

            figures.append({
                "rect": expanded,
                "caption": caption or f"Image on page (area {img_area / page_area:.0%})",
                "fig_label": "",
                "source": "raster",
            })

    return figures


def _render_crop(page, rect: fitz.Rect) -> tuple[bytes, int, int]:
    """Render a cropped region of a page as JPEG bytes.

    Returns (image_bytes, width, height).
    """
    # Clip to page bounds
    clip = rect & page.rect

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)

    w, h = pix.width, pix.height
    if max(w, h) > max_figure_dim:
        scale = max_figure_dim / max(w, h)
        adjusted_zoom = zoom * scale
        mat = fitz.Matrix(adjusted_zoom, adjusted_zoom)
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        w, h = pix.width, pix.height

    if image_format == "jpeg":
        img_bytes = pix.tobytes(output="jpeg", jpg_quality=jpeg_quality)
    else:
        img_bytes = pix.tobytes(output="png")

    return img_bytes, w, h


def extract_figures(pdf_path: Path, doc_id: str) -> list[dict]:
    """Extract individual figures from a PDF.

    Returns list of dicts: {page_num, fig_idx, image_bytes, width, height, caption, fig_label}
    """
    doc = fitz.open(str(pdf_path))
    all_figures = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        regions = _find_figure_regions(page)

        for fig_idx, region in enumerate(regions):
            img_bytes, w, h = _render_crop(page, region["rect"])
            all_figures.append({
                "page_num": page_num + 1,  # 1-indexed
                "fig_idx": fig_idx,
                "image_bytes": img_bytes,
                "width": w,
                "height": h,
                "caption": region["caption"],
                "fig_label": region["fig_label"],
            })

    doc.close()
    return all_figures


async def process_document(
    json_path: Path,
    pdf_dir_path: Path,
    db_path: Path,
    idx: int,
    total: int,
    vision_model=None,
) -> dict:
    """Extract figures from a PDF and store them.

    If vision_model is provided (vlm_verify=True), each crop is sent to the VLM
    for verification and captioning before storage.
    """
    doc_id = json_path.stem
    pdf_path = pdf_dir_path / f"{doc_id}.pdf"
    stats = {
        "doc_id": doc_id,
        "figures_found": 0,
        "figures_stored": 0,
        "vlm_rejected": 0,
        "errors": 0,
    }

    print(f"[{idx}/{total}] {doc_id}... ", end="", flush=True)

    if not pdf_path.exists():
        print("SKIP (no PDF)")
        return stats

    try:
        figures = extract_figures(pdf_path, doc_id)
        stats["figures_found"] = len(figures)

        if not figures:
            print("no figures found")
            return stats

        # VLM verification pass — filter out non-figures, enrich captions
        if vision_model is not None:
            before_count = len(figures)
            figures = await _vlm_verify_batch(figures, vision_model)
            stats["vlm_rejected"] = before_count - len(figures)

        if not figures:
            print(f"0 figures after VLM verification ({stats['vlm_rejected']} rejected)")
            return stats

        # Open image store
        image_store = ImageStore(db_path, table="image_blobs")

        # Load existing JSON payload to update metadata
        payload = dict_to_payload(json.loads(json_path.read_text(encoding="utf-8")))
        updated = False

        # Extract document-level metadata for source links
        doc_meta = payload.metadata or {}
        source_url = doc_meta.get("url", "")
        source_title = payload.title or doc_id

        for fig in figures:
            page_num = fig["page_num"]
            fig_idx = fig["fig_idx"]
            img_bytes = fig["image_bytes"]
            w, h = fig["width"], fig["height"]
            caption = fig["caption"]
            fig_label = fig["fig_label"]

            # Storage key: unique per figure
            storage_key = f"img:{doc_id}:p{page_num}:fig{fig_idx}"

            # Skip if already stored
            existing = image_store._sync_get(storage_key)
            if existing:
                stats["figures_stored"] += 1
                continue

            # Store cropped figure image
            image_store._kv[storage_key] = img_bytes
            stats["figures_stored"] += 1

            # Find the matching section in the document payload
            for section in payload.sections or []:
                if section.metadata.get("page") != page_num:
                    continue

                # Check if this figure is already in the section
                has_this_fig = any(
                    p.metadata.get("image_storage_key") == storage_key
                    for p in section.paragraphs
                )
                if has_this_fig:
                    break

                # Build descriptive caption for embedding — prefer VLM description
                vlm_desc = fig.get("vlm_description", "")
                if fig_label:
                    embed_caption = f"[{fig_label}] {caption}"
                else:
                    embed_caption = f"[figure:{doc_id} p{page_num}] {caption}"
                if vlm_desc:
                    embed_caption += f" — {vlm_desc}"

                section.paragraphs.append(
                    ParagraphPayload(
                        text=embed_caption,
                        sentences=[SentencePayload(text=embed_caption)],
                        metadata={
                            "page": page_num,
                            "image_index": fig_idx,
                            "image_name": fig_label or f"fig_p{page_num}_{fig_idx}",
                            "image_width": w,
                            "image_height": h,
                            "attachment_type": "page_image",
                            "image_storage_key": storage_key,
                            # Structured caption metadata
                            "caption_text": caption,
                            "vlm_description": vlm_desc,
                            "figure_type": fig.get("figure_type", ""),
                            "vlm_verified": fig.get("vlm_verified", False),
                            # Source link metadata
                            "source_url": source_url,
                            "source_title": source_title,
                            "document_id": doc_id,
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

        found = stats["figures_found"]
        stored = stats["figures_stored"]
        rejected = stats["vlm_rejected"]
        size_mb = sum(len(f["image_bytes"]) for f in figures) / 1024 / 1024
        vlm_info = f", {rejected} VLM-rejected" if rejected else ""
        print(f"{stored}/{found} figures ({size_mb:.1f} MB{vlm_info})")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
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

    # Initialize VLM for verification if enabled
    vision_model = None
    if vlm_verify:
        if vlm_provider == "hf_local":
            try:
                from kohakurag.vision import HuggingFaceLocalVisionModel
                vision_model = HuggingFaceLocalVisionModel(
                    model=vlm_local_model,
                    dtype=vlm_local_dtype,
                    max_concurrent=vlm_max_concurrent,
                )
                print(f"VLM verification: ENABLED (local: {vlm_local_model})")
            except (ImportError, ValueError) as e:
                print(f"WARNING: Local VLM requested but unavailable: {e}")
                print("Falling back to heuristic-only extraction.")
        else:
            try:
                from kohakurag.vision import OpenAIVisionModel
                vision_model = OpenAIVisionModel(
                    model=vlm_model,
                    max_concurrent=vlm_max_concurrent,
                )
                print(f"VLM verification: ENABLED (API: {vlm_model})")
            except (ImportError, ValueError) as e:
                print(f"WARNING: VLM API requested but unavailable: {e}")
                print("Falling back to heuristic-only extraction.")

    print("=" * 60)
    print("KohakuRAG — Extract & Store PDF Figures")
    print("=" * 60)
    print(f"Documents:  {len(json_files)}")
    print(f"PDF dir:    {pdf_dir_path}")
    print(f"Database:   {db_path}")
    print(f"Resolution: {dpi} DPI (max {max_figure_dim}px)")
    print(f"Format:     {image_format} (quality={jpeg_quality})")
    if vision_model:
        model_name = vlm_local_model if vlm_provider == "hf_local" else vlm_model
        print(f"VLM verify: ON ({vlm_provider}: {model_name})")
    else:
        print(f"VLM verify: OFF")
    print("=" * 60)

    t0 = time.time()

    # Process sequentially (PyMuPDF isn't thread-safe for writes to same DB)
    results = []
    for i, jp in enumerate(json_files):
        result = await process_document(
            jp, pdf_dir_path, db_path, i + 1, len(json_files),
            vision_model=vision_model,
        )
        results.append(result)

    total_found = sum(r["figures_found"] for r in results)
    total_stored = sum(r["figures_stored"] for r in results)
    total_rejected = sum(r["vlm_rejected"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Figures found:   {total_found}")
    print(f"Figures stored:  {total_stored}")
    if total_rejected:
        print(f"VLM rejected:    {total_rejected}")
    print(f"Errors:          {total_errors}")
    print(f"{'=' * 60}")

    if total_stored > 0:
        print(
            f"\nNext: rebuild text index to pick up figure nodes,\n"
            f"then build the image search index:\n\n"
            f"  kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py\n"
            f"  kogine run scripts/wattbot_build_image_index.py --config configs/jinav4/image_index.py\n"
        )


if __name__ == "__main__":
    asyncio.run(main())
