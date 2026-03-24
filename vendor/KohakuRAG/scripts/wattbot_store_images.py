"""Extract and store compressed images from parsed PDFs into ImageStore.

This is a lightweight alternative to wattbot_add_image_captions.py that
stores images WITHOUT requiring a vision LLM for captioning. Images get
basic auto-generated captions from PDF metadata instead.

After running this, images will be:
1. Stored as compressed JPEG blobs in ImageStore (same DB as vectors)
2. Retrievable by the Streamlit UI via image_storage_key metadata
3. Optionally embeddable via wattbot_build_image_index.py for image search

Usage:
    cd vendor/KohakuRAG
    kogine run scripts/wattbot_store_images.py --config configs/jinav4/index.py
"""

import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pypdf import PdfReader

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kohakurag.datastore import ImageStore
from kohakurag.image_utils import compress_image, get_image_dimensions
from kohakurag.parsers import dict_to_payload, payload_to_dict
from kohakurag.pdf_utils import _extract_images
from kohakurag.types import SentencePayload

# ============================================================================
# GLOBAL CONFIGURATION (overridden by kogine config injection)
# ============================================================================

docs_dir = "../../data/corpus"
pdf_dir = "../../data/pdfs"
db = "../../data/embeddings/wattbot_jinav4.db"
limit = 0  # 0 = all documents


def process_document(
    json_path: Path, pdf_dir_path: Path, db_path: Path, idx: int, total: int
) -> dict:
    """Extract, compress, and store images from one document."""
    doc_id = json_path.stem
    pdf_path = pdf_dir_path / f"{doc_id}.pdf"
    stats = {"doc_id": doc_id, "images_found": 0, "images_stored": 0, "errors": 0}

    print(f"[{idx}/{total}] {doc_id}... ", end="", flush=True)

    if not pdf_path.exists():
        print("SKIP (no PDF)")
        return stats

    try:
        payload = dict_to_payload(json.loads(json_path.read_text(encoding="utf-8")))
        reader = PdfReader(str(pdf_path))
        image_store = ImageStore(db_path, table="image_blobs")
        updated = False

        for section in payload.sections or []:
            page_num = section.metadata.get("page", 1)
            if page_num < 1 or page_num > len(reader.pages):
                continue

            page = reader.pages[page_num - 1]
            try:
                images = _extract_images(page)
            except Exception:
                continue

            if not images:
                continue

            image_lookup = {i: img for i, img in enumerate(images, 1)}

            for para in section.paragraphs:
                if para.metadata.get("attachment_type") != "image":
                    continue

                img_idx = para.metadata.get("image_index")
                img_info = image_lookup.get(img_idx)

                if not img_info or not img_info.get("data"):
                    continue

                stats["images_found"] += 1
                storage_key = ImageStore.make_key(doc_id, page_num, img_idx)

                # Skip if already stored
                if image_store._sync_get(storage_key):
                    stats["images_stored"] += 1
                    # Still update metadata if missing storage_key
                    if "image_storage_key" not in para.metadata:
                        para.metadata["image_storage_key"] = storage_key
                        updated = True
                    continue

                try:
                    # Compress image
                    compressed = compress_image(
                        img_info["data"],
                        max_size=1024,
                        format="jpeg",
                        quality=95,
                    )

                    dims = get_image_dimensions(compressed)
                    width, height = dims if dims else ("?", "?")

                    # Store in ImageStore
                    image_store._kv[storage_key] = compressed

                    # Update paragraph metadata with storage key
                    para.metadata["image_storage_key"] = storage_key
                    para.metadata["compressed_width"] = width if isinstance(width, int) else None
                    para.metadata["compressed_height"] = height if isinstance(height, int) else None

                    # Update caption with dimensions
                    img_name = para.metadata.get("image_name", f"img{img_idx}")
                    caption = f"[img:{img_name} {width}x{height}] Figure from {doc_id} page {page_num}"
                    para.text = caption
                    para.sentences = [SentencePayload(text=caption)]

                    stats["images_stored"] += 1
                    updated = True

                except Exception as e:
                    stats["errors"] += 1

        # Write updated JSON back if we added storage keys
        if updated:
            output_data = payload_to_dict(payload)
            json_path.write_text(
                json.dumps(output_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        stored = stats["images_stored"]
        found = stats["images_found"]
        if found > 0:
            print(f"stored {stored}/{found} images")
        else:
            print("no images")

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
    print("KohakuRAG — Extract & Store PDF Images")
    print("=" * 60)
    print(f"Documents: {len(json_files)}")
    print(f"PDF dir:   {pdf_dir_path}")
    print(f"Database:  {db_path}")
    print("=" * 60)

    t0 = time.time()
    executor = ThreadPoolExecutor(max_workers=8)
    loop = asyncio.get_event_loop()

    tasks = [
        loop.run_in_executor(
            executor,
            process_document,
            jp, pdf_dir_path, db_path, i + 1, len(json_files),
        )
        for i, jp in enumerate(json_files)
    ]

    results = await asyncio.gather(*tasks)

    total_found = sum(r["images_found"] for r in results)
    total_stored = sum(r["images_stored"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Images found:  {total_found}")
    print(f"Images stored: {total_stored}")
    print(f"Errors:        {total_errors}")
    print(f"{'=' * 60}")

    if total_stored > 0:
        print(
            f"\nNext step: rebuild the text index to pick up updated image metadata:\n"
            f"  kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py\n"
            f"\nOptional: build dedicated image search index:\n"
            f"  kogine run scripts/wattbot_build_image_index.py --config configs/jinav4/index.py\n"
        )


if __name__ == "__main__":
    asyncio.run(main())
