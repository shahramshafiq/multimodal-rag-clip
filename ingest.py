"""Build the index: python ingest.py"""
import logging
import sys

from multimodal_rag.chunking import chunk_blocks
from multimodal_rag.config import settings, setup_logging
from multimodal_rag.describe import load_descriptions
from multimodal_rag.embeddings import ClipEmbedder, TextEmbedder
from multimodal_rag.extract import extract_pdf
from multimodal_rag.store import page_key, save_index

log = logging.getLogger("ingest")


def main():
    setup_logging()
    pdfs = sorted(settings.pdf_dir.glob("*.pdf"))
    if not pdfs:
        log.error("no PDF files in %s", settings.pdf_dir)
        return 1

    try:
        clip = ClipEmbedder(settings.clip_model)
        text_embedder = TextEmbedder(settings)
        all_chunks, all_figures, pages = [], [], {}
        for pdf in pdfs:
            blocks, figures = extract_pdf(pdf, settings)
            all_chunks += chunk_blocks(blocks, clip.count_tokens, settings.chunk_max_tokens)
            all_figures += figures
            for block in blocks:
                key = page_key(block.doc, block.page)
                pages[key] = f"{pages[key]}\n{block.text}" if key in pages else block.text

        if not all_chunks or not all_figures:
            log.error("nothing to index: %d chunks, %d figures", len(all_chunks), len(all_figures))
            return 1

        descriptions = load_descriptions(all_figures, settings)
        text_vectors = text_embedder.embed_texts([c.text for c in all_chunks])
        image_vectors = clip.embed_images([f.image_path for f in all_figures])
        description_vectors = text_embedder.embed_texts([descriptions[f.figure_id] for f in all_figures])
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1

    save_index(settings.index_dir, all_chunks, all_figures, pages, text_vectors, image_vectors, description_vectors)
    log.info("index saved: %d text chunks, %d pages, %d figures", len(all_chunks), len(pages), len(all_figures))
    return 0


if __name__ == "__main__":
    sys.exit(main())
