"""Build the index: python ingest.py"""
import logging
import sys

from multimodal_rag.chunking import chunk_blocks
from multimodal_rag.config import settings, setup_logging
from multimodal_rag.embeddings import CLIP_SPECIAL_TOKENS, ClipEmbedder
from multimodal_rag.extract import extract_pdf
from multimodal_rag.store import save_index

log = logging.getLogger("ingest")


def main():
    setup_logging()
    pdfs = sorted(settings.pdf_dir.glob("*.pdf"))
    if not pdfs:
        log.error("no PDF files in %s", settings.pdf_dir)
        return 1

    embedder = ClipEmbedder(settings.clip_model)
    all_chunks, all_figures = [], []
    for pdf in pdfs:
        blocks, figures = extract_pdf(pdf, settings)
        all_chunks += chunk_blocks(blocks, embedder.count_tokens, settings.chunk_max_tokens)
        all_figures += figures

    too_long = [c.chunk_id for c in all_chunks if embedder.count_tokens(c.text) > 77]
    if too_long:
        log.error("these chunks are longer than CLIP can read: %s", too_long)
        return 1
    if not all_chunks or not all_figures:
        log.error("nothing to index: %d chunks, %d figures", len(all_chunks), len(all_figures))
        return 1

    text_vectors = embedder.embed_texts([c.text for c in all_chunks])
    image_vectors = embedder.embed_images([f.image_path for f in all_figures])
    caption_vectors = embedder.embed_texts([f.caption for f in all_figures])
    save_index(settings.index_dir, all_chunks, all_figures, text_vectors, image_vectors, caption_vectors)
    log.info("index saved: %d text chunks, %d figures", len(all_chunks), len(all_figures))
    return 0


if __name__ == "__main__":
    sys.exit(main())
