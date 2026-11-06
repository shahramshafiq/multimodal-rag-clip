"""Step 5: find the text chunks and the figures closest to a question.

Text and figures are ranked in two separate lists, because their scores are not on the same scale.

A figure is found in two ways: CLIP compares the question with the picture, and the text embedding model compares it
with the figure's description. CLIP scores run about 0.3 and text scores about 0.6, so adding them would let text
decide everything. Reciprocal rank fusion uses only each figure's place in each list, so scale does not matter.

Chunks are only used to search. The model gets the whole page each matching chunk came from.
"""
from dataclasses import dataclass, field

import numpy as np

from .store import page_key, top_matches


@dataclass
class Retrieved:
    chunks: list  # chunk dicts with a "score"
    figures: list  # figure dicts with a "score"
    pages: list = field(default_factory=list)  # dicts: doc, page, text. The pages of the chunks, best first, no repeats


def _pages_of(chunks, index):
    pages, seen = [], set()
    for chunk in chunks:
        key = page_key(chunk["doc"], chunk["page"])
        if key not in seen:
            seen.add(key)
            pages.append({"doc": chunk["doc"], "page": chunk["page"], "text": index.pages[key]})
    return pages


def fuse_ranks(score_lists, k):
    """Reciprocal rank fusion: each list gives an item 1 / (k + its place in that list)."""
    fused = np.zeros(len(score_lists[0]))
    for scores in score_lists:
        places = np.empty(len(scores), dtype=int)
        places[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
        fused += 1.0 / (k + places)
    return fused


def figure_scores(query_clip, query_text, index, mode, rrf_k):
    by_image = index.image_vectors @ query_clip
    by_description = index.description_vectors @ query_text
    if mode == "image":
        return by_image
    if mode == "description":
        return by_description
    return fuse_ranks([by_image, by_description], rrf_k)


def retrieve(question, index, clip, text_embedder, settings, top_k_text=None, top_k_figures=None, figure_mode=None):
    top_k_text = top_k_text or settings.top_k_text
    top_k_figures = top_k_figures or settings.top_k_figures
    query_text = text_embedder.embed_texts([question])[0]
    query_clip = clip.embed_texts([question])[0]

    text_scores = index.text_vectors @ query_text
    chunks = [{**index.chunks[i], "score": score} for i, score in top_matches(text_scores, top_k_text)]

    scores = figure_scores(query_clip, query_text, index, figure_mode or settings.figure_search, settings.rrf_k)
    figures = [{**index.figures[i], "score": score} for i, score in top_matches(scores, top_k_figures)]

    return Retrieved(chunks, figures, _pages_of(chunks, index))
