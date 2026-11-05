"""Step 5: find the text chunks and the figures closest to a question.

Text and figures are ranked in two separate lists. In CLIP, text-to-text scores are usually much higher than
text-to-image scores, so one shared list would almost always push the figures out.

Chunks are only used to search. CLIP reads at most 77 tokens, so a chunk is small and often cut off from its
context, and the model gets the whole page each matching chunk came from.
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


def retrieve(question, index, embedder, top_k_text, top_k_figures, use_captions):
    query = embedder.embed_texts([question])[0]

    text_scores = index.text_vectors @ query
    chunks = [{**index.chunks[i], "score": score} for i, score in top_matches(text_scores, top_k_text)]

    figure_scores = index.image_vectors @ query
    if use_captions:
        # a figure counts as close when either its picture or its caption is close
        figure_scores = np.maximum(figure_scores, index.caption_vectors @ query)
    figures = [{**index.figures[i], "score": score} for i, score in top_matches(figure_scores, top_k_figures)]

    return Retrieved(chunks, figures, _pages_of(chunks, index))
