"""Step 5: find the text chunks and the figures closest to a question.

Text and figures are ranked in two separate lists. In CLIP, text-to-text scores are usually much higher than
text-to-image scores, so one shared list would almost always push the figures out.
"""
from dataclasses import dataclass

import numpy as np

from .store import top_matches


@dataclass
class Retrieved:
    chunks: list  # chunk dicts with a "score"
    figures: list  # figure dicts with a "score"


def retrieve(question, index, embedder, top_k_text, top_k_figures, use_captions):
    query = embedder.embed_texts([question])[0]

    text_scores = index.text_vectors @ query
    chunks = [{**index.chunks[i], "score": score} for i, score in top_matches(text_scores, top_k_text)]

    figure_scores = index.image_vectors @ query
    if use_captions:
        # a figure counts as close when either its picture or its caption is close
        figure_scores = np.maximum(figure_scores, index.caption_vectors @ query)
    figures = [{**index.figures[i], "score": score} for i, score in top_matches(figure_scores, top_k_figures)]

    return Retrieved(chunks, figures)
