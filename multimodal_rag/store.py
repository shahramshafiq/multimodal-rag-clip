"""Step 4: keep the vectors and their details on disk and search them.

With only a few dozen vectors, comparing the question with all of them in NumPy is instant, so no vector database
is needed. The vectors are unit length, so a dot product is the cosine similarity.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

VECTORS_FILE = "vectors.npz"
ITEMS_FILE = "items.json"


@dataclass
class Index:
    chunks: list  # dicts: chunk_id, doc, page, text
    figures: list  # dicts: figure_id, doc, page, number, caption, image_path
    pages: dict  # "doc|page" -> the full text of that page
    text_vectors: np.ndarray  # text embedding model
    image_vectors: np.ndarray  # CLIP
    description_vectors: np.ndarray  # text embedding model


def page_key(doc, page):
    return f"{doc}|{page}"


def save_index(index_dir, chunks, figures, pages, text_vectors, image_vectors, description_vectors):
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    np.savez(index_dir / VECTORS_FILE, text=text_vectors, image=image_vectors, description=description_vectors)
    items = {"chunks": [asdict(chunk) for chunk in chunks], "figures": [asdict(figure) for figure in figures], "pages": pages}
    (index_dir / ITEMS_FILE).write_text(json.dumps(items, indent=2), encoding="utf-8")


def load_index(index_dir):
    index_dir = Path(index_dir)
    if not (index_dir / VECTORS_FILE).exists() or not (index_dir / ITEMS_FILE).exists():
        raise FileNotFoundError(f"No index in {index_dir}. Run: python ingest.py")
    items = json.loads((index_dir / ITEMS_FILE).read_text(encoding="utf-8"))
    vectors = np.load(index_dir / VECTORS_FILE)
    return Index(items["chunks"], items["figures"], items["pages"], vectors["text"], vectors["image"], vectors["description"])


def top_matches(scores, k):
    """[(position, score)] of the k highest scores, best first."""
    order = np.argsort(-scores)[:k]
    return [(int(position), float(scores[position])) for position in order]
