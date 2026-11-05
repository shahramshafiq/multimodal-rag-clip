"""Reads every setting from .env, and from .env.example for anything .env does not mention."""
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.example")  # does not overwrite values already set


def _text(name):
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ValueError(f"The setting {name} is missing. See .env.example.")
    return value.strip()


def _flag(name):
    return _text(name).lower() in ("true", "1", "yes")


@dataclass(frozen=True)
class Settings:
    clip_model: str
    openai_model: str
    openai_timeout_seconds: float
    openai_max_retries: int
    image_detail: str
    answer_max_tokens: int
    pdf_dir: Path
    index_dir: Path
    figure_dir: Path
    golden_file: Path
    results_dir: Path
    stop_headings: tuple
    min_block_chars: int
    min_figure_part_size: float
    figure_merge_gap: float
    figure_caption_gap: float
    figure_dpi: int
    chunk_max_tokens: int
    top_k_text: int
    top_k_figures: int
    use_figure_captions: bool
    not_found_sentence: str


settings = Settings(
    clip_model=_text("CLIP_MODEL"),
    openai_model=_text("OPENAI_MODEL"),
    openai_timeout_seconds=float(_text("OPENAI_TIMEOUT_SECONDS")),
    openai_max_retries=int(_text("OPENAI_MAX_RETRIES")),
    image_detail=_text("IMAGE_DETAIL"),
    answer_max_tokens=int(_text("ANSWER_MAX_TOKENS")),
    pdf_dir=ROOT / _text("PDF_DIR"),
    index_dir=ROOT / _text("INDEX_DIR"),
    figure_dir=ROOT / _text("FIGURE_DIR"),
    golden_file=ROOT / _text("GOLDEN_FILE"),
    results_dir=ROOT / _text("RESULTS_DIR"),
    stop_headings=tuple(part.strip().lower() for part in _text("STOP_HEADINGS").split(",")),
    min_block_chars=int(_text("MIN_BLOCK_CHARS")),
    min_figure_part_size=float(_text("MIN_FIGURE_PART_SIZE")),
    figure_merge_gap=float(_text("FIGURE_MERGE_GAP")),
    figure_caption_gap=float(_text("FIGURE_CAPTION_GAP")),
    figure_dpi=int(_text("FIGURE_DPI")),
    chunk_max_tokens=int(_text("CHUNK_MAX_TOKENS")),
    top_k_text=int(_text("TOP_K_TEXT")),
    top_k_figures=int(_text("TOP_K_FIGURES")),
    use_figure_captions=_flag("USE_FIGURE_CAPTIONS"),
    not_found_sentence=_text("NOT_FOUND_SENTENCE"),
)


def setup_logging():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # the Windows console cannot print every PDF character
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "openai", "transformers", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
