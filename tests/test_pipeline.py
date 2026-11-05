"""Small checks that need no OpenAI key and no CLIP download. Run with: python -m pytest"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multimodal_rag.answer import build_content
from multimodal_rag.chunking import chunk_block
from multimodal_rag.config import settings
from multimodal_rag.evaluation import answer_is_correct, hit_rate, retrieval_hit_rank, term_found
from multimodal_rag.extract import extract_pdf
from multimodal_rag.retrieve import Retrieved, retrieve
from multimodal_rag.store import Index, top_matches

COUNT_WORDS = lambda text: len(text.split())


def test_chunks_stay_under_the_limit_and_keep_sentences_whole():
    text = "One two three. Four five six. Seven eight nine."
    assert chunk_block(text, COUNT_WORDS, 6) == ["One two three. Four five six.", "Seven eight nine."]


def test_a_sentence_that_is_too_long_is_cut_between_words():
    chunks = chunk_block(" ".join(["word"] * 25), COUNT_WORDS, 10)
    assert [COUNT_WORDS(chunk) for chunk in chunks] == [10, 10, 5]


@pytest.fixture(scope="module")
def extracted(tmp_path_factory):
    test_settings = replace(settings, figure_dir=tmp_path_factory.mktemp("figures"))
    results = {pdf.stem: extract_pdf(pdf, test_settings) for pdf in settings.pdf_dir.glob("*.pdf")}
    return results


def test_every_figure_is_found_including_charts_drawn_as_vector_shapes(extracted):
    numbers = {doc: sorted(figure.number for figure in figures) for doc, (_, figures) in extracted.items()}
    assert numbers == {"cdc_long_covid_children_2022": [1, 2, 3], "noaa_sea_ice_2020": [1, 2, 3, 4, 5]}
    for _, figures in extracted.values():
        assert all(Path(figure.image_path).stat().st_size > 10_000 for figure in figures)


def test_chart_labels_captions_headers_and_references_are_not_in_the_text(extracted):
    all_text = " ".join(block.text for blocks, _ in extracted.values() for block in blocks)
    assert "Total Boys Girls" not in all_text  # a label inside a chart
    assert "Figure 1. Percentage" not in all_text  # a caption is kept with its figure
    assert "NCHS Data Brief" not in all_text  # a header repeated on every page
    assert "Cavalieri" not in all_text  # the first reference


class FakeEmbedder:
    def embed_texts(self, texts):
        return np.array([[1.0, 0.0]], dtype=np.float32)


def make_index():
    return Index(
        chunks=[{"chunk_id": "a", "doc": "d", "page": 1, "text": "a"}, {"chunk_id": "b", "doc": "d", "page": 2, "text": "b"}],
        figures=[{"figure_id": "image_match"}, {"figure_id": "caption_match"}],
        text_vectors=np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float32),
        image_vectors=np.array([[0.3, 0.9], [0.1, 0.9]], dtype=np.float32),
        caption_vectors=np.array([[0.0, 1.0], [0.8, 0.2]], dtype=np.float32),
    )


def test_top_matches_returns_the_best_first():
    assert [position for position, _ in top_matches(np.array([0.1, 0.9, 0.5]), 2)] == [1, 2]


def test_figures_get_their_own_list_even_when_text_scores_are_higher():
    found = retrieve("question", make_index(), FakeEmbedder(), 1, 1, use_captions=False)
    assert found.chunks[0]["chunk_id"] == "b"
    assert found.figures[0]["figure_id"] == "image_match"


def test_a_caption_can_lift_a_figure():
    found = retrieve("question", make_index(), FakeEmbedder(), 1, 1, use_captions=True)
    assert found.figures[0]["figure_id"] == "caption_match"


def test_term_matching_needs_the_whole_term():
    assert term_found("3", "the axis ends at 3.") and not term_found("3", "it was 13")
    assert term_found("red", "It is RED") and not term_found("red", "it reduced")
    assert term_found("0.5", "about 0.5%") and not term_found("0.5", "it was 10.5")
    assert term_found("pink|magenta", "a magenta line")


def test_answer_scoring():
    assert answer_is_correct({"type": "text", "must_include": ["1.06", "3.43"]}, "from 1.06 to 3.43 m", "none")
    assert not answer_is_correct({"type": "text", "must_include": ["1.06", "3.43"]}, "about 1.06 m", "none")
    assert answer_is_correct({"type": "none"}, "I could not find this.", "I could not find this.")


def test_retrieval_rank_and_hit_rate():
    found = Retrieved(chunks=[{"doc": "d", "page": 1}, {"doc": "d", "page": 4}], figures=[{"figure_id": "f"}])
    assert retrieval_hit_rank({"type": "text", "doc": "d", "pages": [4]}, found) == 2
    assert retrieval_hit_rank({"type": "figure", "figure": "x"}, found) is None
    rows = [{"type": "text", "rank": 1}, {"type": "text", "rank": 3}, {"type": "text", "rank": None}]
    assert hit_rate(rows, "text", 2) == (1, 3)


def test_the_model_receives_the_picture_and_its_caption(tmp_path):
    picture = tmp_path / "chart.png"
    picture.write_bytes(b"png")
    found = Retrieved(chunks=[], figures=[{"doc": "d", "number": 2, "page": 4, "caption": "Fig. 2.", "image_path": str(picture)}])
    content = build_content("question", found, "high")
    assert content[-1]["type"] == "input_image" and content[-1]["image_url"].startswith("data:image/png;base64,")
    assert "Fig. 2." in content[-2]["text"]
