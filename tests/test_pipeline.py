"""Small checks that need no OpenAI key and no CLIP download. Run with: python -m pytest"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multimodal_rag.answer import build_content, unknown_sources
from multimodal_rag.chunking import chunk_block
from multimodal_rag.config import settings
from multimodal_rag.evaluation import answer_is_correct, hit_rate, number_found, retrieval_hit_rank, term_found
from multimodal_rag.extract import extract_pdf
from multimodal_rag.retrieve import Retrieved, fuse_ranks, retrieve
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
    assert "Comiso, J.C." not in all_text  # an entry in the reference list


def test_a_footnote_stays_with_its_figure_and_leaves_the_text(extracted):
    blocks, figures = extracted["cdc_long_covid_children_2022"]
    figure_3 = next(figure for figure in figures if figure.number == 3)
    assert "does not meet National Center for Health Statistics standards" in figure_3.notes
    assert not any("SOURCE:" in block.text for block in blocks)


class FakeEmbedder:
    def embed_texts(self, texts):
        return np.array([[1.0, 0.0]], dtype=np.float32)


def make_index():
    return Index(
        chunks=[{"chunk_id": "a", "doc": "d", "page": 1, "text": "a"}, {"chunk_id": "b", "doc": "d", "page": 2, "text": "b"}],
        figures=[{"figure_id": "image_match"}, {"figure_id": "description_match"}],
        pages={"d|1": "page one", "d|2": "page two"},
        text_vectors=np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float32),
        image_vectors=np.array([[0.3, 0.9], [0.1, 0.9]], dtype=np.float32),
        description_vectors=np.array([[0.0, 1.0], [0.8, 0.2]], dtype=np.float32),
    )


def search(index, top_k_text=1, top_k_figures=1, figure_mode="image"):
    return retrieve("question", index, FakeEmbedder(), FakeEmbedder(), settings, top_k_text, top_k_figures, figure_mode)


def test_top_matches_returns_the_best_first():
    assert [position for position, _ in top_matches(np.array([0.1, 0.9, 0.5]), 2)] == [1, 2]


def test_figures_get_their_own_list_even_when_text_scores_are_higher():
    found = search(make_index())
    assert found.chunks[0]["chunk_id"] == "b"
    assert found.figures[0]["figure_id"] == "image_match"


def test_a_description_can_find_a_figure_the_image_search_ranks_lower():
    assert search(make_index(), figure_mode="description").figures[0]["figure_id"] == "description_match"


def test_rank_fusion_uses_places_so_a_scale_difference_does_not_decide():
    small_scale = np.array([0.30, 0.20, 0.10])
    large_scale = np.array([60.0, 90.0, 10.0])
    fused = fuse_ranks([small_scale, large_scale], 60)
    assert abs(fused[0] - fused[1]) < 1e-9  # first in one list and second in the other, both ways
    assert fused[2] < fused[0]


def test_chunks_search_but_the_model_gets_each_whole_page_once():
    index = make_index()
    index.chunks[0]["page"] = 2  # both chunks now come from one page
    found = replace(search(index, top_k_text=2), figures=[])
    assert found.pages == [{"doc": "d", "page": 2, "text": "page two"}]
    assert "page two" in build_content("question", found, "high")[1]["text"]


def test_term_matching_needs_the_whole_term():
    assert term_found("3", "the axis ends at 3.") and not term_found("3", "it was 13")
    assert term_found("red", "It is RED") and not term_found("red", "it reduced")
    assert term_found("0.5", "about 0.5%") and not term_found("0.5", "it was 10.5")
    assert term_found("pink|magenta", "a magenta line")


def test_chart_numbers_are_accepted_within_a_reading_error():
    assert number_found(24, 2.5, "September was about -23%")
    assert not number_found(2, 1.5, "March was about -6%")


def test_answer_scoring():
    assert answer_is_correct({"type": "text", "must_include": ["1.06", "3.43"]}, "from 1.06 to 3.43 m", "none")
    assert not answer_is_correct({"type": "text", "must_include": ["1.06", "3.43"]}, "about 1.06 m", "none")
    assert answer_is_correct({"type": "none"}, "I could not find this.", "I could not find this.")
    # the 5 in a source is not the answer
    assert not answer_is_correct({"type": "figure", "must_include": ["5"]}, "It is 4 meters. [doc, Figure 5, page 5]", "none")
    assert not answer_is_correct({"type": "figure", "must_include": ["5"]}, "Figure 5 shows 4 meters.", "none")


def test_a_value_from_the_wrong_year_fails_the_answer():
    item = {"type": "figure", "numbers": [[24, 2.5]], "must_not_match": "14\\.79"}
    assert not answer_is_correct(item, "About -24%, and 14.79 million in March", "none")
    assert answer_is_correct(item, "About -24%", "none")


def test_a_source_that_was_not_sent_is_caught():
    found = Retrieved(chunks=[], figures=[{"doc": "d", "number": 1, "page": 3}], pages=[{"doc": "d", "page": 2, "text": ""}])
    assert unknown_sources("Answer [d, Figure 1] and [d, page 2]", found) == []
    assert unknown_sources("Answer [d, Figure 2, page 4]", found) == ["d, Figure 2, page 4"]


def test_retrieval_rank_and_hit_rate():
    found = Retrieved(chunks=[{"doc": "d", "page": 1}, {"doc": "d", "page": 4}], figures=[{"figure_id": "f"}])
    assert retrieval_hit_rank({"type": "text", "doc": "d", "pages": [4]}, found) == 2
    assert retrieval_hit_rank({"type": "figure", "figure": "x"}, found) is None
    rows = [{"type": "text", "rank": 1}, {"type": "text", "rank": 3}, {"type": "text", "rank": None}]
    assert hit_rate(rows, "text", 2) == (1, 3)


def test_the_model_receives_the_picture_its_caption_and_its_notes(tmp_path):
    picture = tmp_path / "chart.png"
    picture.write_bytes(b"png")
    figure = {"doc": "d", "number": 2, "page": 4, "caption": "Fig. 2.", "notes": "* Not reliable.", "image_path": str(picture)}
    content = build_content("question", Retrieved(chunks=[], figures=[figure]), "high")
    assert content[-1]["type"] == "input_image" and content[-1]["image_url"].startswith("data:image/png;base64,")
    assert "Fig. 2." in content[-2]["text"] and "* Not reliable." in content[-2]["text"]
