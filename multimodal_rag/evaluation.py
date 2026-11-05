"""Scoring against the golden dataset. Retrieval scoring is free. Answer scoring calls the paid model."""
import json
import logging
import re
from dataclasses import replace

from .retrieve import retrieve

log = logging.getLogger("evaluation")


def load_golden(path):
    if not path.exists():
        raise FileNotFoundError(f"No golden dataset at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def term_found(term, text):
    """One expected term, with | between accepted spellings. Whole terms only: "red" is not in "reduced", "3" not in "13"."""
    options = "|".join(re.escape(option) for option in term.split("|"))
    return re.search(rf"(?<![\w.])(?:{options})(?!\w)", text, re.IGNORECASE) is not None


def answer_is_correct(item, answer, not_found_sentence):
    if item["type"] == "none":
        return not_found_sentence.lower() in answer.lower()
    return all(term_found(term, answer) for term in item["must_include"])


def retrieval_hit_rank(item, found):
    """Where the expected source is in its list, starting at 1. None when it was not retrieved."""
    if item["type"] == "text":
        for rank, chunk in enumerate(found.chunks, start=1):
            if chunk["doc"] == item["doc"] and chunk["page"] in item["pages"]:
                return rank
    elif item["type"] == "figure":
        for rank, figure in enumerate(found.figures, start=1):
            if figure["figure_id"] == item["figure"]:
                return rank
    return None


def answer_chunk_rank(item, found):
    """A stricter check for text questions: where is the chunk that really holds the answer?"""
    if item["type"] == "text":
        for rank, chunk in enumerate(found.chunks, start=1):
            if all(term_found(term, chunk["text"]) for term in item["must_include"]):
                return rank
    return None


def evaluate_retrieval(golden, index, embedder, max_k_text, max_k_figures, use_captions):
    """Each question's rank of the expected source, searched with the largest k so smaller k can be read from it."""
    rows = []
    for item in golden:
        if item["type"] == "none":
            continue
        found = retrieve(item["question"], index, embedder, max_k_text, max_k_figures, use_captions)
        rows.append({"id": item["id"], "type": item["type"], "rank": retrieval_hit_rank(item, found),
                     "answer_chunk_rank": answer_chunk_rank(item, found)})
    return rows


def hit_rate(rows, kind, k, field="rank"):
    """Share of questions of this kind whose expected source is in the top k."""
    selected = [row for row in rows if row["type"] == kind]
    hits = [row for row in selected if row[field] is not None and row[field] <= k]
    return len(hits), len(selected)


def evaluate_answers(golden, index, embedder, answerer, settings, use_figures):
    """Answer every question. With use_figures=False the same pipeline runs as a text-only RAG, for comparison."""
    rows = []
    for item in golden:
        found = retrieve(item["question"], index, embedder, settings.top_k_text, settings.top_k_figures,
                         settings.use_figure_captions)
        if not use_figures:
            found = replace(found, figures=[])
        try:
            result = answerer.answer(item["question"], found)
        except RuntimeError as exc:
            log.error("%s failed: %s", item["id"], exc)
            rows.append({"id": item["id"], "type": item["type"], "question": item["question"], "answer": None,
                         "correct": False, "cited_doc": False, "input_tokens": 0, "output_tokens": 0})
            continue
        rows.append({
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "answer": result.text,
            "correct": answer_is_correct(item, result.text, settings.not_found_sentence),
            "cited_doc": item["type"] == "none" or item["doc"] in result.text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        })
    return rows
