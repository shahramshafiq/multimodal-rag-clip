"""Scoring against the golden dataset. Retrieval scoring is free. Answer scoring calls the paid model."""
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from .answer import unknown_sources
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


def without_sources(answer):
    """Drop "[doc, Figure 5, page 8]" and "Figure 5", so the number in a source never counts as the answer."""
    return re.sub(r"\[[^\]]*\]|\b(?:Figure|Fig\.|page)\s*\d+", " ", answer, flags=re.IGNORECASE)


def numbers_in(text):
    """Every number in the text, as a positive value: "-24%" and "24 percent below" both give 24."""
    return [float(number.replace(",", "")) for number in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]


def number_found(expected, allowed_error, text):
    return any(abs(number - expected) <= allowed_error for number in numbers_in(text))


def answer_is_correct(item, answer, not_found_sentence):
    if item["type"] == "none":
        return not_found_sentence.lower() in answer.lower()
    claim = without_sources(answer)
    if "must_not_match" in item and re.search(item["must_not_match"], claim, re.IGNORECASE):
        return False  # for example a value taken from the wrong year, or a number made up for a trap question
    if "must_match" in item and not re.search(item["must_match"], claim, re.IGNORECASE):
        return False  # for example a trap answer that never says the value is not given
    return (all(term_found(term, claim) for term in item.get("must_include", []))
            and all(number_found(value, error, claim) for value, error in item.get("numbers", [])))


def question_set(item):
    """easy: worded almost like the captions and the report. natural: worded the way a person asks.
    holdout: added after every setting was chosen, and never tuned on."""
    if item["id"].startswith("holdout_"):
        return "holdout"
    return "natural" if item["id"].startswith(("natural_", "trap_")) else "easy"


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
            if all(term_found(term, chunk["text"]) for term in item.get("must_include", [])):
                return rank
    return None


def evaluate_retrieval(golden, index, clip, text_embedder, settings, max_k_text, max_k_figures, figure_mode):
    """Each question's rank of the expected source, searched with the largest k so smaller k can be read from it."""
    rows = []
    for item in golden:
        if item["type"] not in ("text", "figure"):
            continue
        found = retrieve(item["question"], index, clip, text_embedder, settings, max_k_text, max_k_figures, figure_mode)
        rows.append({"id": item["id"], "type": item["type"], "set": question_set(item), "rank": retrieval_hit_rank(item, found),
                     "answer_chunk_rank": answer_chunk_rank(item, found)})
    return rows


def hit_rate(rows, kind, k, field="rank", only_set=None):
    """Share of questions of this kind whose expected source is in the top k."""
    selected = [row for row in rows if row["type"] == kind and only_set in (None, row.get("set"))]
    hits = [row for row in selected if row[field] is not None and row[field] <= k]
    return len(hits), len(selected)


def evaluate_answers(golden, index, clip, text_embedder, answerer, settings, use_figures):
    """Answer every question. With use_figures=False the same pipeline runs as a text-only RAG, for comparison."""
    questions = []
    for item in golden:
        found = retrieve(item["question"], index, clip, text_embedder, settings)
        questions.append((item, found if use_figures else replace(found, figures=[])))

    def answer_one(question):
        item, found = question
        row = {"id": item["id"], "type": item["type"], "set": question_set(item), "question": item["question"]}
        try:
            result = answerer.answer(item["question"], found)
        except RuntimeError as exc:
            log.error("%s failed: %s", item["id"], exc)
            return {**row, "answer": None, "correct": False, "cited_doc": False, "unknown_sources": [],
                    "input_tokens": 0, "output_tokens": 0}
        return {
            **row,
            "answer": result.text,
            "correct": answer_is_correct(item, result.text, settings.not_found_sentence),
            "cited_doc": item["type"] in ("none", "trap") or item["doc"] in result.text,
            "unknown_sources": unknown_sources(result.text, found),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }

    with ThreadPoolExecutor(settings.evaluation_workers) as pool:
        return list(pool.map(answer_one, questions))
