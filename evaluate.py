"""Evaluate against the golden dataset.
  python evaluate.py             retrieval only, free
  python evaluate.py --answers   also answers every question twice (with and without figures), uses paid model calls
"""
import argparse
import json
import sys

from multimodal_rag.answer import Answerer
from multimodal_rag.config import settings, setup_logging
from multimodal_rag.embeddings import ClipEmbedder
from multimodal_rag.evaluation import evaluate_answers, evaluate_retrieval, hit_rate, load_golden
from multimodal_rag.store import load_index

TEXT_KS = (1, 2, 4, 8)
FIGURE_KS = (1, 2, 3)
KINDS = ("text", "figure", "none")


def report_retrieval(golden, index, embedder):
    results = {}
    for use_captions in (False, True):
        rows = evaluate_retrieval(golden, index, embedder, max(TEXT_KS), max(FIGURE_KS), use_captions)
        name = "image and caption" if use_captions else "image only"
        results[name] = rows
        print(f"\nFigure search by {name}:")
        for k in FIGURE_KS:
            hits, total = hit_rate(rows, "figure", k)
            print(f"  right figure in top {k}: {hits}/{total}")
    print("\nText search:")
    for k in TEXT_KS:
        pages, total = hit_rate(results["image and caption"], "text", k)
        chunks, _ = hit_rate(results["image and caption"], "text", k, field="answer_chunk_rank")
        print(f"  top {k}: right page {pages}/{total}, chunk holding the answer {chunks}/{total}")
    return results


def report_answers(rows, name):
    print(f"\n{name}:")
    for kind in KINDS:
        selected = [row for row in rows if row["type"] == kind]
        correct = sum(row["correct"] for row in selected)
        print(f"  {kind:7} {correct}/{len(selected)} correct")
    print(f"  source named in {sum(row['cited_doc'] for row in rows)}/{len(rows)} answers")
    print(f"  tokens: {sum(r['input_tokens'] for r in rows)} in, {sum(r['output_tokens'] for r in rows)} out")
    for row in rows:
        if not row["correct"]:
            print(f"  WRONG {row['id']}: {row['answer']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true", help="also score answers (paid model calls)")
    args = parser.parse_args()
    setup_logging()

    try:
        golden = load_golden(settings.golden_file)
        index = load_index(settings.index_dir)
        embedder = ClipEmbedder(settings.clip_model)
        results = {"retrieval": report_retrieval(golden, index, embedder)}

        if args.answers:
            answerer = Answerer(settings)
            results["multimodal"] = evaluate_answers(golden, index, embedder, answerer, settings, use_figures=True)
            results["text_only"] = evaluate_answers(golden, index, embedder, answerer, settings, use_figures=False)
            report_answers(results["multimodal"], "Multimodal RAG (text and figures)")
            report_answers(results["text_only"], "Text-only RAG (same pipeline, figures removed)")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    path = settings.results_dir / ("answers.json" if args.answers else "retrieval.json")
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
