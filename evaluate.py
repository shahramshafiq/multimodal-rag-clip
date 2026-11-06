"""Evaluate against the golden dataset.
  python evaluate.py             retrieval only
  python evaluate.py --answers   also answers every question (paid model calls)
  python evaluate.py --answers --compare-text-only   also repeats every question with figures removed (costs more)
"""
import argparse
import json
import sys

from multimodal_rag.answer import Answerer
from multimodal_rag.config import settings, setup_logging
from multimodal_rag.embeddings import ClipEmbedder, TextEmbedder
from multimodal_rag.evaluation import evaluate_answers, evaluate_retrieval, hit_rate, load_golden
from multimodal_rag.store import load_index

TEXT_KS = (1, 2, 4, 8)
FIGURE_KS = (1, 2, 3)
GROUPS = (("easy", "text"), ("easy", "figure"), ("easy", "none"),
          ("natural", "text"), ("natural", "figure"), ("natural", "trap"),
          ("holdout", "text"), ("holdout", "figure"), ("holdout", "none"))
SETS = ("easy", "natural", "holdout")


def report_retrieval(golden, index, clip, text_embedder):
    results = {}
    for name in ("image", "description", "both"):
        rows = evaluate_retrieval(golden, index, clip, text_embedder, settings, max(TEXT_KS), max(FIGURE_KS), name)
        results[name] = rows
        print(f"\nFigure search by {name}:")
        for only_set in SETS:
            scores = [hit_rate(rows, "figure", k, only_set=only_set) for k in FIGURE_KS]
            print(f"  {only_set:8}" + "".join(f"  top {k}: {hits}/{total}" for k, (hits, total) in zip(FIGURE_KS, scores)))
    print("\nText search, right page:")
    rows = results["both"]
    for only_set in SETS:
        scores = [hit_rate(rows, "text", k, only_set=only_set) for k in TEXT_KS]
        print(f"  {only_set:8}" + "".join(f"  top {k}: {hits}/{total}" for k, (hits, total) in zip(TEXT_KS, scores)))
    print("Text search, chunk holding the answer:")
    for only_set in SETS:
        scores = [hit_rate(rows, "text", k, field="answer_chunk_rank", only_set=only_set) for k in TEXT_KS]
        print(f"  {only_set:8}" + "".join(f"  top {k}: {hits}/{total}" for k, (hits, total) in zip(TEXT_KS, scores)))
    return results


def report_answers(rows, name):
    print(f"\n{name}:")
    for only_set, kind in GROUPS:
        selected = [row for row in rows if row["set"] == only_set and row["type"] == kind]
        print(f"  {only_set:8} {kind:7} {sum(row['correct'] for row in selected)}/{len(selected)} correct")
    print(f"  total {sum(row['correct'] for row in rows)}/{len(rows)}")
    print(f"  source named in {sum(row['cited_doc'] for row in rows)}/{len(rows)} answers")
    print(f"  answers citing a source that was not sent: {sum(bool(row['unknown_sources']) for row in rows)}")
    print(f"  tokens: {sum(r['input_tokens'] for r in rows)} in, {sum(r['output_tokens'] for r in rows)} out")
    for row in rows:
        if not row["correct"]:
            print(f"  WRONG {row['id']}: {row['answer']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true", help="also score answers (paid model calls)")
    parser.add_argument("--compare-text-only", action="store_true", help="also answer with figures removed")
    args = parser.parse_args()
    setup_logging()

    try:
        golden = load_golden(settings.golden_file)
        index = load_index(settings.index_dir)
        clip = ClipEmbedder(settings.clip_model)
        text_embedder = TextEmbedder(settings)
        results = {"retrieval": report_retrieval(golden, index, clip, text_embedder)}

        if args.answers:
            answerer = Answerer(settings)
            results["multimodal"] = evaluate_answers(golden, index, clip, text_embedder, answerer, settings, use_figures=True)
            report_answers(results["multimodal"], "Multimodal RAG (text and figures)")
            if args.compare_text_only:
                results["text_only"] = evaluate_answers(golden, index, clip, text_embedder, answerer, settings, use_figures=False)
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
