"""Ask one question: python ask.py "What was the sea ice extent in September 2020?"
Add --no-llm to see what retrieval found without calling the paid model."""
import argparse
import sys

from multimodal_rag.answer import Answerer
from multimodal_rag.config import settings, setup_logging
from multimodal_rag.embeddings import ClipEmbedder
from multimodal_rag.retrieve import retrieve
from multimodal_rag.store import load_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--no-llm", action="store_true", help="show the retrieved items only")
    args = parser.parse_args()
    setup_logging()

    try:
        index = load_index(settings.index_dir)
        embedder = ClipEmbedder(settings.clip_model)
        found = retrieve(args.question, index, embedder, settings.top_k_text, settings.top_k_figures,
                         settings.use_figure_captions)

        print("\nText found:")
        for chunk in found.chunks:
            print(f"  {chunk['score']:.3f}  {chunk['doc']} p{chunk['page']}: {chunk['text'][:90]}")
        print("Figures found:")
        for figure in found.figures:
            print(f"  {figure['score']:.3f}  {figure['figure_id']} (page {figure['page']})")
        print("Pages sent to the model:", ", ".join(f"{page['doc']} p{page['page']}" for page in found.pages))

        if not args.no_llm:
            result = Answerer(settings).answer(args.question, found)
            print(f"\nAnswer: {result.text}")
            print(f"Tokens: {result.input_tokens} in, {result.output_tokens} out")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
