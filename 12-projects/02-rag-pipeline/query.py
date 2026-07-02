"""CLI entry point: retrieve -> rerank -> generate -> print cited answer."""
import argparse

from generate import generate
from retrieval import retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG index.")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--llm", choices=["extractive", "openai", "anthropic"], default="extractive",
        help="Generation backend (default: extractive, no API key required)",
    )
    args = parser.parse_args()

    chunks = retrieve(args.question, top_k=args.top_k)
    if not chunks:
        print("No relevant chunks found. Did you run build_index.py?")
        return

    print("Retrieved sources:")
    for c in chunks:
        print(f"  [{c['source']}#{c['chunk_id']}] rerank_score={c['rerank_score']:.3f}")
    print()

    answer = generate(args.question, chunks, backend=args.llm)
    print("Answer:")
    print(answer)


if __name__ == "__main__":
    main()
