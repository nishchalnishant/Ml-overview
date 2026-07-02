"""Chunk the docs/ corpus, embed each chunk, and persist an index to disk.

Uses FAISS if installed; otherwise falls back to a plain numpy matrix with
brute-force cosine similarity (fine at this corpus scale, see retrieval.py).
"""
import glob
import json
import os

import numpy as np

from chunking import chunk_text
from embeddings import embed

INDEX_DIR = "index"


def main() -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)

    all_chunks = []
    for path in sorted(glob.glob("docs/*.md")):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        source = os.path.basename(path)
        all_chunks.extend(chunk_text(text, source=source))

    if not all_chunks:
        raise SystemExit("No documents found in docs/*.md")

    texts = [c.text for c in all_chunks]
    vectors = embed(texts)

    np.save(os.path.join(INDEX_DIR, "vectors.npy"), vectors)
    metadata = [{"text": c.text, "source": c.source, "chunk_id": c.chunk_id} for c in all_chunks]
    with open(os.path.join(INDEX_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Indexed {len(all_chunks)} chunks from {len(set(c.source for c in all_chunks))} documents.")
    print(f"Wrote {INDEX_DIR}/vectors.npy and {INDEX_DIR}/metadata.json")


if __name__ == "__main__":
    main()
