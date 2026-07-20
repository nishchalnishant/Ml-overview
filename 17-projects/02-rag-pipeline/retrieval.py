"""Two-stage retrieval: dense cosine-similarity search, then a lexical rerank.

The rerank stage is a cheap stand-in for a cross-encoder: it doesn't need an
extra model, but it corrects cases where dense similarity alone surfaces a
topically-close but keyword-irrelevant chunk over a more precise match.
"""
import json
import re

import numpy as np

from embeddings import embed

INDEX_DIR = "index"
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "for",
    "on", "and", "or", "how", "does", "do", "what", "with", "this", "that",
}


def load_index() -> tuple[np.ndarray, list[dict]]:
    vectors = np.load(f"{INDEX_DIR}/vectors.npy")
    with open(f"{INDEX_DIR}/metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return vectors, metadata


def dense_search(query: str, vectors: np.ndarray, metadata: list[dict], top_k: int = 20) -> list[dict]:
    query_vec = embed([query])[0]
    # vectors are already normalized (see embeddings.py) so dot product == cosine similarity
    scores = vectors @ query_vec
    top_idx = np.argsort(-scores)[:top_k]
    results = []
    for idx in top_idx:
        item = dict(metadata[idx])
        item["dense_score"] = float(scores[idx])
        results.append(item)
    return results


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def lexical_rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    query_kw = _keywords(query)
    for c in candidates:
        chunk_kw = _keywords(c["text"])
        overlap = len(query_kw & chunk_kw) / max(len(query_kw), 1)
        # blend dense similarity with lexical overlap — dense score dominates,
        # lexical overlap breaks ties and corrects near-miss dense retrievals
        c["rerank_score"] = 0.7 * c["dense_score"] + 0.3 * overlap
    return sorted(candidates, key=lambda c: -c["rerank_score"])[:top_k]


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    vectors, metadata = load_index()
    candidates = dense_search(query, vectors, metadata, top_k=max(20, top_k * 4))
    return lexical_rerank(query, candidates, top_k=top_k)
