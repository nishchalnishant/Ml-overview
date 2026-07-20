"""Thin wrapper around sentence-transformers for embedding chunks and queries."""
import numpy as np

_MODEL = None
MODEL_NAME = "all-MiniLM-L6-v2"


def get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def embed(texts: list[str]) -> np.ndarray:
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)
