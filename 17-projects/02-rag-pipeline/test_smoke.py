"""Smoke tests for pure-function pieces (chunking, reranking) that need no
embedding model or on-disk index.

Run: pytest test_smoke.py -v
"""
from chunking import chunk_text, split_sentences
from retrieval import _keywords, lexical_rerank


def test_split_sentences_basic():
    text = "This is one. This is two! Is this three?"
    assert split_sentences(text) == ["This is one.", "This is two!", "Is this three?"]


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(50))
    chunks = chunk_text(text, source="doc.txt", chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(c.source == "doc.txt" for c in chunks)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))


def test_chunk_text_short_input_single_chunk():
    chunks = chunk_text("Just one short sentence.", source="doc.txt")
    assert len(chunks) == 1
    assert chunks[0].text == "Just one short sentence."


def test_keywords_strips_stopwords_and_short_tokens():
    kw = _keywords("What is the capital of France?")
    assert "capital" in kw
    assert "france" in kw
    assert "the" not in kw
    assert "of" not in kw


def test_lexical_rerank_orders_by_blended_score():
    candidates = [
        {"text": "unrelated content about cooking recipes", "dense_score": 0.9},
        {"text": "the capital of france is paris", "dense_score": 0.5},
    ]
    reranked = lexical_rerank("capital of france", candidates, top_k=2)
    assert reranked[0]["text"].startswith("the capital")
    assert all("rerank_score" in c for c in reranked)
