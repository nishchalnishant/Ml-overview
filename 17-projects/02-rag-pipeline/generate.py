"""Prompt construction and pluggable generation backends.

The extractive fallback (default, no API key needed) only ever returns
sentences that exist verbatim in a retrieved chunk, so hallucination is
structurally impossible in that mode — useful for testing retrieval quality
in isolation from generation quality. Swap in --llm openai/anthropic for a
real synthesized answer.
"""
import os
import re


def build_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks = "\n\n".join(
        f"[source: {c['source']}#{c['chunk_id']}]\n{c['text']}" for c in chunks
    )
    return (
        "Answer the question using ONLY the context below. "
        "Cite the source tag (e.g. [source: file.md#0]) for every claim. "
        "If the context doesn't contain the answer, say so explicitly.\n\n"
        f"Context:\n{context_blocks}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def _extractive_answer(query: str, chunks: list[dict]) -> str:
    """No-API-key fallback: pick the sentence with the highest keyword overlap
    with the query from the top chunk, cited verbatim."""
    query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
    best_sentence, best_source, best_score = None, None, -1
    for c in chunks:
        for sentence in re.split(r"(?<=[.!?])\s+", c["text"]):
            words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            score = len(query_words & words)
            if score > best_score:
                best_sentence, best_source, best_score = sentence, f"{c['source']}#{c['chunk_id']}", score
    if best_sentence is None:
        return "No relevant context found."
    return f"{best_sentence.strip()} [source: {best_source}]"


def _openai_answer(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def _anthropic_answer(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def generate(query: str, chunks: list[dict], backend: str = "extractive") -> str:
    if backend == "extractive":
        return _extractive_answer(query, chunks)

    prompt = build_prompt(query, chunks)
    if backend == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY to use --llm openai")
        return _openai_answer(prompt)
    if backend == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY to use --llm anthropic")
        return _anthropic_answer(prompt)

    raise ValueError(f"Unknown backend: {backend}")
