"""Fixed-size chunking with overlap and sentence-boundary snapping.

Naive fixed-size chunking can split a sentence in half, which hurts both
embedding quality (the chunk no longer represents one coherent idea) and
readability of the final citation. Snapping to the nearest sentence boundary
within a small window fixes this cheaply, without a full sentence-segmentation
dependency.
"""
import re
from dataclasses import dataclass

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def chunk_text(text: str, source: str, chunk_size: int = 500, overlap: int = 100) -> list[Chunk]:
    """Greedily pack sentences into ~chunk_size-character windows with overlap."""
    sentences = split_sentences(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    chunk_id = 0

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            chunks.append(Chunk(text=" ".join(current), source=source, chunk_id=chunk_id))
            chunk_id += 1
            # Overlap: carry the trailing sentences whose combined length <= overlap
            carry: list[str] = []
            carry_len = 0
            for s in reversed(current):
                if carry_len + len(s) > overlap:
                    break
                carry.insert(0, s)
                carry_len += len(s)
            current = carry
            current_len = carry_len

        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(Chunk(text=" ".join(current), source=source, chunk_id=chunk_id))

    return chunks
