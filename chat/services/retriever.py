import json
import os
import re
from functools import lru_cache
from typing import Any

# Suppress Hugging Face progress bars/noise where supported.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


try:
    from .vector_store import INDEX_PATH, METADATA_PATH
except ImportError:
    from vector_store import INDEX_PATH, METADATA_PATH


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CANDIDATE_K = 50
DEFAULT_TOP_K = 8

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "can", "could", "did", "do", "does", "for", "from", "had",
    "has", "have", "he", "her", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "may", "of", "on", "or", "our", "that", "the",
    "their", "them", "there", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would",
    "you", "your",
}


@lru_cache(maxsize=1)
def _load_resources():
    """Load the embedding model, FAISS index, and metadata once per process."""
    print("Loading RAG resources...")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    index = faiss.read_index(str(INDEX_PATH))

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if index.ntotal != len(metadata):
        raise RuntimeError(
            f"FAISS/metadata mismatch: index has {index.ntotal} vectors "
            f"but metadata has {len(metadata)} chunks."
        )

    print(f"RAG resources loaded: {index.ntotal} vectors")

    return model, index, metadata


def tokenize(text: str) -> list[str]:
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return [word for word in words if word not in STOP_WORDS]


def lexical_score(question: str, text: str) -> float:
    """Fraction of important question terms present in the passage."""
    question_words = set(tokenize(question))
    if not question_words:
        return 0.0

    text_words = set(tokenize(text))
    return len(question_words & text_words) / len(question_words)


def phrase_score(question: str, text: str) -> float:
    """
    Score useful multi-word phrases from the question.

    We especially care about factual relation phrases such as:
    "first avatar", "first incarnation", "second incarnation".
    """
    question_words = tokenize(question)
    if len(question_words) < 2:
        return 0.0

    text_lower = re.sub(r"\s+", " ", text.lower())
    phrases = []

    for i in range(len(question_words) - 1):
        phrases.append(f"{question_words[i]} {question_words[i + 1]}")

    matches = sum(phrase in text_lower for phrase in phrases)
    return min(matches / max(len(phrases), 1), 1.0)


def embed_query(question: str) -> np.ndarray:
    model, _, _ = _load_resources()

    query = QUERY_PREFIX + question.strip()

    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(embedding, dtype=np.float32)


def _search_faiss(question: str, candidate_k: int):
    _, index, metadata = _load_resources()

    candidate_k = min(candidate_k, index.ntotal)
    question_embedding = embed_query(question)

    distances, indices = index.search(question_embedding, candidate_k)

    candidates = []

    for distance, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue

        chunk = metadata[idx]
        text = chunk.get("chunk_text", "")

        if not text:
            continue

        # The index is normalized, so inner-product distance is cosine
        # similarity. Keep the raw value; do not min-max normalize against
        # the current candidate set because that makes scores query-relative.
        faiss_score = float(distance)

        lexical = lexical_score(question, text)
        phrase = phrase_score(question, text)

        # Convert cosine similarity from [-1, 1] to [0, 1].
        semantic = max(0.0, min(1.0, (faiss_score + 1.0) / 2.0))

        # Semantic similarity remains dominant, while exact wording helps
        # factual questions such as "first avatar" / "first incarnation".
        final_score = (
            0.75 * semantic
            + 0.15 * lexical
            + 0.10 * phrase
        )

        candidates.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "book": chunk.get("book"),
                "source_file": chunk.get("source_file"),
                "mythology": chunk.get("mythology"),
                "chunk_text": text,
                "score": final_score,
                "faiss_score": faiss_score,
                "lexical_score": lexical,
                "phrase_score": phrase,
            }
        )

    return candidates


def _diversify(candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """
    Avoid filling the entire context with near-duplicate chunks from one book.

    We first prefer strong results, but allow multiple chunks from a source
    when they are clearly useful. This is intentionally conservative.
    """
    selected = []
    source_counts: dict[str, int] = {}

    # First pass: at most 3 chunks per source.
    for item in candidates:
        source = item.get("book") or item.get("source_file") or "unknown"
        if source_counts.get(source, 0) >= 3:
            continue

        selected.append(item)
        source_counts[source] = source_counts.get(source, 0) + 1

        if len(selected) >= top_k:
            break

    # Second pass: fill any remaining slots.
    if len(selected) < top_k:
        selected_ids = {id(item) for item in selected}
        for item in candidates:
            if id(item) in selected_ids:
                continue
            selected.append(item)
            if len(selected) >= top_k:
                break

    return selected


def retrieve_context(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = CANDIDATE_K,
) -> list[dict[str, Any]]:
    """Retrieve the best evidence chunks for a question."""
    if not question or not question.strip():
        return []

    top_k = max(1, top_k)
    candidates = _search_faiss(question.strip(), candidate_k)

    if not candidates:
        return []

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return _diversify(candidates, top_k)


def print_retrieval_results(question: str, top_k: int = DEFAULT_TOP_K):
    results = retrieve_context(question, top_k=top_k)

    print()
    print("=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"\n#{rank}")
        print(f"Book: {result['book']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Retrieval score: {result['score']:.4f}")
        print(f"FAISS score: {result['faiss_score']:.4f}")
        print(f"Lexical score: {result['lexical_score']:.4f}")
        print(f"Phrase score: {result['phrase_score']:.4f}")
        print("-" * 80)
        print(result["chunk_text"])
        print("-" * 80)


if __name__ == "__main__":
    # Debug only. Importing this module no longer performs retrieval.
    print_retrieval_results("What was Lord Vishnu's first avatar?")