import json
import os
import re
from functools import lru_cache
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import faiss
import numpy as np
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

# Your Hugging Face Dataset repository.
HF_REPO_ID = "notSoAman/rag-demo-knowledge"
HF_REPO_TYPE = "dataset"

# Only required if the HF dataset is private.
# Your current repository is public, so this can remain unset.
HF_TOKEN = os.getenv("HF_TOKEN")

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE retrieval instruction: use for queries, not document embeddings.
QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)

CANDIDATE_K = 50
DEFAULT_TOP_K = 8


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "but", "by", "can", "could", "did", "do", "does", "for",
    "from", "had", "has", "have", "he", "her", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "may", "of",
    "on", "or", "our", "that", "the", "their", "them", "there",
    "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "would", "you", "your",
}


# ============================================================
# Hugging Face / RAG resources
# ============================================================

@lru_cache(maxsize=1)
def _download_vector_store() -> tuple[str, str]:
    """Download and cache the FAISS index and metadata from HF."""

    print(
        f"Loading RAG resources from "
        f"{HF_REPO_ID}..."
    )

    index_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        filename="index.faiss",
        token=HF_TOKEN,
    )

    metadata_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        filename="metadata.json",
        token=HF_TOKEN,
    )

    return index_path, metadata_path


@lru_cache(maxsize=1)
def _load_resources():
    """Load model, FAISS index, and metadata once per worker."""

    print("Loading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    index_path, metadata_path = _download_vector_store()

    print("Loading FAISS index...")

    index = faiss.read_index(index_path)

    print("Loading metadata...")

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    if index.ntotal != len(metadata):
        raise RuntimeError(
            "FAISS/metadata mismatch: "
            f"index has {index.ntotal} vectors, "
            f"metadata has {len(metadata)} chunks."
        )

    print(
        f"RAG resources loaded: {index.ntotal} vectors"
    )

    return model, index, metadata


# ============================================================
# Lexical scoring
# ============================================================

def tokenize(text: str) -> list[str]:
    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower(),
    )

    return [
        word
        for word in words
        if word not in STOP_WORDS
    ]


def lexical_score(
    question: str,
    text: str,
) -> float:
    """Fraction of important question terms found in the passage."""

    question_words = set(
        tokenize(question)
    )

    if not question_words:
        return 0.0

    text_words = set(
        tokenize(text)
    )

    return len(
        question_words & text_words
    ) / len(question_words)


def phrase_score(
    question: str,
    text: str,
) -> float:
    """Small boost for exact adjacent phrases from the question."""

    question_words = tokenize(question)

    if len(question_words) < 2:
        return 0.0

    text_lower = re.sub(
        r"\s+",
        " ",
        text.lower(),
    )

    phrases = [
        f"{question_words[i]} {question_words[i + 1]}"
        for i in range(
            len(question_words) - 1
        )
    ]

    matches = sum(
        phrase in text_lower
        for phrase in phrases
    )

    return min(
        matches / len(phrases),
        1.0,
    )


# ============================================================
# Query embedding
# ============================================================

def embed_query(
    question: str,
) -> np.ndarray:
    """Create a normalized BGE embedding for the user question."""

    model, _, _ = _load_resources()

    query = (
        QUERY_PREFIX
        + question.strip()
    )

    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(
        embedding,
        dtype=np.float32,
    )


# ============================================================
# FAISS retrieval
# ============================================================

def _search_faiss(
    question: str,
    candidate_k: int,
) -> list[dict[str, Any]]:
    """
    Search the FAISS index and score the returned candidates.

    FAISS performs the vector search internally. We only process
    the requested top candidate_k results in Python.
    """

    _, index, metadata = _load_resources()

    candidate_k = min(
        candidate_k,
        index.ntotal,
    )

    if candidate_k <= 0:
        return []

    question_embedding = embed_query(
        question
    )

    distances, indices = index.search(
        question_embedding,
        candidate_k,
    )

    candidates = []

    for distance, idx in zip(
        distances[0],
        indices[0],
    ):
        if idx < 0 or idx >= len(metadata):
            continue

        chunk = metadata[idx]

        text = chunk.get(
            "chunk_text",
            "",
        )

        if not text:
            continue

        # Normalized vectors + inner product = cosine similarity.
        faiss_score = float(distance)

        lexical = lexical_score(
            question,
            text,
        )

        phrase = phrase_score(
            question,
            text,
        )

        # Map cosine similarity [-1, 1] to [0, 1].
        semantic = max(
            0.0,
            min(
                1.0,
                (faiss_score + 1.0) / 2.0,
            ),
        )

        # Semantic score remains dominant.
        final_score = (
            0.75 * semantic
            + 0.15 * lexical
            + 0.10 * phrase
        )

        candidates.append(
            {
                "chunk_id": chunk.get(
                    "chunk_id"
                ),
                "book": chunk.get(
                    "book"
                ),
                "source_file": chunk.get(
                    "source_file"
                ),
                "mythology": chunk.get(
                    "mythology"
                ),
                "chunk_text": text,
                "score": final_score,
                "faiss_score": faiss_score,
                "lexical_score": lexical,
                "phrase_score": phrase,
            }
        )

    return candidates


# ============================================================
# Source diversification
# ============================================================

def _diversify(
    candidates: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Avoid filling the LLM context with chunks from only one book.
    """

    selected = []
    source_counts: dict[str, int] = {}

    # First pass: maximum three chunks from one source.
    for item in candidates:
        source = (
            item.get("book")
            or item.get("source_file")
            or "unknown"
        )

        if source_counts.get(
            source,
            0,
        ) >= 3:
            continue

        selected.append(item)

        source_counts[source] = (
            source_counts.get(
                source,
                0,
            )
            + 1
        )

        if len(selected) >= top_k:
            break

    # Fill remaining slots if necessary.
    if len(selected) < top_k:
        selected_ids = {
            id(item)
            for item in selected
        }

        for item in candidates:
            if id(item) in selected_ids:
                continue

            selected.append(item)

            if len(selected) >= top_k:
                break

    return selected


# ============================================================
# Public API
# ============================================================

def retrieve_context(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = CANDIDATE_K,
) -> list[dict[str, Any]]:
    """Retrieve the best evidence chunks for a question."""

    if not question or not question.strip():
        return []

    top_k = max(
        1,
        top_k,
    )

    candidates = _search_faiss(
        question.strip(),
        candidate_k,
    )

    if not candidates:
        return []

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return _diversify(
        candidates,
        top_k,
    )


# ============================================================
# Debugging helper
# ============================================================

def print_retrieval_results(
    question: str,
    top_k: int = DEFAULT_TOP_K,
):
    results = retrieve_context(
        question,
        top_k=top_k,
    )

    print()
    print("=" * 80)
    print(
        f"QUESTION: {question}"
    )
    print("=" * 80)

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print()
        print(f"#{rank}")
        print(
            f"Book: {result['book']}"
        )
        print(
            f"Chunk ID: {result['chunk_id']}"
        )
        print(
            f"Retrieval score: "
            f"{result['score']:.4f}"
        )
        print(
            f"FAISS score: "
            f"{result['faiss_score']:.4f}"
        )
        print(
            f"Lexical score: "
            f"{result['lexical_score']:.4f}"
        )
        print(
            f"Phrase score: "
            f"{result['phrase_score']:.4f}"
        )
        print("-" * 80)
        print(result["chunk_text"])
        print("-" * 80)


if __name__ == "__main__":
    print_retrieval_results(
        "What was Lord Vishnu's first avatar?"
    )