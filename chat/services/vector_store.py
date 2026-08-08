import json
from pathlib import Path

import faiss
import numpy as np


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDINGS_DIR = BASE_DIR / "knowledge" / "embeddings"
VECTOR_STORE_DIR = BASE_DIR / "knowledge" / "vector_store"

INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"


# ---------------------------------------------------------
# Load embeddings
# ---------------------------------------------------------

def load_embeddings():
    """
    Load all embedding JSON files and return:

    embeddings -> numpy array containing vectors
    metadata   -> list containing information about each vector
    """

    embeddings = []
    metadata = []

    json_files = list(EMBEDDINGS_DIR.rglob("*.json"))

    print(f"Found {len(json_files)} embedding files.")

    vector_id = 0

    for json_file in json_files:
        print(f"Loading: {json_file}")

        with open(json_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        for chunk in chunks:

            embedding = chunk.get("embedding")

            if embedding is None:
                continue

            embeddings.append(embedding)

            metadata.append({
                "vector_id": vector_id,
                "chunk_id": chunk.get("chunk_id"),
                "book": chunk.get("book"),
                "source_file": chunk.get("source_file"),
                "mythology": chunk.get("mythology"),
                "chunk_text": chunk.get("chunk_text"),
                "start_index": chunk.get("start_index"),
                "end_index": chunk.get("end_index"),
            })

            vector_id += 1

    if not embeddings:
        raise ValueError("No embeddings were found.")

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    return embeddings, metadata


# ---------------------------------------------------------
# Build FAISS index
# ---------------------------------------------------------

def build_index():

    embeddings, metadata = load_embeddings()

    print()
    print(f"Loaded {len(embeddings)} vectors.")
    print(f"Embedding dimensions: {embeddings.shape[1]}")

    # -----------------------------------------------------
    # Normalize vectors
    #
    # After normalization:
    #
    # inner product == cosine similarity
    # -----------------------------------------------------

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    # -----------------------------------------------------
    # CPU FAISS index
    # -----------------------------------------------------

    cpu_index = faiss.IndexFlatIP(dimension)

    # -----------------------------------------------------
    # Move index to GPU
    # -----------------------------------------------------

    print("Moving FAISS index to GPU...")

    gpu_resources = faiss.StandardGpuResources()

    gpu_index = faiss.index_cpu_to_gpu(
        gpu_resources,
        0,
        cpu_index
    )

    # -----------------------------------------------------
    # Add embeddings
    # -----------------------------------------------------

    print("Adding embeddings to GPU index...")

    gpu_index.add(embeddings)

    print(f"FAISS index contains {gpu_index.ntotal} vectors.")

    # -----------------------------------------------------
    # Move index back to CPU
    #
    # FAISS indexes are saved from CPU.
    # The GPU is used for building/searching, while the
    # persistent index is stored on disk.
    # -----------------------------------------------------

    final_index = faiss.index_gpu_to_cpu(gpu_index)

    VECTOR_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Save FAISS index
    # -----------------------------------------------------

    faiss.write_index(
        final_index,
        str(INDEX_PATH)
    )

    # -----------------------------------------------------
    # Save metadata
    # -----------------------------------------------------

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("Vector store created successfully.")
    print(f"Index:    {INDEX_PATH}")
    print(f"Metadata: {METADATA_PATH}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    build_index()