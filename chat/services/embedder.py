from sentence_transformers import SentenceTransformer
import torch
import os
import json


device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5",
    device=device
)

print(f"Using device: {device}")

BATCH_SIZE = 32



def embed_chunks(input_folder_path: str, output_folder_path: str):

    os.makedirs(output_folder_path, exist_ok=True)

    for filename in os.listdir(input_folder_path):

        if not filename.endswith(".json"):
            continue

        input_file_path = os.path.join(
            input_folder_path,
            filename
        )

        output_file_path = os.path.join(
            output_folder_path,
            filename
        )

        # Load chunks
        with open(input_file_path, "r", encoding="utf-8") as infile:
            chunks = json.load(infile)

        print(f"\nEmbedding {filename} ({len(chunks)} chunks)...")

        # Extract text
        texts = [
            chunk["chunk_text"]
            for chunk in chunks
        ]

        # Embed in batches
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Attach embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()

        # Save
        with open(output_file_path, "w", encoding="utf-8") as outfile:
            json.dump(
                chunks,
                outfile,
                ensure_ascii=False,
                indent=2
            )

        print(f"✓ Saved {output_file_path}")


embed_chunks(
    "chat/knowledge/chunks/hindu",
    "chat/knowledge/embeddings/hindu"
)

embed_chunks(
    "chat/knowledge/chunks/greek",
    "chat/knowledge/embeddings/greek"
)

embed_chunks(
    "chat/knowledge/chunks/norse",
    "chat/knowledge/embeddings/norse"
)