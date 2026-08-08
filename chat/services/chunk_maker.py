import json
import os
import re
from pathlib import Path


CHUNK_SIZE = 1000
OVERLAP = 200


def clean_text(text: str) -> str:
    """
    Normalize whitespace while preserving paragraphs.
    """

    # normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # collapse many blank lines into two
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def make_chunks(text: str,
                chunk_size: int = CHUNK_SIZE,
                overlap: int = OVERLAP):
    """
    Character chunks that never end in the middle of a word.
    """

    chunks = []

    start = 0
    chunk_id = 0
    text_length = len(text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        # Move end forward until whitespace
        while end < text_length and not text[end].isspace():
            end += 1

        chunk = text[start:end].strip()

        chunks.append({
            "chunk_id": chunk_id,
            "chunk_text": chunk,
            "start_index": start,
            "end_index": end,
            "length": len(chunk)
        })

        if end == text_length:
            break

        start = end - overlap
        chunk_id += 1

    return chunks


# -----------------------
# Main function
# -----------------------

def chunk_text_file(txt_path: str,
                    json_path: str,
                    mythology: str):

    with open(txt_path, encoding="utf-8") as f:
        text = clean_text(f.read())

    book = Path(txt_path).stem

    chunks = make_chunks(text)

    final = []

    for chunk in chunks:

        final.append({
            "chunk_id": chunk["chunk_id"],
            "book": book,
            "source_file": os.path.basename(txt_path),
            "mythology": mythology,
            "chunk_text": chunk["chunk_text"],
            "start_index": chunk["start_index"],
            "end_index": chunk["end_index"],
            "length": chunk["length"]
        })

    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"✓ {book} ({len(final)} chunks)")


# -----------------------
# Folder processor
# -----------------------

def chunk_folder(processed_folder, chunks_folder):

    mythology = os.path.basename(processed_folder)

    for file in sorted(os.listdir(processed_folder)):

        if not file.endswith(".txt"):
            continue

        txt_path = os.path.join(processed_folder, file)

        json_path = os.path.join(
            chunks_folder,
            Path(file).stem + ".json"
        )

        chunk_text_file(
            txt_path,
            json_path,
            mythology
        )


if __name__ == "__main__":

    chunk_folder(
        "chat/knowledge/processed/hindu",
        "chat/knowledge/chunks/hindu"
    )

    chunk_folder(
        "chat/knowledge/processed/greek",
        "chat/knowledge/chunks/greek"
    )

    chunk_folder(
        "chat/knowledge/processed/norse",
        "chat/knowledge/chunks/norse"
    )