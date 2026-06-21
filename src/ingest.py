"""
ingest.py

Loads every .txt document in data/sample_docs, splits each into overlapping
word-based chunks, embeds them, and writes them into the persistent Chroma
vector store. Run this once before starting the Streamlit app:

    python -m src.ingest
"""

import os
import glob
import uuid

from src.vectorstore import VectorStore

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs")
CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 30


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def load_and_chunk_documents(data_dir: str = DATA_DIR) -> list:
    all_chunks = []
    filepaths = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        for i, chunk in enumerate(chunk_text(text)):
            all_chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "metadata": {"source": filename, "chunk_index": i},
            })
    return all_chunks


def main():
    chunks = load_and_chunk_documents()
    print(f"Loaded {len(chunks)} chunks from {DATA_DIR}")
    if not chunks:
        print("No documents found. Add .txt files to data/sample_docs/ first.")
        return
    store = VectorStore()
    store.add_documents(chunks)
    print(f"Ingestion complete. Vector store now has {store.count()} chunks persisted to ./chroma_db")


if __name__ == "__main__":
    main()
