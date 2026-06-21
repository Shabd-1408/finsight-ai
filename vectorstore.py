"""
vectorstore.py

Thin wrapper around a persistent ChromaDB collection. Embeddings are
computed locally by ChromaDB's built-in default embedding model
(all-MiniLM-L6-v2 via onnxruntime) -- this runs on your own machine, costs
nothing, and needs no API key at all. Only the agent's reasoning step
(src/llm_client.py) talks to an external API (Groq, also free).
"""

import chromadb
from chromadb.utils import embedding_functions


class VectorStore:
    def __init__(self, persist_dir: str = "chroma_db", collection_name: str = "finsight_docs"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

    def add_documents(self, chunks: list):
        """
        chunks: list of dicts, each shaped like:
            {"id": str, "text": str, "metadata": {"source": str, "chunk_index": int}}
        Embeddings are computed automatically by the collection's embedding
        function -- no explicit embedding call needed here.
        """
        if not chunks:
            return
        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )

    def query(self, query_text: str, n_results: int = 4):
        return self.collection.query(query_texts=[query_text], n_results=n_results)

    def count(self) -> int:
        return self.collection.count()
