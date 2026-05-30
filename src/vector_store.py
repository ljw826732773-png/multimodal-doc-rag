from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from src.embeddings import embed_texts
from src.text_splitter import Chunk


class DocumentVectorStore:
    def __init__(self, persist_dir: str | Path, collection_name: str = "documents") -> None:
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        ids = [chunk.id for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "source": chunk.source,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]
        embeddings = embed_texts(texts)

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunks)

    def search(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        question_embedding = embed_texts([question])[0]
        result = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict[str, Any]] = []
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            hits.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "chunk_index": metadata.get("chunk_index"),
                    "score": max(0.0, 1.0 - float(distance)),
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()
