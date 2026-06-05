from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from src.embeddings import embed_texts
from src.reranker import lexical_similarity
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

    def search_hybrid(
        self,
        question: str,
        top_k: int = 5,
        vector_k: int | None = None,
        keyword_k: int | None = None,
        vector_weight: float = 0.7,
    ) -> list[dict[str, Any]]:
        vector_k = vector_k or top_k
        keyword_k = keyword_k or top_k
        vector_hits = self.search(question, top_k=vector_k)
        keyword_hits = self._keyword_search(question, top_k=keyword_k)

        merged: dict[str, dict[str, Any]] = {}
        for hit in vector_hits:
            item = dict(hit)
            item["vector_score"] = float(hit.get("score", 0.0))
            item["keyword_score"] = 0.0
            merged[item["id"]] = item

        for hit in keyword_hits:
            item = merged.get(hit["id"], dict(hit))
            item["keyword_score"] = max(
                float(item.get("keyword_score", 0.0)),
                float(hit.get("keyword_score", 0.0)),
            )
            item.setdefault("vector_score", 0.0)
            merged[item["id"]] = item

        results = []
        for item in merged.values():
            vector_score = float(item.get("vector_score", 0.0))
            keyword_score = float(item.get("keyword_score", 0.0))
            hybrid_score = vector_weight * vector_score + (1 - vector_weight) * keyword_score
            item["hybrid_score"] = hybrid_score
            item["score"] = hybrid_score
            results.append(item)

        return sorted(results, key=lambda item: item["hybrid_score"], reverse=True)[:top_k]

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> int:
        existing = self.collection.get(include=[])
        ids = existing.get("ids", [])
        if not ids:
            return 0
        self.collection.delete(ids=ids)
        return len(ids)

    def _keyword_search(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        existing = self.collection.get(include=["documents", "metadatas"])
        ids = existing.get("ids", [])
        documents = existing.get("documents", [])
        metadatas = existing.get("metadatas", [])
        hits: list[dict[str, Any]] = []

        for chunk_id, text, metadata in zip(ids, documents, metadatas):
            score = lexical_similarity(question, text or "")
            if score <= 0:
                continue
            hits.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "chunk_index": metadata.get("chunk_index"),
                    "keyword_score": score,
                    "score": score,
                }
            )

        return sorted(hits, key=lambda item: item["keyword_score"], reverse=True)[:top_k]
