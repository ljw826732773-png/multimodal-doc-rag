from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.embeddings import embed_texts
from src.retrieval_algorithms import (
    bm25_search,
    normalize_scores,
    reciprocal_rank_fusion,
    select_mmr,
)
from src.text_splitter import Chunk


class DocumentVectorStore:
    def __init__(self, persist_dir: str | Path, collection_name: str = "documents") -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.persist_dir / f"{collection_name}.json"
        if not self.path.exists():
            self._save([])

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        records = self._load()
        by_id = {record["id"]: record for record in records}
        texts = [chunk.text for chunk in chunks]
        embeddings = embed_texts(texts)

        for chunk, embedding in zip(chunks, embeddings):
            by_id[chunk.id] = {
                "id": chunk.id,
                "text": chunk.text,
                "embedding": embedding,
                "metadata": {
                    "source": chunk.source,
                    "file_id": chunk.file_id,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                },
            }

        self._save(list(by_id.values()))
        return len(chunks)

    def search(self, question: str, top_k: int = 5, use_mmr: bool = False) -> list[dict[str, Any]]:
        records = self._load()
        if not records:
            return []

        question_embedding = np.asarray(embed_texts([question])[0], dtype=np.float32)
        hits: list[dict[str, Any]] = []
        for record in records:
            embedding = np.asarray(record["embedding"], dtype=np.float32)
            score = _cosine(question_embedding, embedding)
            hits.append(self._record_to_hit(record, score=score))

        hits = sorted(hits, key=lambda item: item["score"], reverse=True)
        if use_mmr:
            return select_mmr(hits, top_k=top_k)
        return hits[:top_k]

    def search_hybrid(
        self,
        question: str,
        top_k: int = 5,
        vector_k: int | None = None,
        keyword_k: int | None = None,
        vector_weight: float = 0.7,
        use_mmr: bool = False,
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

        results = sorted(results, key=lambda item: item["hybrid_score"], reverse=True)
        if use_mmr:
            return select_mmr(results, top_k=top_k)
        return results[:top_k]

    def search_multi(
        self,
        queries: list[dict[str, str]],
        retrieval_mode: str = "hybrid",
        top_k: int = 5,
        per_query_k: int | None = None,
        use_mmr: bool = False,
    ) -> list[dict[str, Any]]:
        per_query_k = per_query_k or max(top_k, 8)
        ranked_lists: list[list[dict[str, Any]]] = []

        for variant_index, variant in enumerate(queries):
            query_text = variant.get("query", "").strip()
            if not query_text:
                continue

            if retrieval_mode == "hybrid":
                hits = self.search_hybrid(
                    query_text,
                    top_k=per_query_k,
                    vector_k=per_query_k,
                    keyword_k=per_query_k,
                    use_mmr=False,
                )
            else:
                hits = self.search(query_text, top_k=per_query_k, use_mmr=False)

            for hit in hits:
                hit["query_variant"] = variant.get("strategy", f"variant_{variant_index}")
                hit["query_text"] = query_text
                hit["retrieval_mode"] = retrieval_mode
            ranked_lists.append(hits)

        fused_limit = max(top_k * 3, per_query_k)
        fused = reciprocal_rank_fusion(ranked_lists, top_k=fused_limit)
        if use_mmr:
            return select_mmr(fused, top_k=top_k)
        return fused[:top_k]

    def count(self) -> int:
        return len(self._load())

    def clear(self) -> int:
        records = self._load()
        self._save([])
        return len(records)

    def list_documents(self) -> list[dict[str, Any]]:
        docs: dict[str, dict[str, Any]] = {}

        for record in self._load():
            metadata = record["metadata"]
            source = metadata.get("source", "unknown")
            file_id = metadata.get("file_id", source)
            page = int(metadata.get("page", 0) or 0)
            item = docs.setdefault(
                file_id,
                {
                    "file_id": file_id,
                    "source": source,
                    "chunks": 0,
                    "pages": set(),
                },
            )
            item["chunks"] += 1
            if page:
                item["pages"].add(page)

        rows = []
        for item in docs.values():
            rows.append(
                {
                    "file_id": item["file_id"],
                    "source": item["source"],
                    "chunks": item["chunks"],
                    "pages": len(item["pages"]),
                }
            )
        return sorted(rows, key=lambda row: row["source"])

    def delete_document(self, file_id: str) -> int:
        records = self._load()
        kept = [
            record
            for record in records
            if record["metadata"].get("file_id", record["metadata"].get("source")) != file_id
        ]
        deleted = len(records) - len(kept)
        self._save(kept)
        return deleted

    def _keyword_search(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        scored_records = bm25_search(question, self._load(), top_k=top_k)
        normalized_scores = normalize_scores([score for _, score in scored_records])

        hits: list[dict[str, Any]] = []
        for (record, raw_score), normalized_score in zip(scored_records, normalized_scores):
            hit = self._record_to_hit(record, score=normalized_score)
            hit["keyword_score"] = normalized_score
            hit["bm25_score"] = round(raw_score, 6)
            hits.append(hit)

        return hits

    def _record_to_hit(self, record: dict[str, Any], score: float) -> dict[str, Any]:
        metadata = record["metadata"]
        return {
            "id": record["id"],
            "text": record["text"],
            "source": metadata.get("source"),
            "file_id": metadata.get("file_id", metadata.get("source")),
            "page": metadata.get("page"),
            "chunk_index": metadata.get("chunk_index"),
            "score": max(0.0, float(score)),
            "_embedding": record.get("embedding"),
        }

    def _load(self) -> list[dict[str, Any]]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, records: list[dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))
