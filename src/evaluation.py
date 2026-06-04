from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.document_loader import load_document
from src.rag_chain import build_llm_answer
from src.reranker import rerank_hits
from src.text_splitter import split_pages
from src.vector_store import DocumentVectorStore


def run_retrieval_eval(
    document_path: str | Path,
    questions_path: str | Path,
    persist_dir: str | Path,
    top_k: int = 3,
    fetch_k: int = 8,
    use_rerank: bool = True,
) -> dict[str, Any]:
    document_path = Path(document_path)
    questions_path = Path(questions_path)

    pages = load_document(document_path, enable_ocr=True)
    chunks = split_pages(pages)
    store = DocumentVectorStore(persist_dir, collection_name="eval")
    store.clear()
    store.add_chunks(chunks)

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    rows = []
    started = time.perf_counter()

    for item in questions:
        question = item["question"]
        expected_keywords = item["expected_keywords"]
        hits = store.search(question, top_k=fetch_k)
        if use_rerank:
            hits = rerank_hits(question, hits, top_k=top_k)
        else:
            hits = hits[:top_k]

        answer = build_llm_answer(question, hits, provider="disabled")
        retrieved_context = "\n".join(hit.get("text", "") for hit in hits)
        matched = [
            keyword
            for keyword in expected_keywords
            if keyword.lower() in retrieved_context.lower()
        ]
        rows.append(
            {
                "question": question,
                "matched_keywords": len(matched),
                "total_keywords": len(expected_keywords),
                "answer_preview": answer[:160],
                "top_source": hits[0]["source"] if hits else None,
                "top_score": round(hits[0]["score"], 4) if hits else 0,
            }
        )

    elapsed = time.perf_counter() - started
    keyword_hits = sum(row["matched_keywords"] for row in rows)
    keyword_total = sum(row["total_keywords"] for row in rows)
    return {
        "document": str(document_path),
        "questions": len(rows),
        "top_k": top_k,
        "fetch_k": fetch_k,
        "rerank": use_rerank,
        "keyword_recall": round(keyword_hits / keyword_total, 4) if keyword_total else 0,
        "elapsed_seconds": round(elapsed, 3),
        "rows": rows,
    }
