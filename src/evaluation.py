from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.document_loader import load_document
from src.rag_chain import build_llm_answer
from src.query_router import route_query
from src.query_transform import build_query_variants
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
    retrieval_mode: str = "vector",
    use_router: bool = False,
    use_mmr: bool = False,
    use_multi_query: bool = False,
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
        route = route_query(question) if use_router else None
        effective_mode = route.retrieval_mode if route else retrieval_mode
        effective_rerank = route.use_rerank if route else use_rerank
        effective_top_k = route.top_k if route else top_k
        effective_fetch_k = route.fetch_k if route else fetch_k

        query_variants = [
            {
                "query": variant.query,
                "strategy": variant.strategy,
                "reason": variant.reason,
            }
            for variant in build_query_variants(question, contextual_query=question)
        ]
        if use_multi_query:
            hits = store.search_multi(
                query_variants,
                retrieval_mode=effective_mode,
                top_k=effective_fetch_k,
                per_query_k=effective_fetch_k,
                use_mmr=use_mmr,
            )
        elif effective_mode == "hybrid":
            hits = store.search_hybrid(
                question,
                top_k=effective_fetch_k,
                vector_k=effective_fetch_k,
                keyword_k=effective_fetch_k,
                use_mmr=use_mmr,
            )
        else:
            hits = store.search(question, top_k=effective_fetch_k, use_mmr=use_mmr)
        if effective_rerank:
            hits = rerank_hits(question, hits, top_k=effective_top_k)
        else:
            hits = hits[:effective_top_k]

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
                "route_intent": route.intent if route else None,
                "route_reason": route.reason if route else None,
                "query_variants": [variant["strategy"] for variant in query_variants] if use_multi_query else [],
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
        "retrieval_mode": retrieval_mode,
        "use_router": use_router,
        "use_mmr": use_mmr,
        "use_multi_query": use_multi_query,
        "keyword_recall": round(keyword_hits / keyword_total, 4) if keyword_total else 0,
        "elapsed_seconds": round(elapsed, 3),
        "rows": rows,
    }
