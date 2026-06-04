from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.document_loader import load_document
from src.rag_chain import build_llm_answer
from src.reranker import rerank_hits
from src.text_splitter import split_pages
from src.vector_store import DocumentVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small RAG evaluation.")
    parser.add_argument("--document", default="examples/sample.txt")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--fetch-k", type=int, default=8)
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    pages = load_document(ROOT / args.document)
    chunks = split_pages(pages)
    store = DocumentVectorStore(ROOT / "data" / "eval_chroma", collection_name="eval")
    store.add_chunks(chunks)

    questions = json.loads((ROOT / args.questions).read_text(encoding="utf-8"))
    rows = []
    started = time.perf_counter()

    for item in questions:
        question = item["question"]
        expected_keywords = item["expected_keywords"]
        hits = store.search(question, top_k=args.fetch_k)
        if args.rerank:
            hits = rerank_hits(question, hits, top_k=args.top_k)
        else:
            hits = hits[: args.top_k]

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
    report = {
        "document": args.document,
        "questions": len(rows),
        "top_k": args.top_k,
        "fetch_k": args.fetch_k,
        "rerank": args.rerank,
        "keyword_recall": round(keyword_hits / keyword_total, 4) if keyword_total else 0,
        "elapsed_seconds": round(elapsed, 3),
        "rows": rows,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
