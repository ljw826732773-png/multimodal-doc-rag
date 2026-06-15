from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import run_retrieval_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small RAG evaluation.")
    parser.add_argument("--document", default="examples/sample.txt")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--fetch-k", type=int, default=8)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--retrieval-mode", choices=["vector", "hybrid"], default="vector")
    parser.add_argument("--router", action="store_true")
    parser.add_argument("--mmr", action="store_true")
    parser.add_argument("--multi-query", action="store_true")
    args = parser.parse_args()

    report = run_retrieval_eval(
        document_path=ROOT / args.document,
        questions_path=ROOT / args.questions,
        persist_dir=ROOT / "data" / "eval_chroma",
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        use_rerank=args.rerank,
        retrieval_mode=args.retrieval_mode,
        use_router=args.router,
        use_mmr=args.mmr,
        use_multi_query=args.multi_query,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
