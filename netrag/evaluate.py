"""Evaluate ranking, calculations and separately reviewed answers on JSONL cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from time import perf_counter

from .app import create_service, source_view
from .corpus import read_jsonl
from .generation import ModelConfig


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as stream:
        return read_jsonl(stream, str(path))


def retrieval_metrics(returned: list[str], relevant: set[str], k: int = 3) -> tuple[float, float, float, float]:
    """Fixed-denominator precision, mean precision at relevant ranks, hit and reciprocal rank."""
    positions = [rank for rank, doc_id in enumerate(returned[:k], 1) if doc_id in relevant]
    precision = len(positions) / k
    # Precision at each relevant position, normalised by the number of relevant
    # candidates returned. Zero when none are relevant.
    context_precision = statistics.mean(i / rank for i, rank in enumerate(positions, 1)) if positions else 0.0
    return precision, context_precision, float(bool(positions)), 1 / positions[0] if positions else 0.0


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def evaluate(cases: list[dict], svc, strategy: str, judgments: dict[str, dict] | None = None) -> dict:
    detail = []
    precision, context, hits, reciprocal = [], [], [], []
    assertions, abstentions, accuracy, hallucinations = [], [], [], []
    judgments = judgments or {}
    seen = set()
    for case in cases:
        name, query = case["id"], case["query"]
        if name in seen:
            raise ValueError(f"重复的评测 ID: {name}")
        seen.add(name)
        relevant = set(case.get("relevant_ids", []))
        missing = relevant - svc.by_id.keys()
        if missing:
            raise ValueError(f"{name} 的相关来源不在当前语料库: {sorted(missing)}")
        start = perf_counter()
        result = svc.retriever.search(query, k=3, year=case.get("year"), strategy=strategy)
        sources = [source_view(hit["document"], hit) for hit in result["results"]]
        answer, mode = svc.generator.answer(query, sources, exact=bool(result["query_info"].get("exam_id")))
        ms = (perf_counter() - start) * 1000
        ids = [source["id"] for source in sources]
        digest = hashlib.sha256(answer.encode()).hexdigest()
        row = {"id": name, "query": query, "source_ids": ids, "answer": answer, "answer_sha256": digest,
               "mode": mode, "elapsed_ms": round(ms, 3)}
        if relevant:
            p, cp, hit, rr = retrieval_metrics(ids, relevant)
            precision.append(p)
            context.append(cp)
            hits.append(hit)
            reciprocal.append(rr)
            row.update(precision_at_3=p, context_precision=cp, relevant_ids=sorted(relevant))
        if "answer_contains" in case:
            passed = all(token.casefold() in answer.casefold() for token in case["answer_contains"])
            row["answer_assertions_pass"] = passed
            assertions.append(float(passed))
        if case.get("expect_insufficient"):
            abstentions.append(float(mode == "insufficient"))
            row["abstention_pass"] = mode == "insufficient"
        if name in judgments:
            review = judgments[name]
            if review.get("answer_sha256") != digest:
                raise ValueError(f"{name} 的人工评审针对不同答案；请重新评审")
            if type(review.get("correct")) is not bool or type(review.get("hallucinated")) is not bool:
                raise ValueError(f"{name} 的人工评审需要布尔字段 correct 和 hallucinated")
            accuracy.append(float(review["correct"]))
            hallucinations.append(float(review["hallucinated"]))
        detail.append(row)
    extra_reviews = judgments.keys() - seen
    if extra_reviews:
        raise ValueError(f"评审包含未知案例: {sorted(extra_reviews)}")
    elapsed = sorted(row["elapsed_ms"] for row in detail)
    return {
        "strategy": strategy, "corpus_mode": svc.corpus.mode, "dense_backend": svc.retriever.dense_backend,
        "reranker": svc.retriever.reranker_backend, "vector_index": svc.retriever.vector_index,
        "tokenizer": svc.retriever.tokenizer_backend, "cases": len(cases), "ranking_cases": len(precision),
        "precision_at_3": _mean(precision), "context_precision": _mean(context), "hit_at_3": _mean(hits),
        "mrr_at_3": _mean(reciprocal), "assertion_cases": len(assertions), "assertion_pass_rate": _mean(assertions),
        "abstention_cases": len(abstentions), "abstention_pass_rate": _mean(abstentions),
        "human_judgments": len(accuracy), "answer_accuracy": _mean(accuracy), "hallucination_rate": _mean(hallucinations),
        "p50_ms": round(statistics.median(elapsed), 3), "p95_ms": elapsed[math.ceil(0.95 * len(elapsed)) - 1],
        "max_ms": elapsed[-1], "fraction_within_5s": _mean([float(ms <= 5000) for ms in elapsed]), "details": detail,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, help="JSONL 检索标注与断言用例")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--strategy", choices=("hybrid", "dense", "bm25", "all"), default="all")
    parser.add_argument("--judgments", type=Path, help="仅在单一策略下使用的人工审查 JSONL")
    parser.add_argument("--allow-llm", action="store_true", help="允许使用环境变量设置的生成服务")
    parser.add_argument("--output", type=Path, help="另存含答案的详细评测记录")
    args = parser.parse_args()
    if args.judgments and args.strategy == "all":
        parser.error("--judgments 需要指定一个 --strategy")
    cases = read_rows(args.cases)
    review_rows = read_rows(args.judgments) if args.judgments else []
    reviews = {row["id"]: row for row in review_rows}
    if len(reviews) != len(review_rows):
        parser.error("人工评审包含重复 ID")
    service = create_service(args.data_dir, None if args.allow_llm else ModelConfig())
    strategies = ("bm25", "dense", "hybrid") if args.strategy == "all" else (args.strategy,)
    results = [evaluate(cases, service, name, reviews) for name in strategies]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{k: v for k, v in result.items() if k != "details"} for result in results], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
