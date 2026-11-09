"""Record one generation run and score independently adjudicated answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .app import create_service
from .evaluate import evaluate, read_rows
from .generation import ModelConfig


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(cases_path: Path, output: Path, data_dir: Path | None, strategy: str, capture_drafts: bool) -> dict:
    cases = read_rows(cases_path)
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("生成评测用例 ID 必须唯一")
    config = ModelConfig.from_env()
    if not config.configured:
        raise ValueError("请先设置 NETRAG_LLM_BASE_URL 和 NETRAG_LLM_MODEL")
    svc = create_service(data_dir, config)
    if (svc.corpus.mode == "exam" or capture_drafts) and not output.resolve().is_relative_to(Path("eval_local").resolve()):
        raise ValueError("真题和模型原始稿的运行记录必须保存在忽略提交的 eval_local 目录")
    for case in cases:
        missing = set(case["evidence_ids"]) - svc.by_id.keys()
        if missing:
            raise ValueError(f"{case['id']} 的参考证据不在当前语料库: {sorted(missing)}")

    attempts: list[dict] = []

    def record_request(request: httpx.Request) -> None:
        body = json.loads(request.content)
        settings = {key: body[key] for key in ("temperature", "max_tokens", "top_p", "seed", "response_format") if key in body}
        attempts.append({"http_status": None, "request_parameters": settings})

    def record_response(response: httpx.Response) -> None:
        attempt = attempts[-1]
        attempt["http_status"] = response.status_code
        response.read()
        if not response.is_success:
            return
        try:
            payload = response.json()
            choice = payload["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError):
            return
        attempt.update(model=payload.get("model"), finish_reason=choice.get("finish_reason"),
                       completion_received=isinstance(content, str) and bool(content.strip()))
        if not isinstance(content, str):
            return
        if capture_drafts:
            attempt["draft"] = content
        try:
            declared = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I))
        except ValueError:
            return
        attempt["declared_abstention"] = (
            isinstance(declared, dict) and declared.get("insufficient") is True
            and declared.get("citations") == [] and declared.get("answer") == "资料不足"
        )

    details = []
    with httpx.Client(event_hooks={"request": [record_request], "response": [record_response]}) as client:
        svc.generator.client = client
        for case in cases:
            attempts.clear()
            row = evaluate([case], svc, strategy)["details"][0]
            row["category"] = case["category"]
            row["expected_abstention"] = case.get("expect_insufficient", False)
            row["model_attempts"] = attempts.copy()
            row["model_abstention_accepted"] = row["mode"] == "insufficient" and any(
                attempt.get("declared_abstention", False) for attempt in attempts
            )
            details.append(row)

    report = {
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "cases_file": cases_path.name,
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": config.model,
        "model_timeout_seconds": config.timeout,
        "corpus_mode": svc.corpus.mode,
        "strategy": strategy,
        "dense_backend": svc.retriever.dense_backend,
        "reranker": svc.retriever.reranker_backend,
        "vector_index": svc.retriever.vector_index,
        "tokenizer": svc.retriever.tokenizer_backend,
        "model_request_parameters": next((attempt["request_parameters"] for row in details for attempt in row["model_attempts"]), None),
        "details": details,
    }
    write_json(output, report)
    return report


def rate(rows: list[dict], field: str) -> dict:
    numerator = sum(row[field] for row in rows)
    return {"count": numerator, "total": len(rows), "rate": round(numerator / len(rows), 6) if rows else None}


def telemetry(details: list[dict]) -> dict:
    attempts = [attempt for row in details for attempt in row["model_attempts"]]
    latencies = sorted(row["elapsed_ms"] for row in details)
    return {
        "cases": len(details),
        "modes": dict(Counter(row["mode"] for row in details)),
        "model_calls": len(attempts),
        "cases_with_model_call": sum(bool(row["model_attempts"]) for row in details),
        "accepted_model_answers": sum(row["mode"] == "llm" for row in details),
        "accepted_model_abstentions": sum(row["model_abstention_accepted"] for row in details),
        "completions_replaced_by_fallback": sum(
            row["mode"] != "llm" and not row["model_abstention_accepted"]
            and any(attempt.get("completion_received") for attempt in row["model_attempts"])
            for row in details
        ),
        "calls_without_completion": sum(not attempt.get("completion_received") for attempt in attempts),
        "p95_ms": latencies[math.ceil(0.95 * len(latencies)) - 1],
        "max_ms": latencies[-1],
        "within_5s": sum(ms <= 5000 for ms in latencies),
    }


def score(run_data: dict, judgments: list[dict]) -> dict:
    details = run_data["details"]
    reviewers = {review["id"]: review for review in judgments}
    if len(reviewers) != len(judgments) or reviewers.keys() != {row["id"] for row in details}:
        raise ValueError("每个用例必须恰好有一条评审记录")
    joined = []
    for row in details:
        review = reviewers[row["id"]]
        digest = hashlib.sha256(row["answer"].encode()).hexdigest()
        if review["answer_sha256"] != digest or row["answer_sha256"] != digest:
            raise ValueError(f"{row['id']} 的答案与评审摘要不匹配")
        if (any(type(review[key]) is not bool for key in ("correct", "hallucinated"))
                or not isinstance(review.get("reason"), str) or not review["reason"].strip()):
            raise ValueError(f"{row['id']} 的评审需要两个布尔判定和理由")
        if review["correct"] and review["hallucinated"]:
            raise ValueError(f"{row['id']} 的评审相互矛盾")
        joined.append({**row, "correct": review["correct"], "hallucinated": review["hallucinated"]})

    slices = {
        "system_all": joined,
        "system_answerable": [row for row in joined if not row["expected_abstention"]],
        "system_abstention": [row for row in joined if row["expected_abstention"]],
        "model_answers": [row for row in joined if row["mode"] == "llm"],
        "model_accepted": [row for row in joined if row["mode"] == "llm" or row["model_abstention_accepted"]],
    }
    return {
        "model": run_data["model"],
        "model_request_parameters": run_data.get("model_request_parameters"),
        "corpus_mode": run_data["corpus_mode"],
        "strategy": run_data["strategy"],
        "cases_sha256": run_data["cases_sha256"],
        "telemetry": telemetry(details),
        "scores": {name: {"answer_accuracy": rate(rows, "correct"), "hallucination_rate": rate(rows, "hallucinated")}
                   for name, rows in slices.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    record = subcommands.add_parser("run", help="调用模型并保存一次系统输出")
    record.add_argument("cases", type=Path)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--data-dir", type=Path)
    record.add_argument("--strategy", choices=("hybrid", "dense", "bm25"), default="hybrid")
    record.add_argument("--capture-drafts", action="store_true", help="把模型原始稿保存在 eval_local 中以排查拒答和验证失败")
    review = subcommands.add_parser("score", help="汇总已经记录的评审，不重新调用模型")
    review.add_argument("run", type=Path)
    review.add_argument("judgments", type=Path)
    review.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        data = run(args.cases, args.output, args.data_dir, args.strategy, args.capture_drafts)
        summary = telemetry(data["details"])
    else:
        data = json.loads(args.run.read_text(encoding="utf-8-sig"))
        summary = score(data, read_rows(args.judgments))
        if args.output:
            write_json(args.output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
