"""Load original concept cards and optional locally imported exam questions."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEMO_PATH = Path(__file__).with_name("data") / "demo.jsonl"


@dataclass(frozen=True)
class Corpus:
    documents: list[dict]
    frequency: dict[str, dict[str, int]]
    mode: str
    years: list[int]


def read_jsonl(lines: Iterable[str], label: str) -> list[dict]:
    records = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{label}:{number} 不是有效的 JSON 记录") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{label}:{number} 应为 JSON 对象")
        records.append(record)
    if not records:
        raise ValueError(f"{label} 没有数据记录")
    return records


def normalize_document(record: dict, label: str) -> dict:
    for key in ("id", "question", "chapter", "kind"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"{label} 缺少有效字段 {key}")
    doc_id = record["id"]
    kind = record["kind"]
    if kind not in ("exam", "concept"):
        raise ValueError(f"{label} 的 kind 应为 exam 或 concept")

    year, number = record.get("year"), record.get("number")
    if kind == "exam":
        if (type(year) is not int or not 1900 <= year <= 9999
                or type(number) is not int or number < 1
                or doc_id != f"{year}_{number}"):
            raise ValueError(f"{label} 的年份、题号或 ID 不一致")
    elif year is not None or number is not None:
        raise ValueError(f"{label} 的原创知识卡不应标注真题年份或题号")

    tags = record.get("tags")
    if not isinstance(tags, list) or not tags or any(
        not isinstance(tag, str) or not tag.strip() for tag in tags
    ):
        raise ValueError(f"{label} 缺少有效考点列表")
    tags = list(dict.fromkeys(tag.strip() for tag in tags))

    options = record.get("options", {})
    if not isinstance(options, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in options.items()
    ):
        raise ValueError(f"{label} 的选项应为字符串字典")
    options = {key.strip(): value.strip() for key, value in options.items()}
    answer, analysis = record.get("answer", ""), record.get("analysis", "")
    if not isinstance(answer, str) or not isinstance(analysis, str):
        raise ValueError(f"{label} 的答案和解析应为字符串")
    answer, analysis = answer.strip(), analysis.strip()
    question, chapter = record["question"].strip(), record["chapter"].strip()
    parts = [question, *(f"{key}. {value}" for key, value in options.items())]
    if answer:
        parts.append(f"【答案】{answer}")
    if analysis:
        parts.append(f"【解析】{analysis}")
    parts.extend((f"【章节】{chapter}", f"【考点】{'、'.join(tags)}"))
    return {
        **record,
        "id": doc_id,
        "question": question,
        "chapter": chapter,
        "year": year,
        "number": number,
        "tags": tags,
        "options": options,
        "answer": answer,
        "analysis": analysis,
        "text": "\n".join(parts),
    }


def _read_documents(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as file:
        return [
            normalize_document(record, f"{path.name}/{record.get('id', '?')}")
            for record in read_jsonl(file, path.name)
        ]


def load_corpus(path: str | Path | None = None) -> Corpus:
    """Include original demo cards by default; a local path adds exam questions.

    Frequencies count exam questions containing each tag once per question.
    `recent` covers the latest exam year and its four preceding calendar years.
    """
    documents = _read_documents(DEMO_PATH)
    if path is not None:
        local = Path(path)
        if local.is_dir():
            local /= "documents.jsonl"
        documents.extend(_read_documents(local))

    ids = set()
    exams = []
    for doc in documents:
        if doc["id"] in ids:
            raise ValueError(f"题库存在重复文档 ID {doc['id']}")
        ids.add(doc["id"])
        if doc["kind"] == "exam":
            exams.append(doc)
    if not exams:
        return Corpus(documents, {}, "demo", [])

    years = sorted({doc["year"] for doc in exams})
    end = years[-1]
    start = end - 4
    total = Counter(tag for doc in exams for tag in doc["tags"])
    recent = Counter(tag for doc in exams if doc["year"] >= start for tag in doc["tags"])
    frequency = {
        tag: {"recent": recent[tag], "total": count, "start_year": start, "end_year": end}
        for tag, count in sorted(total.items())
    }
    return Corpus(documents, frequency, "exam", years)
