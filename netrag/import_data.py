"""Import the networking subset from a locally supplied data-layer delivery.

Usage: python -m netrag.import_data SOURCE [--output local_data]
SOURCE may be the RAR, its extracted root, or its structured questions.jsonl.
RAR input needs libarchive's bsdtar (the built-in Windows tar also works).
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from .corpus import normalize_document, read_jsonl

SOURCE_NAME = "questions.jsonl"
STRUCTURED_MEMBER = f"data/structured/{SOURCE_NAME}".encode("ascii")
JSON_LIMIT = 24 * 1024 * 1024


def _tar_stdout(tar: str, flags: str, path: Path, limit: int, *patterns: str) -> bytes:
    with subprocess.Popen(
        [tar, flags, str(path), *patterns],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ) as process:
        assert process.stdout is not None
        payload = process.stdout.read(limit + 1)
        if len(payload) > limit:
            process.kill()
            raise ValueError("压缩包内容超出导入大小限制")
        if process.wait():
            raise ValueError("无法读取 RAR，请安装支持 RAR 的 bsdtar 或先手动解压")
    return payload


def _archive_source(path: Path) -> bytes:
    tar = shutil.which("bsdtar") or shutil.which("tar")
    if not tar:
        raise ValueError("找不到 bsdtar 或 tar，请安装 bsdtar 或先手动解压")
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("RAR 超出导入大小限制，请先手动解压")
    listing = _tar_stdout(tar, "-tf", path, 1024 * 1024)
    members = [
        name for name in listing.split(b"\n")
        if name.rstrip(b"\r") == STRUCTURED_MEMBER
        or name.rstrip(b"\r").endswith(b"/" + STRUCTURED_MEMBER)
    ]
    if len(members) != 1:
        raise ValueError("RAR 内应有且仅有一个 data/structured/questions.jsonl")
    # Match the ASCII suffix so that the member's localized root directory
    # works even when tar lists filenames in a legacy Windows code page.
    pattern = STRUCTURED_MEMBER.decode("ascii")
    if members[0].rstrip(b"\r") != STRUCTURED_MEMBER:
        pattern = "*/" + pattern
    return _tar_stdout(tar, "-xOf", path, JSON_LIMIT, pattern)


def _read_source(path: Path) -> list[dict]:
    if path.is_dir():
        candidates = [
            path / SOURCE_NAME,
            path / "data" / "structured" / SOURCE_NAME,
            *path.glob(f"*/data/structured/{SOURCE_NAME}"),
        ]
        files = [candidate for candidate in candidates if candidate.is_file()]
        if len(files) != 1:
            raise ValueError("目录中应有且仅有一个结构化 questions.jsonl")
        path = files[0]
    if path.suffix.lower() == ".rar":
        data = _archive_source(path)
    elif path.suffix.lower() == ".jsonl":
        with path.open("rb") as file:
            data = file.read(JSON_LIMIT + 1)
        if len(data) > JSON_LIMIT:
            raise ValueError("questions.jsonl 超出导入大小限制")
    else:
        raise ValueError("SOURCE 必须是 RAR、已解压目录或结构化 JSONL 文件")
    return read_jsonl(io.StringIO(data.decode("utf-8-sig")), SOURCE_NAME)


def _exam_document(row: dict) -> dict:
    label = f"{SOURCE_NAME}/{row.get('id', '?')}"
    metadata = row.get("metadata")
    flags = row.get("quality_flags", [])
    if not isinstance(metadata, dict):
        raise ValueError(f"{label} 缺少章节和考点元数据")
    if not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags):
        raise ValueError(f"{label} 的质量标记应为字符串列表")
    chapter_number = metadata.get("chapter")
    if type(chapter_number) is not int or not 1 <= chapter_number <= 6:
        raise ValueError(f"{label} 缺少有效的章节编号")

    analysis = row.get("analysis")
    errata = row.get("errata")
    if "source_answer_corrected" in flags and isinstance(errata, str) and errata.strip():
        # The old source explanation may still argue for the superseded answer.
        analysis = errata
    return normalize_document({
        "id": row.get("id"),
        "kind": "exam",
        "year": row.get("year"),
        "number": row.get("number"),
        "type": row.get("type"),
        "score_points": row.get("score"),
        "question": row.get("question"),
        "options": {} if row.get("options") is None else row["options"],
        "answer": "" if row.get("answer") is None else row["answer"],
        "analysis": "" if analysis is None else analysis,
        "chapter": metadata.get("chapter_name"),
        "chapter_number": chapter_number,
        "osi_layer": metadata.get("osi_layer"),
        "tcpip_layer": metadata.get("tcpip_layer"),
        "tags": metadata.get("exam_points"),
        "quality_flags": flags,
        "analysis_source": row.get("analysis_source"),
        "errata": errata,
    }, label)


def import_data(source: Path, output: Path) -> tuple[int, int]:
    """Write only normalized network records and return source and exam counts."""
    rows = _read_source(source.resolve())
    documents = []
    ids = set()
    for row in rows:
        subject = row.get("subject")
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError(f"{SOURCE_NAME}/{row.get('id', '?')} 缺少科目信息")
        if subject != "计算机网络":
            continue
        document = _exam_document(row)
        if document["id"] in ids:
            raise ValueError(f"原题库存在重复 ID {document['id']}")
        ids.add(document["id"])
        documents.append(document)
    if not documents:
        raise ValueError("题库中没有计算机网络题目")

    output.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", suffix=".tmp", dir=output, delete=False
        ) as file:
            temporary = Path(file.name)
            for document in documents:
                file.write(json.dumps(document, ensure_ascii=False) + "\n")
        temporary.replace(output / "documents.jsonl")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(rows), len(documents)


def main() -> None:
    parser = argparse.ArgumentParser(description="在本地导入数据层交付包的计算机网络题库")
    parser.add_argument("source", type=Path, help="RAR、解压目录或 questions.jsonl")
    parser.add_argument("--output", type=Path, default=Path("local_data"), help="输出目录")
    args = parser.parse_args()
    try:
        source_count, exam_count = import_data(args.source, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"导入失败：{exc}\n")
    print(f"已读取 {source_count} 条题库记录，导入 {exam_count} 道计算机网络真题。")
    print(f"本地题库：{args.output.resolve() / 'documents.jsonl'}")


if __name__ == "__main__":
    main()
