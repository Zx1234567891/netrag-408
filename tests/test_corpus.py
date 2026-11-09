import json

import pytest

from netrag.corpus import load_corpus
from netrag.import_data import import_data


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )


def exam(year, number, **changes):
    return {
        "id": f"{year}_{number}",
        "year": year,
        "number": number,
        "subject": "计算机网络",
        "question": "这是一条用于验证导入格式的测试记录。",
        "options": {"A": "选项一", "B": "选项二"},
        "answer": "B",
        "analysis": None,
        "metadata": {
            "chapter": 5,
            "chapter_name": "传输层",
            "exam_points": ["TCP协议", "TCP协议"],
        },
        **changes,
    }


def test_original_corpus_has_no_exam_counts():
    corpus = load_corpus()

    assert corpus.mode == "demo"
    assert corpus.frequency == {}
    assert corpus.years == []
    assert all(doc["kind"] == "concept" and doc["year"] is None for doc in corpus.documents)
    assert {doc["chapter_number"] for doc in corpus.documents} == set(range(1, 7))


def test_import_preserves_exam_scope_and_excludes_original_cards_from_frequency(tmp_path):
    structured = tmp_path / "extracted" / "data" / "structured"
    structured.mkdir(parents=True)
    write_jsonl(structured / "questions.jsonl", [
        exam(2019, 33),
        exam(2022, 33),
        exam(2026, 37, analysis="旧解析", errata="勘误依据", quality_flags=["source_answer_corrected"]),
        {"id": "2026_1", "subject": "数据结构"},
    ])
    output = tmp_path / "local_data"

    assert import_data(tmp_path / "extracted", output) == (4, 3)
    corpus = load_corpus(output)
    assert corpus.mode == "exam"
    assert corpus.years == [2019, 2022, 2026]
    assert corpus.frequency == {"TCP协议": {"recent": 2, "total": 3, "start_year": 2022, "end_year": 2026}}
    corrected = next(doc for doc in corpus.documents if doc["id"] == "2026_37")
    assert corrected["analysis"] == "勘误依据"
    assert "旧解析" not in corrected["text"]
    assert all(doc["id"] != "2026_1" for doc in corpus.documents)


def test_rejecting_duplicate_input_keeps_last_good_output(tmp_path):
    source = tmp_path / "questions.jsonl"
    output = tmp_path / "local_data"
    record = exam(2026, 33)
    write_jsonl(source, [record])
    import_data(source, output)
    before = (output / "documents.jsonl").read_bytes()
    write_jsonl(source, [record, record])

    with pytest.raises(ValueError, match="重复 ID 2026_33"):
        import_data(source, output)
    assert (output / "documents.jsonl").read_bytes() == before
