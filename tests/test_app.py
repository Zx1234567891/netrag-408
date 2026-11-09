import json
import os

import httpx
import pytest
from fastapi.testclient import TestClient

from netrag.app import create_app
from netrag.evaluate import retrieval_metrics
from netrag.generation import AnswerGenerator, ModelConfig, calculate


@pytest.fixture(autouse=True)
def clean_netrag_environment(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("NETRAG_"):
            monkeypatch.delenv(key)


@pytest.fixture
def client():
    with TestClient(create_app(llm_config=ModelConfig())) as client:
        yield client


def test_demo_routes_and_no_claimed_exam_frequency(client):
    home = client.get("/")
    assert home.status_code == 200
    assert "text/html" in home.headers["Content-Type"]
    health = client.get("/api/health").json()
    assert health["mode"] == "demo"
    assert health["exam_count"] == 0 and health["concept_count"] >= 38
    assert health["years"] == [] and not health["llm_configured"]
    search = client.post("/api/search", json={"query": "TCP 三次握手 SYN ACK"})
    assert search.status_code == 200
    assert search.headers["cache-control"] == "no-store"
    data = search.json()
    assert 1 <= len(data["sources"]) <= 3 and data["frequency"] == []
    source = data["sources"][0]
    assert source["kind"] == "concept" and source["channels"]
    assert client.get(f"/api/sources/{source['id']}").json()["id"] == source["id"]
    assert client.get("/api/sources/2024_38").status_code == 404


def test_exact_unavailable_exam_and_input_validation(client):
    missing = client.post("/api/ask", json={"query": "2024年第38题"}).json()
    assert missing["mode"] == "insufficient" and missing["sources"] == []
    assert client.post("/api/ask", json={"query": "  "}).status_code == 422
    assert client.post("/api/ask", json={"query": "TCP", "strategy": "invalid"}).status_code == 422


def test_exact_exam_and_frequency_api(tmp_path):
    doc = {"id": "2024_38", "kind": "exam", "year": 2024, "number": 38,
           "question": "一条 TCP 连接怎样确认数据？", "chapter": "运输层", "tags": ["TCP协议"],
           "answer": "D", "analysis": "使用累计确认。", "options": {"D": "累计确认"}}
    (tmp_path / "documents.jsonl").write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")
    with TestClient(create_app(tmp_path, ModelConfig())) as client:
        data = client.post("/api/ask", json={"query": "２０２４年第３８题"}).json()
        assert data["mode"] == "extractive"
        assert "[2024_38]" in data["answer"] and "累计确认" in data["answer"]
        assert [d["id"] for d in data["sources"]] == ["2024_38"]
        assert data["frequency"][0]["recent"] == 1
        wrong_year = client.post("/api/search", json={"query": "2024_38", "year": 2025}).json()
        assert wrong_year["sources"] == []


@pytest.mark.parametrize(("question", "expected"), [
    ("192.168.10.77/26 的网络地址、广播地址、可用主机数是什么？", ("192.168.10.64", "192.168.10.127", "62")),
    ("192.0.2.7/31 的可用主机地址是多少？", ("192.0.2.6", "192.0.2.7", "点到点")),
    ("192.0.2.0/24 等分为 8 个子网，192.0.2.101 属于哪个子网？", ("192.0.2.96/27", "32")),
    ("TCP 接收窗口 8 KB，拥塞窗口 6 KB，在途数据 2 KB，还能发送多少？", ("4096", "6144")),
    ("TCP rwnd=80 B，cwnd=60 B，flight=70 B，还能发送多少？", ("= 0",)),
    ("TCP 序列号 4294967290，载荷 20 字节，下一个 ACK 确认号是多少？", ("= 14",)),
    ("ＴＣＰ 的 rwnd 为 １．５ ＫＢ，cwnd 为 4 KB，在途数据 100 字节，还能发送多少？", ("1536", "= 1436")),
    ("TCP rwnd 8000 bit，cwnd 9000 比特，在途 1000 b，还能发送多少？", ("= 875",)),
    ("TCP 的接收窗口是 4,000 字节，拥塞窗口足够大，已发送未确认 1000 字节，还能发送多少？", ("= 3000",)),
    ("TCP seq 100，载荷 20 字节，带 FIN=0，确认号是什么？", ("= 120",)),
    ("TCP seq 100，载荷 20 字节，FIN 标志位为 1，确认号是什么？", ("= 121",)),
    ("１９８．５１．１００．７８／２７的网络地址和广播地址是什么？", ("198.51.100.64", "198.51.100.95")),
])
def test_calculators(question, expected):
    result = calculate(question)
    assert result is not None and all(text in result for text in expected)


def test_calculation_not_run_for_multiple_candidate_networks():
    assert calculate("192.0.2.0/25 与 192.0.2.128/25 聚合后广播地址是什么？") is None


@pytest.mark.parametrize("query", [
    "192.0.2.77/27.5 的网络地址是什么？",
    "192.0.2.77/27/24 的网络地址是什么？",
    "TCP seq 100，载荷 20 字节，FIN=2，确认号是什么？",
    "TCP rwnd 8000 bit/s，cwnd 9000 bit，在途 1000 bit，还能发送多少？",
    "TCP 接收窗口 4000 字节，在途 1000 字节，拥塞窗口足够大吗？最多还能发送多少？",
])
def test_calculation_rejects_ambiguous_or_invalid_parameters(query):
    assert calculate(query) is None


@pytest.mark.parametrize(("answer", "refs", "expected"), [
    ("SYN 同步初始序号。[tcp-handshake]", ["tcp-handshake"], "llm"),
    ("SYN 同步初始序号。[made-up]", ["made-up"], "extractive"),
    ("[tcp-handshake] [made-up]", ["tcp-handshake"], "extractive"),
])
def test_model_protocol_and_unknown_citations(answer, refs, expected):
    def provider(request):
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert "tcp-handshake" in body["messages"][1]["content"]
        return httpx.Response(200, json={"choices": [{"message": {
            "content": json.dumps({"answer": answer, "citations": refs}, ensure_ascii=False)}}]})

    doc = {"id": "tcp-handshake", "kind": "concept", "question": "TCP 握手？", "options": {},
           "answer": "同步序号。", "analysis": "SYN，SYN-ACK，ACK。"}
    with httpx.Client(transport=httpx.MockTransport(provider)) as http:
        _, mode = AnswerGenerator(ModelConfig("http://test.local/v1", "test-model"), http).answer("TCP 三次握手", [doc])
    assert mode == expected


def test_precision_denominator_and_context_order():
    assert retrieval_metrics(["x", "gold", "other"], {"gold"}) == (1 / 3, 1 / 2, 1, 1 / 2)
