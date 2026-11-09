"""FastAPI application for the network question-answering system."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .corpus import Corpus, load_corpus
from .generation import AnswerGenerator, ModelConfig
from .retrieval import HybridRetriever

STATIC = Path(__file__).with_name("static")


class Query(BaseModel):
    query: str = Field(min_length=2, max_length=1500)
    year: int | None = Field(default=None, ge=1900, le=2100)
    strategy: Literal["hybrid", "dense", "bm25"] = "hybrid"

    @field_validator("query", mode="before")
    @classmethod
    def trim_query(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


@dataclass
class Service:
    corpus: Corpus
    retriever: HybridRetriever
    generator: AnswerGenerator
    by_id: dict[str, dict]


def create_service(data_dir: str | Path | None = None, llm_config: ModelConfig | None = None) -> Service:
    corpus = load_corpus(data_dir or os.getenv("NETRAG_DATA_DIR") or None)
    retriever = HybridRetriever(corpus.documents, embedding_model=os.getenv("NETRAG_EMBEDDING_MODEL") or None,
                                reranker_model=os.getenv("NETRAG_RERANKER_MODEL") or None,
                                candidate_k=int(os.getenv("NETRAG_CANDIDATE_K", "20")),
                                bm25_weight=float(os.getenv("NETRAG_BM25_WEIGHT", "1")),
                                dense_weight=float(os.getenv("NETRAG_DENSE_WEIGHT", "1")),
                                use_jieba=os.getenv("NETRAG_USE_JIEBA", "0") == "1",
                                use_faiss=os.getenv("NETRAG_USE_FAISS", "0") == "1")
    return Service(corpus, retriever, AnswerGenerator(llm_config), {doc["id"]: doc for doc in corpus.documents})


def source_view(doc: dict, hit: dict | None = None) -> dict:
    public_fields = ("id", "kind", "year", "number", "question", "options", "answer", "analysis", "chapter", "tags",
                     "osi_layer", "tcpip_layer", "quality_flags", "source_url", "type", "score_points")
    view = {k: doc[k] for k in public_fields if k in doc}
    if hit is not None:
        view.update(score=hit["score"], channels=hit["channels"])
        view["ranking"] = {key: hit[key] for key in ("bm25_rank", "dense_rank", "bm25_score", "dense_score", "fused_score", "rerank_score") if key in hit}
    return view


def relevant_frequency(corpus: Corpus, docs: list[dict], limit: int = 6) -> list[dict]:
    tags = dict.fromkeys(tag for doc in docs for tag in doc["tags"] if tag in corpus.frequency)
    ordered = sorted(tags, key=lambda tag: -corpus.frequency[tag]["recent"])
    return [{"tag": tag, **corpus.frequency[tag]} for tag in ordered[:limit]]


def create_app(data_dir: str | Path | None = None, llm_config: ModelConfig | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = create_service(data_dir, llm_config)
        yield

    app = FastAPI(title="网络问答实验室 · NetRAG 408", version=__version__, lifespan=lifespan,
                  description="计算机网络考点与 408 真题的混合检索及来源可追溯问答")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def privacy_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        elif request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
            response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/", include_in_schema=False)
    def homepage():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health(request: Request):
        svc: Service = request.app.state.service
        exams = [doc for doc in svc.corpus.documents if doc["kind"] == "exam"]
        return {"status": "ok", "mode": svc.corpus.mode, "exam_count": len(exams),
                "concept_count": len(svc.corpus.documents) - len(exams), "years": svc.corpus.years,
                "dense_backend": svc.retriever.dense_backend, "reranker": svc.retriever.reranker_backend,
                "vector_index": svc.retriever.vector_index, "tokenizer": svc.retriever.tokenizer_backend,
                "llm_configured": svc.generator.config.configured}

    @app.get("/api/sources/{source_id}")
    def source(source_id: str, request: Request):
        svc: Service = request.app.state.service
        if source_id not in svc.by_id:
            raise HTTPException(404, "未找到此来源。")
        return source_view(svc.by_id[source_id])

    @app.get("/api/frequency")
    def frequency(request: Request):
        svc: Service = request.app.state.service
        return {"mode": svc.corpus.mode, "tags": [{"tag": tag, **stats} for tag, stats in
                sorted(svc.corpus.frequency.items(), key=lambda item: (-item[1]["recent"], item[0]))]}

    def run(query: Query, svc: Service, ask: bool):
        start = perf_counter()
        result = svc.retriever.search(query.query, k=3, year=query.year, strategy=query.strategy)
        sources = [source_view(hit["document"], hit) for hit in result["results"]]
        body = {"query_info": result["query_info"], "sources": sources,
                "frequency": relevant_frequency(svc.corpus, sources)}
        if ask:
            answer, mode = svc.generator.answer(query.query, sources, exact=bool(result["query_info"].get("exam_id")))
            body.update(answer=answer, mode=mode)
        body["elapsed_ms"] = round((perf_counter() - start) * 1000, 2)
        return body

    @app.post("/api/search")
    def search(query: Query, request: Request):
        return run(query, request.app.state.service, ask=False)

    @app.post("/api/ask")
    def ask(query: Query, request: Request):
        return run(query, request.app.state.service, ask=True)

    return app


app = create_app()
