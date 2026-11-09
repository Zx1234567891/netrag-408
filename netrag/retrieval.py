"""Query rewriting, BM25 and cosine retrieval, RRF, and optional neural reranking."""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any


BGE_EMBEDDING = "BAAI/bge-small-zh-v1.5"
BGE_RERANKER = "BAAI/bge-reranker-base"
_BGE_QUERY_PROMPT = "为这个句子生成表示以用于检索相关文章："
_LOG = logging.getLogger(__name__)

# The first phrase is the preferred name. Chapter numbers follow the course taxonomy.
_TERMS = (
    (1, "OSI七层模型", "七层模型", "OSI参考模型", "开放系统互连", "osi"),
    (1, "TCP/IP体系结构", "四层模型", "五层模型", "TCP/IP模型", "协议栈", "tcp/ip"),
    (1, "协议与服务", "协议与接口", "服务访问点", "网络协议"),
    (1, "网络性能指标", "吞吐量", "信道利用率", "网络性能"),
    (1, "时延带宽积", "时延和带宽的乘积", "带宽时延积"),
    (1, "端到端时延", "发送时延", "传播时延", "传输时延", "排队时延"),
    (2, "奈奎斯特定理", "奈氏准则", "奈奎斯特", "nyquist"),
    (2, "香农定理", "香农公式", "shannon", "信噪比"),
    (2, "曼彻斯特编码", "曼彻斯特", "manchester"),
    (2, "信道复用", "频分复用", "时分复用", "波分复用", "码分复用", "cdma"),
    (2, "电路交换", "分组交换", "报文交换", "存储转发"),
    (3, "CSMA/CD", "以太网碰撞", "冲突检测", "碰撞检测", "最短帧长", "最小帧长", "帧最短", "帧最小", "二进制指数退避"),
    (3, "CSMA/CA", "碰撞避免", "wifi退避", "隐藏站", "隐蔽站", "rts/cts"),
    (3, "CRC循环冗余校验", "循环冗余检验", "循环冗余校验", "生成多项式", "crc"),
    (3, "海明码", "汉明码", "海明距离", "汉明距离", "hamming"),
    (3, "组帧与透明传输", "透明传输", "字节填充", "比特填充", "零比特填充"),
    (3, "数据链路层滑动窗口", "后退n帧", "回退n帧", "回退n", "选择重传", "gbn", "go-back-n"),
    (3, "以太网交换机", "交换机自学习", "mac地址表", "mac表", "交换机转发表", "二层交换机", "广播域", "冲突域"),
    (3, "虚拟局域网", "vlan", "802.1q"),
    (4, "子网划分与CIDR", "子网掩码", "子网划分", "无类域间路由", "地址块", "网络号", "主机位", "主机数", "cidr", "vlsm"),
    (4, "路由聚合", "路由汇总", "超网", "前缀聚合", "合并网段", "地址聚合", "路由摘要"),
    (4, "最长前缀匹配", "最长匹配", "最长网络前缀", "最长匹配原则"),
    (4, "IP数据报与分片", "ip分片", "数据报分片", "数据包太大", "最大传输单元", "片偏移", "mtu"),
    (4, "IPv4地址与分类", "ipv4地址", "地址分类", "a类地址", "b类地址", "c类地址", "ipv4"),
    (4, "IPv6协议", "ipv6"),
    (4, "ARP地址解析", "找mac地址", "知道ip找mac", "ip转mac", "ip查mac", "地址解析协议", "arp"),
    (4, "ICMP控制报文", "网际控制报文", "icmp", "traceroute", "ping"),
    (4, "NAT网络地址转换", "地址转换", "端口映射", "nat", "napt"),
    (4, "RIP距离向量路由", "距离向量", "跳数", "rip"),
    (4, "OSPF链路状态路由", "链路状态", "开放最短路径优先", "ospf"),
    (4, "BGP边界网关协议", "自治系统间路由", "路径向量", "bgp"),
    (5, "TCP流量控制", "接收窗口", "接收方窗口", "接收方限速", "接收缓冲区", "零窗口", "rwnd", "win"),
    (5, "TCP拥塞控制", "拥塞窗口", "慢开始", "慢启动", "拥塞避免", "网络太堵", "快恢复", "cwnd", "ssthresh"),
    (5, "TCP可靠传输", "累计确认", "累积确认", "超时重传", "快重传", "重复确认", "确认号", "序号确认", "丢包重发"),
    (5, "TCP连接建立", "tcp连接管理", "三次握手", "三次交互", "建立tcp连接", "建立连接", "握手", "syn"),
    (5, "TCP连接释放", "四次挥手", "关闭tcp连接", "关闭连接", "断开连接", "挥手", "time_wait", "fin"),
    (5, "UDP用户数据报协议", "用户数据报协议", "udp"),
    (6, "DNS域名解析", "域名解析", "域名查ip", "网址怎么变成ip", "递归查询", "迭代查询", "dns"),
    (6, "DHCP动态主机配置", "自动分配ip", "自动获取ip", "动态分配地址", "dhcp"),
    (6, "HTTP超文本传输协议", "持久连接", "非持久连接", "http", "https"),
    (6, "电子邮件协议", "邮件协议", "smtp", "pop3", "imap"),
    (6, "FTP文件传输协议", "ftp", "文件传输协议"),
)

_CJK = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_PARTS = re.compile(rf"[{_CJK}]+|[a-z][a-z0-9]*(?:[._/-][a-z0-9]+)*|\d+(?:\.\d+)*(?:/\d+)?")
_CJK_CHAR = re.compile(rf"[{_CJK}]")
_STOP = frozenset("什么 如何 怎样 怎么 哪些 是否 为什么 多少 下面 下列 关于 计算 说明 以下 有关 分别 试问 答案 解析 考研 真题 请问 参考 第几 这一 一个 两个 这个 那个 可以 能否 该题 408 what how why which a an the of to in for and or is be".split())
_EXAM_ID = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*(?:[_-]\s*([1-9]\d?)(?![0-9a-z])|年?\s*(?:考研\s*)?(?:408\s*)?(?:真题\s*)?(?:第\s*)?([1-9]\d?)\s*(?:题|问))")
_CHAPTERS = {
    1: ("计算机网络体系结构", "计算机网络概述", "网络体系结构", "网络概述"),
    2: ("物理层",),
    3: ("数据链路层", "链路层"),
    4: ("网络层", "网际层"),
    5: ("传输层", "运输层"),
    6: ("应用层",),
}


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    # Normalize spaces at Chinese word boundaries and around Latin terms inside Chinese.
    normalized = re.sub(rf"(?<=[{_CJK}])[ \t]+(?=[{_CJK}a-z0-9])|(?<=[a-z0-9])[ \t]+(?=[{_CJK}])", "", normalized)
    return " ".join(normalized.split())


def _term_pattern(term: str) -> str:
    term = _normalize(term)
    before = r"(?<![a-z0-9])" if term[0].isascii() and term[0].isalnum() else ""
    after = r"(?![a-z0-9])" if term[-1].isascii() and term[-1].isalnum() else ""
    return before + re.escape(term) + after


_LEXICON = tuple(
    (chapter, terms[0], re.compile("|".join(_term_pattern(term) for term in terms)))
    for chapter, *terms in _TERMS
)
_WINDOW_CONTEXT = (
    (3, "数据链路层滑动窗口", re.compile(r"数据链路层|帧|(?<![a-z0-9])(?:gbn|sr)(?![a-z0-9])")),
    (5, "TCP流量控制", re.compile(r"接收方|拥塞窗口|(?<![a-z0-9])tcp(?![a-z0-9/])")),
)


def _tokens(text: str) -> list[str]:
    tokens = []
    for part in _PARTS.findall(text):
        if _CJK_CHAR.fullmatch(part[0]):
            # Chinese bigrams make exact terms searchable without an external dictionary.
            terms = (part[i : i + 2] for i in range(len(part) - 1))
            tokens.extend(term for term in terms if term not in _STOP)
        elif part not in _STOP:
            tokens.append(part)
            if any(separator in part for separator in "._/-"):
                tokens.extend(piece for piece in re.split(r"[._/-]", part) if len(piece) > 1)
    return tokens


def _concepts(text: str) -> list[tuple[int, str]]:
    concepts = [(chapter, term) for chapter, term, pattern in _LEXICON if pattern.search(text)]
    if "滑动窗口" in text:
        for chapter, term, context in _WINDOW_CONTEXT:
            if (chapter, term) not in concepts and context.search(text):
                concepts.append((chapter, term))
    return concepts


@dataclass(frozen=True)
class _Query:
    normalized: str
    expanded: str
    aliases: tuple[str, ...]
    chapters: frozenset[int]
    exam_id: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "normalized": self.normalized,
            "expanded": self.expanded,
            "aliases": list(self.aliases),
            "chapters": sorted(self.chapters),
            "exam_id": self.exam_id,
        }


def _rewrite(query: str) -> _Query:
    normalized = _normalize(query)
    concepts = _concepts(normalized)
    aliases = tuple(term for _, term in concepts if _normalize(term) not in normalized)[:8]
    expanded = " ".join((normalized, *aliases)).strip()
    chapter_counts = Counter(chapter for chapter, _ in concepts)
    peak = max(chapter_counts.values(), default=0)
    chapters = frozenset(chapter for chapter, count in chapter_counts.items() if count == peak)
    exam = _EXAM_ID.search(normalized)
    exam_id = f"{exam[1]}_{int(exam[2] or exam[3])}" if exam else None
    return _Query(normalized, expanded, aliases, chapters, exam_id)


def rewrite_query(query: str) -> dict[str, Any]:
    """Expose the same rewrite used in retrieval for evaluation and UI inspection."""
    return _rewrite(query).as_dict()


def _chapter_number(chapter: str) -> int | None:
    chapter = _normalize(chapter)
    explicit = re.match(r"(?:第\s*)?([1-6])\s*(?:章|[:.、]|$)", chapter)
    if explicit:
        return int(explicit[1])
    return next((number for number, names in _CHAPTERS.items() if any(name in chapter for name in names)), None)


def _document_text(document: dict) -> str:
    text = str(document.get("text") or "").strip()
    if not text:
        options = " ".join(str(option) for option in (document.get("options") or {}).values())
        text = "\n".join(str(part or "") for part in (document.get("question"), options, document.get("answer"), document.get("analysis")))
    return "\n".join((str(document.get("chapter") or ""), " ".join(document.get("tags") or []), text)).strip()


class HybridRetriever:
    """In-memory retrieval for the small course corpus.

    Set embedding_model to a SentenceTransformers ID or local path for dense
    embeddings, and reranker_model to a CrossEncoder ID or local path for
    learned reranking. Defaults require no model downloads.
    """

    def __init__(
        self,
        documents: list[dict],
        *,
        embedding_model: str | None = None,
        reranker_model: str | None = None,
        dense_threshold: float | None = None,
        candidate_k: int = 20,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        use_jieba: bool = False,
        use_faiss: bool = False,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k must be positive")
        if any(not math.isfinite(weight) or weight <= 0 for weight in (bm25_weight, dense_weight)):
            raise ValueError("RRF weights must be positive finite numbers")
        if use_faiss and not embedding_model:
            raise ValueError("FAISS requires a configured embedding model")
        self._tokenize = _tokens
        self.tokenizer_backend = "cjk-ngram"
        if use_jieba:
            import jieba

            segmenter = jieba.Tokenizer()
            segmenter.initialize()

            def jieba_tokens(text: str) -> list[str]:
                # Retain the deterministic bigrams for protocol spellings and rare terms.
                return _tokens(text) + [word for word in segmenter.lcut(text, HMM=False)
                                        if len(word) > 2 and re.fullmatch(r"[\u3400-\u9fff]+", word)]

            self._tokenize = jieba_tokens
            self.tokenizer_backend = "jieba+cjk-ngram"
        self.documents = list(documents)
        self._passages = [_document_text(document) for document in documents]
        self._chapters = [_chapter_number(str(document.get("chapter") or "")) for document in documents]
        self._ids = {document["id"]: i for i, document in enumerate(documents)}
        if len(self._ids) != len(self.documents):
            raise ValueError("duplicate document ID")
        self._candidate_k = candidate_k
        self._bm25_weight = bm25_weight
        self._dense_weight = dense_weight
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lengths: list[int] = []
        self._document_concepts: list[set[str]] = []
        token_counts = []
        for i, passage in enumerate(self._passages):
            normalized = _normalize(passage)
            counts = Counter(self._tokenize(normalized))
            self._lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self._postings[term].append((i, count))
            concepts = {term for _, term in _concepts(normalized)}
            self._document_concepts.append(concepts)
            token_counts.append(counts)
        n = len(documents)
        self._avg_length = max(1.0, sum(self._lengths) / max(1, n))
        self._idf = {term: math.log1p((n - len(postings) + 0.5) / (len(postings) + 0.5)) for term, postings in self._postings.items()}
        self._encoder = None
        self._reranker = None
        self._faiss = None
        self.vector_index = "in-memory"
        self.dense_backend = embedding_model or "lexical-concept-cosine"
        self.reranker_backend = reranker_model or "rrf-chapter-affinity"
        self._dense_threshold = dense_threshold if dense_threshold is not None else (0.5 if embedding_model else 0.07)
        if embedding_model:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Dense model requires sentence-transformers") from exc
            self._encoder = SentenceTransformer(embedding_model, trust_remote_code=False)
            self._vectors = self._encoder.encode(self._passages, normalize_embeddings=True, show_progress_bar=False) if self._passages else None
            if use_faiss and self._vectors is not None:
                import faiss

                self._faiss = faiss.IndexFlatIP(self._vectors.shape[1])
                self._faiss.add(self._vectors.astype("float32", copy=False))
                self.vector_index = "faiss-flat-ip"
        else:
            self._vectors = [self._concept_vector(counts, concepts) for counts, concepts in zip(token_counts, self._document_concepts)]
        if reranker_model:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("Reranker model requires sentence-transformers") from exc
            self._reranker = CrossEncoder(reranker_model, trust_remote_code=False)

    def _concept_vector(self, counts: Counter[str], concepts: set[str]) -> dict[str, float]:
        vector = {
            token: math.sqrt(count) * (0.5 + math.sqrt(self._idf[token]))
            for token, count in counts.items()
            if token in self._idf
        }
        for term in sorted(concepts):
            vector["#" + term] = 3.0
        magnitude = math.sqrt(sum(value**2 for value in vector.values()))
        return {token: weight / magnitude for token, weight in vector.items()} if magnitude else {}

    def _lexical_support(self, original: set[str], concepts: set[str], allowed: set[int]) -> set[int]:
        hits: Counter[int] = Counter()
        for term in original:
            hits.update(i for i, _ in self._postings.get(term, ()) if i in allowed)
        required = max(min(3, len(original)), math.ceil(0.2 * len(original)))
        return {i for i in allowed if concepts & self._document_concepts[i] or (required > 0 and hits[i] >= required)}

    def _bm25(self, query: _Query, original: set[str], allowed: set[int]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        expanded = {token for alias in query.aliases for token in self._tokenize(_normalize(alias))} - original
        for term in sorted(original | expanded):
            boost = 1.0 if term in original else 0.65
            for i, count in self._postings.get(term, ()):
                if i in allowed:
                    length_factor = 0.25 + 0.75 * self._lengths[i] / self._avg_length
                    scores[i] += boost * self._idf[term] * count * 2.5 / (count + 1.5 * length_factor)
        return scores

    def _cosines(self, query: _Query, concepts: set[str], allowed: set[int], supported: set[int]) -> dict[int, float]:
        if self._encoder is not None:
            query_text = _BGE_QUERY_PROMPT + query.expanded if "bge-small-zh" in self.dense_backend.casefold() else query.expanded
            query_vector = self._encoder.encode([query_text], normalize_embeddings=True, show_progress_bar=False)[0]
            if self._faiss is not None:
                similarities, indices = self._faiss.search(query_vector[None, :].astype("float32", copy=False), len(self.documents))
                scores = {int(i): float(score) for i, score in zip(indices[0], similarities[0]) if i in allowed}
            else:
                similarities = self._vectors @ query_vector
                scores = {i: float(similarities[i]) for i in allowed}
        else:
            counts = Counter(self._tokenize(query.normalized))
            for alias in query.aliases:
                counts.update(self._tokenize(_normalize(alias)))
            query_vector = self._concept_vector(counts, concepts)
            scores = {i: sum(weight * self._vectors[i].get(term, 0.0) for term, weight in query_vector.items()) for i in supported}
        # A learned encoder can recognize unseen paraphrases, but demands a stronger score when neither terms nor concept labels corroborate a match.
        return {i: score for i, score in scores.items() if math.isfinite(score) and score >= (self._dense_threshold if i in supported else max(0.65, self._dense_threshold))}

    def _rank(self, scores: dict[int, float], k: int) -> dict[int, int]:
        ordered = sorted(scores, key=lambda i: (-scores[i], self.documents[i]["id"]))
        return {i: rank for rank, i in enumerate(ordered[:k], start=1)}

    def search(self, query: str, k: int = 3, *, year: int | None = None, strategy: str = "hybrid") -> dict:
        """Return ranked document records and signals without mutating the corpus."""
        if strategy not in {"hybrid", "dense", "bm25"}:
            raise ValueError(f"unsupported strategy: {strategy}")
        if k < 1:
            raise ValueError("k must be positive")
        rewritten = _rewrite(query)
        query_info = rewritten.as_dict()
        allowed = {i for i, document in enumerate(self.documents) if year is None or document.get("year") == year}
        if not rewritten.normalized or not allowed:
            return {"query_info": query_info, "results": []}
        if rewritten.exam_id:
            index = self._ids.get(rewritten.exam_id)
            results = [{"document": self.documents[index], "score": 1.0, "channels": ["exact"]}] if index in allowed else []
            return {"query_info": query_info, "results": results}

        limit = max(k, self._candidate_k)
        original = set(self._tokenize(rewritten.normalized))
        concepts = {term for _, term in _concepts(rewritten.normalized)}
        supported = self._lexical_support(original, concepts, allowed)
        bm25 = self._bm25(rewritten, original, supported) if strategy != "dense" else {}
        dense = self._cosines(rewritten, concepts, allowed, supported) if strategy != "bm25" else {}
        bm25_rank = self._rank(bm25, limit)
        dense_rank = self._rank(dense, limit)
        candidates = bm25_rank.keys() | dense_rank.keys()
        scores = {}
        for i in candidates:
            if strategy == "hybrid":
                scores[i] = sum(weight / (60 + ranks[i]) for ranks, weight in
                                ((bm25_rank, self._bm25_weight), (dense_rank, self._dense_weight)) if i in ranks)
                if self._chapters[i] in rewritten.chapters:
                    scores[i] *= 1.1
            else:
                scores[i] = (bm25 if strategy == "bm25" else dense)[i]
        ranked = list(self._rank(scores, limit))
        neural = {}
        if self._reranker is not None and strategy == "hybrid" and ranked:
            pairs = [(rewritten.expanded, self._passages[i]) for i in ranked]
            try:
                predictions = [float(value) for value in self._reranker.predict(pairs, show_progress_bar=False)]
                if len(predictions) == len(ranked) and all(math.isfinite(score) for score in predictions):
                    neural = dict(zip(ranked, predictions))
                    ranked.sort(key=lambda i: (-neural[i], -scores[i], self.documents[i]["id"]))
            except (RuntimeError, ValueError):
                _LOG.warning("CrossEncoder failed, using RRF scores", exc_info=True)

        results = []
        for i in ranked[:k]:
            channels = [name for name, ranks in (("bm25", bm25_rank), ("dense", dense_rank)) if i in ranks]
            results.append({
                "document": self.documents[i],
                "score": round(neural.get(i, scores[i]), 6),
                "channels": channels,
                "bm25_rank": bm25_rank.get(i),
                "dense_rank": dense_rank.get(i),
                "bm25_score": round(bm25[i], 6) if i in bm25_rank else None,
                "dense_score": round(dense[i], 6) if i in dense_rank else None,
                "fused_score": round(scores[i], 6) if strategy == "hybrid" else None,
                "rerank_score": round(neural[i], 6) if i in neural else None,
            })
        return {"query_info": query_info, "results": results}
