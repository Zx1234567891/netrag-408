import unittest
from unittest.mock import Mock

from netrag.retrieval import HybridRetriever, rewrite_query


DOCUMENTS = [
    {
        "id": "2024_37", "kind": "exam", "year": 2024, "number": 37, "chapter": "第4章 网络层",
        "question": "192.0.2.130/26 所属的子网地址是什么？", "tags": ["子网划分与CIDR"],
        "analysis": "子网掩码与主机地址逐位与运算，得到网络地址 192.0.2.128。",
    },
    {
        "id": "2025_38", "kind": "exam", "year": 2025, "number": 38, "chapter": "传输层",
        "question": "拥塞窗口达到慢开始门限后如何增长？", "tags": ["TCP拥塞控制"],
        "analysis": "慢开始后进入拥塞避免阶段，cwnd 每个往返时间增加约一个 MSS。",
    },
    {
        "id": "dns", "kind": "concept", "year": None, "chapter": "应用层",
        "question": "DNS 客户端与递归解析器如何解析域名？", "tags": ["DNS域名解析"],
        "analysis": "解析器可经根服务器、顶级域和权威服务器获得地址。",
    },
    {
        "id": "tcp-flow", "kind": "concept", "year": None, "chapter": "传输层",
        "question": "接收方缓冲区容量如何影响 TCP 接收窗口？", "tags": ["TCP流量控制"],
        "analysis": "接收方通过窗口字段通告可接收的字节数。",
    },
]


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.retriever = HybridRetriever(DOCUMENTS)

    def test_rewrite_normalizes_fullwidth_and_expands_vague_terms(self):
        rewritten = rewrite_query("ＴＣＰ  网络太堵，窗口怎么变？")
        self.assertEqual(rewritten["normalized"], "tcp网络太堵,窗口怎么变?")
        self.assertIn("TCP拥塞控制", rewritten["aliases"])
        self.assertEqual(rewritten["chapters"], [5])

    def test_alias_can_retrieve_a_canonically_labeled_passage(self):
        result = self.retriever.search("网络太堵时窗口怎么变？")
        best = result["results"][0]
        self.assertEqual(best["document"]["id"], "2025_38")
        self.assertEqual(best["channels"], ["bm25", "dense"])
        self.assertEqual(best["bm25_rank"], 1)
        self.assertEqual(best["dense_rank"], 1)
        self.assertGreater(best["fused_score"], 2 / 61)

    def test_sliding_window_rewrite_respects_protocol_layer(self):
        link = rewrite_query("滑动窗口收到乱序帧后该怎么做？")
        self.assertEqual(link["chapters"], [3])
        self.assertIn("数据链路层滑动窗口", link["aliases"])
        transport = rewrite_query("TCP 的滑动窗口怎么改变？")
        self.assertEqual(transport["chapters"], [5])
        self.assertIn("TCP流量控制", transport["aliases"])
        self.assertEqual(rewrite_query("滑动窗口")["chapters"], [])

    def test_exact_exam_reference_respects_year_and_missing_id(self):
        for query in ("２０２４ 年 ４０８ 真题 第 ３７ 题", "2024_37 怎么做？"):
            with self.subTest(query=query):
                results = self.retriever.search(query)["results"]
                self.assertEqual([item["document"]["id"] for item in results], ["2024_37"])
                self.assertEqual(results[0]["channels"], ["exact"])
        self.assertFalse(self.retriever.search("2024_37", year=2025)["results"])
        self.assertFalse(self.retriever.search("2026 年第 37 题")["results"])

    def test_single_channel_ablation_and_year_filter(self):
        for strategy in ("bm25", "dense"):
            with self.subTest(strategy=strategy):
                result = self.retriever.search("192.0.2.130/26 的子网掩码", year=2024, strategy=strategy)
                self.assertEqual([item["document"]["id"] for item in result["results"]], ["2024_37"])
                self.assertEqual(result["results"][0]["channels"], [strategy])

    def test_repeated_keywords_in_a_wrong_option_do_not_outrank_relevant_question(self):
        documents = [
            {
                "id": "unrelated",
                "question": "物理层传输时钟怎样同步？",
                "answer": "B",
                "options": {"A": "IPv6 邻居发现 " * 16, "B": "根据码元恢复时钟"},
            },
            {
                "id": "relevant",
                "question": "IPv6 主机如何做邻居发现？",
                "analysis": "发送邻居请求，获得邻居通告。",
            },
        ]
        retriever = HybridRetriever(documents)
        for strategy in ("bm25", "hybrid"):
            with self.subTest(strategy=strategy):
                top = retriever.search("IPv6 邻居发现", strategy=strategy)["results"][0]
                self.assertEqual(top["document"]["id"], "relevant")

    def test_unrelated_questions_return_no_context(self):
        for query in ("", "下面如何炒番茄鸡蛋？", "冬天适合种植哪些花卉？"):
            with self.subTest(query=query):
                self.assertEqual(self.retriever.search(query)["results"], [])

    def test_reranker_reorders_and_falls_back_to_rrf_on_runtime_error(self):
        query = "拥塞窗口与接收窗口"
        fused = self.retriever.search(query)["results"]
        self.assertEqual(len(fused), 2)
        self.retriever._reranker = Mock()
        self.retriever._reranker.predict.return_value = [0.1, 0.8]
        reranked = self.retriever.search(query)["results"]
        self.assertEqual(reranked[0]["document"], fused[1]["document"])
        self.assertEqual(reranked[0]["rerank_score"], 0.8)
        self.retriever._reranker.predict.side_effect = RuntimeError("model unavailable")
        with self.assertLogs("netrag.retrieval", level="WARNING"):
            self.assertEqual(self.retriever.search(query)["results"], fused)
