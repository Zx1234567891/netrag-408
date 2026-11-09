# 生成评测与实测结果

2026 年 11 月 10 日在 RTX 4060 Ti 上以 llama.cpp `b8469` 的 Vulkan 后端运行 Qwen3-4B-Instruct-2507 Q4_K_M。模型文件 SHA-256 为 `8cdb57cbb880d313736a9bc4e3d3d2485f145b5e19cf33783746e753e82641fc`。服务使用单推理槽、8192 上下文和种子 408。应用请求使用 JSON 对象模式、温度 0.1、最多 950 输出 token 和 35 秒请求超时。检索使用 BGE-small-zh-v1.5、FAISS、jieba 和 BGE-reranker-base，单路候选 20，OpenMP 线程数为 4。[模型卡](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) · [实测 GGUF](https://huggingface.co/lmstudio-community/Qwen3-4B-Instruct-2507-GGUF/blob/4edb920b6f14e3b9284d4502a6485103d72cde05/Qwen3-4B-Instruct-2507-Q4_K_M.gguf) · [推理服务版本](https://github.com/ggml-org/llama.cpp/releases/tag/b8469)

本次生成测试在最后一轮物理层同义词补齐前完成。保留原运行记录以便复算，最终检索配置的成绩单独列出。[检索评测](retrieval-holdout.md)

| 样本与口径 | 答案准确率 | 幻觉率 |
| --- | ---: | ---: |
| 公开用例 · 全链路 | 18/28（64.29%） | 8/28（28.57%） |
| 公开用例 · 全链路可回答题 | 14/24（58.33%） | 8/24（33.33%） |
| 公开用例 · 送达的模型正文 | 13/21（61.90%） | 8/21（38.10%） |
| 公开用例 · 送达的模型正文与模型主动拒答 | 15/23（65.22%） | 8/23（34.78%） |
| 本地真题 · 全链路 | 4/6（66.67%） | 2/6（33.33%） |
| 本地真题 · 送达的模型正文 | 2/4（50.00%） | 2/4（50.00%） |

公开用例有 26 次真实模型调用，得到 21 条通过验证的正文和 2 条通过验证的主动拒答，另有 3 条被系统回退替代。4 条资料不足对照均正确拒答，其中 2 条由系统在无资料时直接拒答。本地真题有 6 次调用，4 条模型正文通过验证，2 条被替换。Codex 逐题对照预设答案要点和返回证据，另核对了 5 条被替换的模型原稿。被替换的原稿含实质错误、无法由证据支持的主张或无效引用 ID，另列在汇总文件的原稿核对项。 [公开输出](../eval/results/generation-demo-run.json) · [公开逐题判定](../eval/results/generation-demo-judgments.jsonl) · [公开汇总](../eval/results/generation-demo-summary.json) · [真题逐题判定](../eval/results/generation-exam-judgments.jsonl) · [真题汇总](../eval/results/generation-exam-summary.json)

公开组检索至回答的 P95 为 12.37 秒，24/28 在 5 秒内返回。真题组 P95 为 8.05 秒，4/6 在 5 秒内返回。错误主要发生在数值或中间步骤自相矛盾、协议术语混淆及未经证据支持的附加说明。准确率和幻觉率均未达到报告的 88% 和 5% 验收要求。结论仅适用于这一批题和本次配置。

## 口径和复现

公开用例包含 24 条可回答问题和 4 条资料不足对照，覆盖概念、网络情境及需要从题设推导的计算。题面与 60 张原创知识卡分开编写。每条用例列出必须回答的事实和可核对的证据 ID。另有 6 条按题号定位的本地真题复核问题，使用自行编写的提问。公开仓库中的真题评测用例只含题号、评审要点和自编提问。 [公开用例](../eval/generation.jsonl) · [真题用例](../eval/generation-exam.jsonl)

评测沿用产品的检索、提示词、引用验证和回退路径。运行程序记录送达用户的答案、检索证据 ID、模型调用、应答形式和延迟。模型请求有返回正文而系统最终回退时单独计数。模型按结构请求拒答并通过应用验证时，拒答单独记录。

- **准确率**按题计。送达的回答应覆盖问题要求的关键结论，没有实质错误或证据无法支持的实质主张，计算结果允许等价表达和合理舍入。对照题的准确回答应说明资料不足，不猜测答案。
- **幻觉率**按题计。如果一条答案含至少一项实质性错误或题设与证据无法支持的事实主张，记为一次幻觉。对证据或选项的虚构引用也计入幻觉。只答了一部分、格式失败、请求超时或对可回答问题拒答均可降低准确率，这些情况不自动计为幻觉。
- **范围**分开报告整个系统、可回答题、资料不足题、用户看到的模型正文，以及模型正文加上模型主动拒答。模型正文子集受到引用验证和回退机制影响，应结合模型调用数和被替换数阅读。
- **评审**逐题比较实测输出、预先定义的评审要点和实际返回的证据。评审 JSONL 含 `id`、`answer_sha256`、`correct`、`hallucinated`、`reason`。评分程序要求每题都有记录，并检查答案摘要，以便对冻结结果重新计算。

在仓库根目录设置兼容 Chat Completions 的 `NETRAG_LLM_BASE_URL` 与 `NETRAG_LLM_MODEL`，需要鉴权时设置 `NETRAG_LLM_API_KEY`。本次使用 `NETRAG_LLM_JSON_MODE=1` 与 `NETRAG_LLM_TIMEOUT=35`。公开组运行前清除 `NETRAG_DATA_DIR`，本地真题使用本地模型服务。下列命令重新保存输出，并能用已发布的冻结记录重算公开指标。`--capture-drafts` 会在忽略提交的 `eval_local` 下保留模型原始稿，以便分析未通过验证的响应。

```powershell
llama-server --model Qwen3-4B-Instruct-2507-Q4_K_M.gguf --alias qwen3-4b-instruct-2507-q4km --host 127.0.0.1 --port 8090 --gpu-layers 99 --ctx-size 8192 --parallel 1 --seed 408
```

```powershell
$env:NETRAG_LLM_BASE_URL = 'http://127.0.0.1:8090/v1'
$env:NETRAG_LLM_MODEL = 'qwen3-4b-instruct-2507-q4km'
$env:NETRAG_LLM_JSON_MODE = '1'
$env:NETRAG_LLM_TIMEOUT = '35'
python -m netrag.generation_benchmark run eval/generation.jsonl --output eval_local/generation-demo-run.json --capture-drafts
python -m netrag.generation_benchmark run eval/generation-exam.jsonl --data-dir local_data --output eval_local/generation-exam-run.json --capture-drafts
python -m netrag.generation_benchmark score eval/results/generation-demo-run.json eval/results/generation-demo-judgments.jsonl
```
