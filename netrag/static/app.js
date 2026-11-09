"use strict";

const element = (id) => document.getElementById(id);
const askForm = element("ask-form");
const questionInput = element("question");
const submitButton = element("submit-button");
const answerSection = element("answer-section");
const copyButton = element("copy-answer");
const yearFilter = element("year-filter");
const sourceIndex = new Map();
let lastAnswer = "";
let corpusMode = "";

const answerModes = {
  llm: "大模型解析",
  extractive: "知识库解析",
  calculation: "公式推导",
  insufficient: "检索证据不足",
};

const channelNames = {vector: "向量检索", dense: "向量检索", bm25: "关键词检索", exact: "题号精确匹配"};
const figurePattern = /!\[([^\]\n]*)\]\(([^)\s]+)\)/g;

function node(tag, value, className) {
  const child = document.createElement(tag);
  if (value !== undefined && value !== null) child.textContent = String(value);
  if (className) child.className = className;
  return child;
}

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

async function readJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data?.detail;
    if (typeof detail === "string") throw new Error(detail);
    throw new Error(response.status === 422 ? "请检查问题内容和检索设置。" : "请求失败，请稍后重试。");
  }
  if (!data || typeof data !== "object") throw new Error("服务器返回了无法读取的结果。");
  return data;
}

function readCoverage(data) {
  const years = Array.isArray(data.years) ? data.years.filter(Number.isInteger).sort((a, b) => a - b) : [];
  const examCount = data.exam_count ?? 0;
  corpusMode = data.mode;
  element("stat-exams").textContent = examCount;
  element("stat-years").textContent = years.length;
  element("stat-period").textContent = years.length > 1
    ? `${years[0]}—${years.at(-1)} 年真题`
    : years.length === 1 ? `${years[0]} 年真题` : "当前未导入真题";
  element("scope-badge").dataset.mode = corpusMode;
  if (corpusMode === "demo") {
    element("scope-label").textContent = "演示模式 · 原创示例";
    element("frequency-period").textContent = "导入计网真题后可查看频次";
  }
  if (data.dense_backend === "lexical-concept-cosine") {
    element("dense-backend").textContent = "考点词向量 + BM25 关键词检索";
  }
  if (data.concept_count > 0) {
    const supplement = element("stat-supplement");
    const label = corpusMode === "demo" ? "原创示例与考点单元" : "补充考点单元";
    supplement.querySelector("span").textContent = `${data.concept_count} 个${label}`;
    supplement.hidden = false;
  }
  for (const year of years.slice().reverse()) {
    const option = node("option", `${year} 年`);
    option.value = year;
    yearFilter.append(option);
  }
}

function showMessage(id, value) {
  const box = element(id);
  box.textContent = value;
  box.hidden = !value;
}

function addSourceText(container, text) {
  let cursor = 0;
  for (const match of text.matchAll(figurePattern)) {
    let url;
    try {
      url = new URL(match[2]);
    } catch {
      continue;
    }
    if (url.protocol !== "https:" || url.username || url.password) continue;
    container.append(text.slice(cursor, match.index));
    const link = node("a", match[1] ? `题图 · ${match[1]}` : "题图", "figure-link");
    link.href = url.href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.title = "在新标签页打开题图";
    container.append(link);
    cursor = match.index + match[0].length;
  }
  container.append(text.slice(cursor));
}

function renderSource(source, index) {
  const details = node("details", null, "evidence-item");
  details.id = `source-${index + 1}`;
  sourceIndex.set(String(source.id), details);
  const summary = node("summary");
  const header = node("div");
  header.append(node("p", source.id, "source-id"));
  const firstLine = String(source.question ?? "").split(/\r?\n/, 1)[0].replace(figurePattern, "［题图］");
  header.append(node("p", firstLine.length > 110 ? `${firstLine.slice(0, 110)}…` : firstLine, "source-title"));
  const badges = node("div", null, "source-labels");
  const type = source.kind === "exam" ? "计网真题" : "原创考点";
  badges.append(node("span", type, "meta-badge"));
  if (source.kind === "exam" && source.year != null && source.number != null) {
    badges.append(node("span", `${source.year} 年第 ${source.number} 题`, "meta-badge"));
  }
  for (const channel of new Set((source.channels ?? []).map((item) => channelNames[item] ?? item))) {
    badges.append(node("span", channel, "meta-badge"));
  }
  header.append(badges);
  summary.append(header, icon("chevron"));
  const content = node("div", null, "source-content");
  for (const [label, value] of [["题目", source.question], ["答案", source.answer], ["解析", source.analysis]]) {
    if (!value) continue;
    const part = node("div");
    const paragraph = node("p");
    addSourceText(paragraph, String(value));
    part.append(node("h3", label), paragraph);
    if (label === "题目" && source.options && typeof source.options === "object") {
      const options = node("ul");
      for (const [key, value] of Object.entries(source.options)) {
        const option = node("li");
        addSourceText(option, `${key}. ${value}`);
        options.append(option);
      }
      part.append(options);
    }
    content.append(part);
  }
  if (source.quality_flags?.includes("stem_has_image") && !content.querySelector(".figure-link")) {
    content.append(node("p", "原题含图，当前题库未提供可打开的题图链接。", "empty-note"));
  }
  if (Array.isArray(source.tags) && source.tags.length) {
    const tags = node("div");
    tags.append(node("h3", "关联考点"), node("p", source.tags.join("、")));
    content.append(tags);
  }
  details.append(summary, content);
  return details;
}

function renderEvidence(sources) {
  sourceIndex.clear();
  const list = element("evidence-list");
  list.replaceChildren(...sources.map(renderSource));
  list.hidden = sources.length === 0;
  const placeholder = element("evidence-empty");
  placeholder.textContent = "本次查询没有返回足够相关的知识库片段。";
  placeholder.hidden = sources.length > 0;
  const count = element("evidence-count");
  count.textContent = `${sources.length} 条证据`;
  count.hidden = sources.length === 0;
}

function renderTrace(info) {
  const rows = [
    ["归一化", info.normalized],
    ["扩展检索词", info.expanded === info.normalized ? "" : info.expanded],
    ["同义词", Array.isArray(info.aliases) ? info.aliases.join("、") : ""],
    ["关联章节", Array.isArray(info.chapters) ? info.chapters.map((chapter) => `第 ${chapter} 章`).join("、") : ""],
  ];
  const traceList = element("trace-list");
  traceList.replaceChildren();
  for (const [key, value] of rows) {
    if (value) traceList.append(node("dt", key), node("dd", value));
  }
  element("trace").hidden = traceList.children.length === 0;
}

function renderFrequency(items) {
  const list = element("frequency-content");
  list.replaceChildren();
  if (!items.length) {
    const notice = corpusMode === "demo" ? "当前使用原创示例，没有计网真题频次。" : "本次未找到可统计的关联考点。";
    list.append(node("p", notice, "empty-note"));
    return;
  }
  const {start_year: start, end_year: end} = items[0];
  if (start != null && end != null) element("frequency-period").textContent = `${start}—${end} 年计网真题`;
  const largest = Math.max(1, ...items.map((item) => item.recent ?? 0));
  for (const item of items) {
    const row = node("div", null, "frequency-row");
    const bar = node("meter", null, "frequency-bar");
    bar.setAttribute("aria-hidden", "true");
    bar.min = 0;
    bar.max = largest;
    bar.value = Math.max(0, item.recent ?? 0);
    row.append(node("h3", item.tag), node("p", `近五年 ${item.recent} 次 · 历年 ${item.total} 次`), bar);
    list.append(row);
  }
}

function addInlineText(container, text) {
  const markup = /\*\*([^*\n]+)\*\*|`([^`\n]+)`|\[([^\]\n]+)\]|【([^】\n]+)】/g;
  let cursor = 0;
  for (const match of text.matchAll(markup)) {
    container.append(text.slice(cursor, match.index));
    const source = sourceIndex.get(match[3] ?? match[4]);
    if (match[1]) {
      const bold = node("strong");
      addInlineText(bold, match[1]);
      container.append(bold);
    } else if (match[2]) {
      container.append(node("code", match[2]));
    } else if (source) {
      const link = node("a", match[0]);
      link.href = `#${source.id}`;
      link.addEventListener("click", () => { source.open = true; });
      container.append(link);
    } else {
      container.append(match[0]);
    }
    cursor = match.index + match[0].length;
  }
  container.append(text.slice(cursor));
}

function renderAnswer(text) {
  const content = element("answer-content");
  content.replaceChildren();
  let currentList = null;
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    const heading = /^#{1,6}\s+(.+)$/.exec(line);
    const listItem = /^([-*•]|\d+[.、])\s+(.+)$/.exec(line);
    if (listItem) {
      const tag = /^\d/.test(listItem[1]) ? "OL" : "UL";
      if (currentList?.tagName !== tag) {
        currentList = node(tag.toLowerCase());
        content.append(currentList);
      }
      const item = node("li");
      addInlineText(item, listItem[2]);
      currentList.append(item);
      continue;
    }
    currentList = null;
    if (line) {
      const block = node(heading ? "h3" : "p");
      addInlineText(block, heading?.[1] ?? line);
      content.append(block);
    }
  }
}

function showResult(result, query) {
  lastAnswer = typeof result.answer === "string" ? result.answer : "没有收到解题解析。";
  element("answered-question").textContent = query;
  copyButton.removeAttribute("data-copied");
  copyButton.title = "复制答案";
  copyButton.setAttribute("aria-label", "复制答案");
  const meta = element("answer-meta");
  meta.replaceChildren(node("span", answerModes[result.mode] ?? "解析", "meta-badge"));
  if (Number.isFinite(result.elapsed_ms)) meta.append(node("span", `${(result.elapsed_ms / 1000).toFixed(2)} 秒`, "meta-badge"));
  meta.hidden = false;
  renderEvidence(Array.isArray(result.sources) ? result.sources : []);
  renderTrace(result.query_info ?? {});
  renderFrequency(Array.isArray(result.frequency) ? result.frequency : []);
  renderAnswer(lastAnswer);
  answerSection.hidden = false;
  element("answer-title").focus({preventScroll: true});
  answerSection.scrollIntoView({block: "nearest"});
}

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (submitButton.disabled) return;
  const query = questionInput.value.trim();
  showMessage("request-error", "");
  if (Array.from(query).length < 2) {
    showMessage("request-error", "请输入至少两个字符的问题。");
    questionInput.focus();
    return;
  }
  submitButton.disabled = true;
  showMessage("request-status", "正在查找相关证据并整理解析…");
  try {
    const result = await readJson("/api/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({query, year: yearFilter.value ? Number(yearFilter.value) : null, strategy: element("strategy").value}),
    });
    showResult(result, query);
  } catch (error) {
    showMessage("request-error", error instanceof TypeError ? "无法连接问答服务，请检查服务后重试。" : error.message);
  } finally {
    submitButton.disabled = false;
    showMessage("request-status", "");
  }
});

questionInput.addEventListener("keydown", (event) => {
  if (!event.isComposing && (event.ctrlKey || event.metaKey) && event.key === "Enter" && !submitButton.disabled) {
    event.preventDefault();
    askForm.requestSubmit();
  }
});

for (const button of document.querySelectorAll("[data-example]")) {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.example;
    questionInput.focus();
  });
}

copyButton.addEventListener("click", async () => {
  let feedback = "已复制答案";
  try {
    await navigator.clipboard.writeText(lastAnswer);
    copyButton.dataset.copied = "true";
  } catch {
    feedback = "无法访问剪贴板，请手动选择答案文本";
  }
  copyButton.title = feedback;
  copyButton.setAttribute("aria-label", feedback);
});

function updateNavigation() {
  const hash = location.hash || "#workspace";
  for (const link of document.querySelectorAll(".side-links a")) {
    if (link.hash === hash || (link.hash === "#evidence" && hash.startsWith("#source-"))) {
      link.setAttribute("aria-current", "location");
    } else {
      link.removeAttribute("aria-current");
    }
  }
}

window.addEventListener("hashchange", updateNavigation);
updateNavigation();
readJson("/api/health").then(readCoverage).catch(() => {
  element("stat-period").textContent = "暂时无法读取知识库状态";
});
