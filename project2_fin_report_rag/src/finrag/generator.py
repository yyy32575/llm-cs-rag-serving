"""Answer generation with evidence binding, low-confidence refusal and post-hoc citation verification.

幻觉缓解三件套：
  1) 证据绑定：prompt 中证据编号 [1]..[n]，要求每个数字都标注来源编号，只允许依据证据作答
  2) 低置信度拒答：rerank 最高分 < 阈值 -> 不调用 LLM，直接返回“资料中未找到”
  3) 生成后引用校验：回答里的每个数字必须能在其引用的证据中找到，否则降级为“部分可信”并提示
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

from .chunker import Chunk
from .utils import norm_text

SYSTEM = ("你是严谨的金融研究助理。只能依据给定的【证据】回答问题，不得使用证据之外的知识。"
          "回答中出现的每个数字后面都要用 [编号] 标注来源证据；如果证据不足以回答，"
          "请直接回答“资料中未找到相关信息，无法回答”。回答简洁，不超过120字。")
TEMPLATE = "【证据】\n{ctx}\n\n【问题】{q}\n【回答】"
# 拒答决策前置：检索侧（rerank 阈值 + 约束校验）已判定“证据可答”时使用，不再给模型拒答出口，
# 解决小模型“证据明明在却说无法回答”的过度拒答问题
SYSTEM_ANSWER = ("你是严谨的金融研究助理。下面的【证据】中已经包含问题的答案，请从证据中找到对应的数字直接回答，"
                 "不得使用证据之外的知识，注意区分年份和季度。回答中的每个数字后面用 [编号] 标注来源证据。回答简洁，不超过120字。")
REFUSAL = "资料中未找到相关信息，无法回答。"
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
QUARTER_RE = re.compile(r"(第[一二三四]季度|Q[1-4]|[一二三四]季度)")
ENTITY_RE = re.compile(r"^([\u4e00-\u9fa5]{2,6}?)(?=\d|的|20|今年|去年|[，,]|三季度|上半年)")
_QMAP = {"Q1": "第一季度", "Q2": "第二季度", "Q3": "第三季度", "Q4": "第四季度", "一季度": "第一季度", "二季度": "第二季度",
         "三季度": "第三季度", "四季度": "第四季度"}


def check_constraints(query: str, chunks: List[Chunk]) -> List[str]:
    """查询硬约束校验：问题里的年份 / 季度 / 主体，必须至少在一条证据（正文或文档名）里出现，否则判为“证据不足”。
    这是比 rerank 分数更可靠的拒答信号：“2028年营收预测”这类问题，Top1 证据和问题主题高度相关（分数高），
    但证据里根本没有 2028。"""
    hay = norm_text("".join(c.doc_id + c.text for c in chunks))
    missing = []
    for y in set(re.findall(r"(20\d{2})(?:年|E|Q)", query)):
        if y not in hay:
            missing.append(y)
    for q in set(QUARTER_RE.findall(query)):
        qn = _QMAP.get(q, q)
        if qn not in hay and q not in hay and qn.replace("第", "") not in hay:
            missing.append(qn)
    m = ENTITY_RE.match(query)
    if m and m.group(1) not in hay and not any(g in m.group(1) for g in GENERIC_PREFIX):
        missing.append(m.group(1))
    metric = extract_metric(query)
    if metric and not metric_covered(metric, hay):
        missing.append("指标:" + metric)
    return missing


GENERIC_PREFIX = ("研报", "根据", "请问", "预测", "报告")
METRIC_RE = re.compile(r"([\u4e00-\u9fa5A-Za-z]{2,16})(?:分别)?(?:是多少|是谁|有多少|多少)")
METRIC_DELIM = re.compile(r"[的年度末中，,、在]|\d")
METRIC_STOP = {"可以", "获得", "公司", "多少", "分别", "预测值", "单季", "累计", "研报", "根据"}


def extract_metric(query: str) -> str:
    """“江海银行2025年的分红率是多少” -> “分红率”；“宁川动力的董事长是谁” -> “董事长”。"""
    m = METRIC_RE.search(query)
    if not m:
        return ""
    head = query[:m.end(1)]
    parts = METRIC_DELIM.split(head)
    return parts[-1] if parts else ""


def metric_covered(metric: str, hay: str) -> bool:
    from .query_rewrite import SYNONYMS

    for k, vs in SYNONYMS.items():
        if k.lower() in metric.lower() and any(norm_text(v) in hay for v in vs):
            return True
    import jieba

    toks = [t for t in jieba.lcut(metric) if len(t) >= 2 and t not in METRIC_STOP]
    if not toks:
        return True
    return sum(norm_text(t) in hay for t in toks) / len(toks) >= 0.5


def build_context(chunks: List[Chunk], max_chars: int = 3000) -> str:
    parts, used = [], 0
    for i, c in enumerate(chunks, 1):
        t = f"[{i}]《{c.doc_id}》第{c.page}页：{c.text}"
        if used + len(t) > max_chars and parts:
            break
        parts.append(t)
        used += len(t)
    return "\n".join(parts)


def verify_citations(answer: str, chunks: List[Chunk]) -> Dict:
    """Check that every number in a sentence appears in at least one of the evidences cited by that sentence."""
    cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer) if 0 < int(x) <= len(chunks)})
    sents = [s for s in re.split(r"(?<=[。；！？\n])", answer) if s.strip()]
    bad = []
    for s in sents:
        refs = [int(x) for x in re.findall(r"\[(\d+)\]", s) if 0 < int(x) <= len(chunks)] or cited
        body = re.sub(r"\[\d+\]", "", s)
        for num in NUM_RE.findall(body):
            if len(num) < 2 and "." not in num:
                continue
            if not any(num in norm_text(chunks[r - 1].text) for r in refs):
                bad.append(num)
    return {"cited": cited, "unsupported_numbers": bad, "verified": not bad}


class Generator:
    def __init__(self, model: str = None, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 256):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url or os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
                             api_key=api_key or os.environ.get("LLM_API_KEY", "EMPTY"))
        self.model = model or os.environ.get("LLM_MODEL", "Qwen/Qwen3-8B")
        self.temperature, self.max_tokens = temperature, max_tokens

    llm_refusal = True  # False = 拒答决策前置

    def __call__(self, q: str, chunks: List[Chunk]) -> str:
        msgs = [{"role": "system", "content": SYSTEM if self.llm_refusal else SYSTEM_ANSWER},
                {"role": "user", "content": TEMPLATE.format(ctx=build_context(chunks), q=q)}]
        r = self.client.chat.completions.create(model=self.model, messages=msgs, temperature=self.temperature,
                                                max_tokens=self.max_tokens,
                                                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S).strip()


class HFGenerator(Generator):
    """Local transformers generation (CPU/GPU) — used for the reproducible small-model demo."""

    def __init__(self, model_path: str, max_tokens: int = 160, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32 if device == "cpu"
                                                          else torch.bfloat16).to(device or "cpu").eval()
        self.max_tokens = max_tokens

    def __call__(self, q: str, chunks: List[Chunk]) -> str:
        msgs = [{"role": "system", "content": SYSTEM if self.llm_refusal else SYSTEM_ANSWER},
                {"role": "user", "content": TEMPLATE.format(ctx=build_context(chunks, 1800), q=q)}]
        try:
            p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(p, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_tokens, do_sample=False, repetition_penalty=1.05)
        txt = self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
