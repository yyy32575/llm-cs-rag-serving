"""Query rewriting: 同义词/口语 -> 研报术语、时间表达归一化；可选 LLM 改写。

空召回 / 低召回的主要原因是“用户说法 != 文档说法”：
  用户：宁川动力装机量 / 25年Q3 / NIM / 稼动率
  文档：动力电池装车量 / 2025年第三季度 / 净息差 / 产能利用率
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

SYNONYMS: Dict[str, List[str]] = {
    "装机量": ["装车量"], "发货量": ["出货量"], "稼动率": ["产能利用率"], "NIM": ["净息差"], "新接订单": ["新签订单"],
    "直营渠道占比": ["直销收入占比"], "直营占比": ["直销收入占比"], "液奶收入": ["液态奶收入"],
    "新药收入占比": ["创新药收入占比"], "电动车销量": ["新能源汽车销量"], "营收": ["营业收入"],
    "净利润": ["归母净利润"], "每股收益": ["EPS"], "市盈率": ["PE"], "市占率": ["市场份额"], "研发费用": ["研发投入"],
}
TIME_PATTERNS = [
    (r"(\d{4})\s*Q([1-4])", lambda m: f"{m.group(1)}年第{'一二三四'[int(m.group(2)) - 1]}季度"),
    (r"(?<!\d)(\d{2})年", lambda m: f"20{m.group(1)}年"),
    (r"Q1|(?<![前第])一季度", lambda m: "第一季度"), (r"Q2|(?<![前第])二季度", lambda m: "第二季度"),
    (r"Q3|(?<![前第])三季度", lambda m: "第三季度"), (r"Q4|(?<![前第])四季度", lambda m: "第四季度"),
]


def normalize_time(q: str) -> str:
    for p, f in TIME_PATTERNS:
        q = re.sub(p, f, q)
    return q.replace("年年", "年")


def rewrite(q: str, use_synonyms: bool = True, use_time: bool = True) -> str:
    """Return the rewritten query used by both BM25 and dense retrieval."""
    out = normalize_time(q) if use_time else q
    if use_synonyms:
        extra = []
        for k, vs in SYNONYMS.items():
            if k.lower() in out.lower():
                extra += [v for v in vs if v not in out]
        if extra:
            out = out + "（" + " ".join(dict.fromkeys(extra)) + "）"
    return out


LLM_REWRITE_PROMPT = """你是金融检索助手。把用户问题改写成适合在券商研报中检索的规范表述：
补全公司全称、把口语指标换成研报术语、把时间写成“2025年第三季度”格式。只输出改写后的问题。
用户问题：{q}"""


def llm_rewrite(q: str, client, model: str) -> Optional[str]:
    try:
        r = client.chat.completions.create(model=model, temperature=0, max_tokens=64,
                                           messages=[{"role": "user", "content": LLM_REWRITE_PROMPT.format(q=q)}])
        return (r.choices[0].message.content or "").strip() or None
    except Exception:
        return None
