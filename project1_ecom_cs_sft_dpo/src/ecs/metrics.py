"""Automatic metrics for customer-service answers.

Definitions (keep them in the doc so that numbers are explainable in interview):
  * 准确率 accuracy      : 关键点覆盖率 >= 2/3 且未命中禁止表述且无复读；需转人工的样本则要求正确转人工
  * 自动解决率 resolve    : 非转人工样本中，模型“准确回答且没有把用户推给人工”的比例（分母为全部样本）
  * 满意度 satisfaction   : 1-5 分，默认规则代理分；可选 LLM-as-judge（--judge_api）
  * 复读率 repetition     : 6-gram 重复比例 > 0.3 或同一句子重复 >= 3 次
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from .sop import TRANSFER_HINTS

EMPATHY = ["抱歉", "理解", "不好意思", "久等", "麻烦"]
ACTION = ["【", "申请售后", "您可以", "请在", "请您", "点击", "提交", "联系"]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def keypoint_coverage(answer: str, keypoints: List[List[str]]) -> float:
    if not keypoints:
        return 1.0
    a = _norm(answer)
    hit = sum(1 for group in keypoints if any(_norm(k) in a for k in group))
    return hit / len(keypoints)


NEGATIONS = ("不", "无法", "没", "非", "未", "暂不", "不能")


def forbidden_hit(answer: str, forbidden: List[str]) -> bool:
    """Negation-aware: “不支持无理由退货” 不算命中禁止表述 “支持无理由退货”。"""
    a = _norm(answer)
    for f in forbidden or []:
        f = _norm(f)
        start = a.find(f)
        while start != -1:
            prefix = a[max(0, start - 2):start]
            if not any(prefix.endswith(n) for n in NEGATIONS):
                return True
            start = a.find(f, start + 1)
    return False


def mentions_transfer(answer: str) -> bool:
    return any(h in answer for h in TRANSFER_HINTS)


def repetition_ratio(answer: str, n: int = 6) -> float:
    a = _norm(answer)
    if len(a) < n * 2:
        return 0.0
    grams = [a[i:i + n] for i in range(len(a) - n + 1)]
    c = Counter(grams)
    repeated = sum(v for v in c.values() if v > 1)
    return repeated / len(grams)


def is_repetitive(answer: str) -> bool:
    if repetition_ratio(answer) > 0.3:
        return True
    sents = [s for s in re.split(r"[。！？!?\n]", answer) if len(s.strip()) > 3]
    return any(v >= 3 for v in Counter(sents).values())


def satisfaction_proxy(answer: str, item: Dict, cov: float, rep: bool, forb: bool) -> float:
    score = 3.0
    if item.get("domain") in ("投诉", "退换货") and any(e in answer for e in EMPATHY):
        score += 0.5
    if any(a in answer for a in ACTION):
        score += 0.5
    score += 1.0 * cov - 0.5
    if rep:
        score -= 1.5
    if forb:
        score -= 1.0
    if len(answer) > 220:
        score -= 0.5
    if len(answer) < 20:
        score -= 1.0
    if mentions_transfer(answer) and not item.get("transfer"):
        score -= 1.0
    return max(1.0, min(5.0, score))


def evaluate_item(pred: str, item: Dict) -> Dict:
    cov = keypoint_coverage(pred, item.get("keypoints", []))
    forb = forbidden_hit(pred, item.get("forbidden", []))
    rep = is_repetitive(pred)
    trans = mentions_transfer(pred)
    need = max(1, math.ceil(len(item.get("keypoints", [])) * 2 / 3))
    hits = round(cov * len(item.get("keypoints", [])))
    if item.get("transfer"):
        accurate = trans and not rep
        resolved = False
    else:
        accurate = (hits >= need) and not forb and not rep
        resolved = accurate and not trans
    return {
        "coverage": round(cov, 4),
        "forbidden": forb,
        "repetitive": rep,
        "transfer": trans,
        "accurate": accurate,
        "resolved": resolved,
        "satisfaction": satisfaction_proxy(pred, item, cov, rep, forb),
        "length": len(pred),
    }


def aggregate(results: List[Dict], items: List[Dict]) -> Dict:
    def _agg(rs):
        n = max(1, len(rs))
        return {
            "n": len(rs),
            "accuracy": round(sum(r["accurate"] for r in rs) / n, 4),
            "auto_resolve_rate": round(sum(r["resolved"] for r in rs) / n, 4),
            "satisfaction": round(sum(r["satisfaction"] for r in rs) / n, 3),
            "repetition_rate": round(sum(r["repetitive"] for r in rs) / n, 4),
            "forbidden_rate": round(sum(r["forbidden"] for r in rs) / n, 4),
            "transfer_rate": round(sum(r["transfer"] for r in rs) / n, 4),
            "avg_len": round(sum(r["length"] for r in rs) / n, 1),
        }

    by_domain = defaultdict(list)
    by_subset = defaultdict(list)
    for r, it in zip(results, items):
        by_domain[it.get("domain", "unk")].append(r)
        by_subset[it.get("subset", "normal")].append(r)
    return {
        "overall": _agg(results),
        "by_domain": {k: _agg(v) for k, v in sorted(by_domain.items())},
        "by_subset": {k: _agg(v) for k, v in sorted(by_subset.items())},
    }


JUDGE_PROMPT = """你是电商客服质检员。请根据【参考答案要点】对【客服回复】打分（1-5分）：
5=政策准确、给出明确操作路径、语气得体；4=基本准确有小瑕疵；3=部分正确或含糊；2=明显错误或答非所问；1=错误政策/复读/敷衍。
只输出一个数字。
【用户问题】{q}
【参考答案要点】{ref}
【客服回复】{pred}"""


def llm_judge(client, model: str, q: str, ref: str, pred: str) -> Optional[float]:
    """Optional LLM-as-judge with an OpenAI-compatible client."""
    try:
        r = client.chat.completions.create(
            model=model, temperature=0, max_tokens=4,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(q=q, ref=ref, pred=pred)}],
        )
        m = re.search(r"[1-5]", r.choices[0].message.content or "")
        return float(m.group()) if m else None
    except Exception as e:  # pragma: no cover
        print("[judge] error:", e)
        return None
