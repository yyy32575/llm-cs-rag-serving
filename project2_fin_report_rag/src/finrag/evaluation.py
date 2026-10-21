"""Evaluation for retrieval and end-to-end QA.

Retrieval (金标 = 文档 + 证据片段):
  一条证据被“命中”= top-k 中存在 chunk 满足  chunk.doc_id == 金标 doc_id  且  证据字符串完整出现在 chunk 中。
  Recall@k = 平均证据命中率（对比题需要 2 条证据都命中才算满分）；MRR 取第一条证据的倒数排名。
  “无答案”题不参与检索指标，只参与拒答指标。
End-to-end:
  answer_acc   = 所有证据数字都出现在回答里（数值型问题）
  citation_acc = 回答引用的 [n] 证据里确实包含答案数字
  refusal      = 无答案题正确拒答率 / 有答案题误拒率
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

from .chunker import Chunk
from .utils import norm_text

REFUSAL_MARKERS = ["无法回答", "未找到", "没有找到", "无法确定", "资料中未", "文档中未", "未提及", "无相关"]


def evidence_hit(ev: Dict, chunk: Chunk) -> bool:
    docs = ev.get("doc_ids") or [ev.get("doc_id")]
    return chunk.doc_id in docs and norm_text(ev["must"]) in norm_text(chunk.text)


def retrieval_metrics(qa: List[Dict], ranked: List[List[int]], chunks: List[Chunk], ks: Sequence[int] = (1, 3, 5, 10)):
    res = {f"recall@{k}": [] for k in ks}
    res["mrr"] = []
    by_type = defaultdict(lambda: defaultdict(list))
    for q, idxs in zip(qa, ranked):
        if not q["evidence"]:
            continue
        for k in ks:
            top = [chunks[i] for i in idxs[:k]]
            r = sum(any(evidence_hit(ev, c) for c in top) for ev in q["evidence"]) / len(q["evidence"])
            res[f"recall@{k}"].append(r)
            by_type[q["type"]][f"recall@{k}"].append(r)
        rr = 0.0
        for rank, i in enumerate(idxs, 1):
            if evidence_hit(q["evidence"][0], chunks[i]):
                rr = 1.0 / rank
                break
        res["mrr"].append(rr)
        by_type[q["type"]]["mrr"].append(rr)

    def avg(xs):
        return round(sum(xs) / max(1, len(xs)), 4)

    out = {k: avg(v) for k, v in res.items()}
    out["n"] = len(res["mrr"])
    out["by_type"] = {t: {k: avg(v) for k, v in d.items()} for t, d in by_type.items()}
    return out


def is_refusal(ans: str) -> bool:
    return any(m in ans for m in REFUSAL_MARKERS)


def answer_metrics(qa: List[Dict], answers: List[Dict], chunks_by_id: Dict[str, Chunk]):
    acc, cite, refuse_ok, false_refuse, n_ans, n_unans, n_cite = 0, 0, 0, 0, 0, 0, 0
    by_type = defaultdict(list)
    for q, a in zip(qa, answers):
        text = a["answer"]
        if not q["evidence"]:
            n_unans += 1
            ok = is_refusal(text)
            refuse_ok += ok
            by_type[q["type"]].append(ok)
            continue
        n_ans += 1
        if is_refusal(text):
            false_refuse += 1
            by_type[q["type"]].append(False)
            continue
        ok = all(norm_text(ev["must"]) in norm_text(text) for ev in q["evidence"])
        acc += ok
        by_type[q["type"]].append(ok)
        cited = [chunks_by_id[c] for c in a.get("cited_chunk_ids", []) if c in chunks_by_id]
        if cited:
            n_cite += 1
            cite += all(any(evidence_hit(ev, c) for c in cited) for ev in q["evidence"])
    return {
        "answer_acc": round(acc / max(1, n_ans), 4),
        "citation_acc": round(cite / max(1, n_cite), 4),
        "refusal_recall": round(refuse_ok / max(1, n_unans), 4),
        "false_refusal_rate": round(false_refuse / max(1, n_ans), 4),
        "by_type": {t: round(sum(v) / len(v), 4) for t, v in by_type.items()},
    }
