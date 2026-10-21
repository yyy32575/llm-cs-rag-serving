"""SFT data cleaning: normalization, PII masking, dedup (exact + MinHash near-dup),
short-answer filtering and policy-contradiction filtering.

Interview talking point: 原始约 12000 条 -> 去重 / 去短回复(<20字) / 去矛盾答案 -> 约 5800 条。
Every rule reports how many samples it removed (see stats) so that the number is explainable.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from .sop import POLICY, USER_PREFIX, USER_SUFFIX

PUNCT = r"[\s,，。.!！?？~、；;：:“”\"'（）()【】\[\]…-]+"
_PREFIX = sorted({p.rstrip("，？,?") for p in USER_PREFIX if p}, key=len, reverse=True)

# Map: policy key -> regexes that capture the number appearing in an answer.
POLICY_PATTERNS: Dict[str, List[str]] = {
    "no_reason_days": [r"(\d+)天无理由", r"签收后(\d+)天内、商品不影响", r"签收后(\d+)天内支持换码"],
    "quality_exchange_days": [r"签收后(\d+)天内提供照片", r"签收后(\d+)天内，在【申请售后】中选择“质量问题”",
                              r"超过(\d+)天的质量问题退换期"],
    "warranty_months": [r"(\d+)个月官方保修", r"(\d+)个月全国联保", r"(\d+)个月保修期内"],
    "price_protect_days": [r"签收后(\d+)天内如果同款降价", r"超过(\d+)天价保期"],
    "ship_hours": [r"(\d+)小时内发货", r"(\d+)小时内发出"],
    "refund_days": [r"(\d+)个工作日内原路退回"],
    "fresh_report_hours": [r"签收后(\d+)小时内"],
    "invoice_days": [r"电子发票会在(\d+)个工作日"],
}

PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ORDER_RE = re.compile(r"(?<!\d)\d{16,20}(?!\d)")


def normalize_query(q: str) -> str:
    q = q.strip()
    changed = True
    while changed:
        changed = False
        for p in _PREFIX:
            if q.startswith(p):
                q = q[len(p):].lstrip("，,？? ")
                changed = True
    for s in sorted({s for s in USER_SUFFIX if s}, key=len, reverse=True):
        if q.endswith(s):
            q = q[: -len(s)]
    return re.sub(PUNCT, "", q).lower()


def mask_pii(text: str) -> str:
    text = PHONE_RE.sub("<PHONE>", text)
    return ORDER_RE.sub("<ORDER_ID>", text)


def policy_values(answer: str) -> Dict[str, set]:
    found = defaultdict(set)
    for key, pats in POLICY_PATTERNS.items():
        for p in pats:
            for m in re.finditer(p, answer):
                found[key].add(int(m.group(1)))
    return found


def policy_violations(answer: str, policy: Dict[str, int] = POLICY) -> List[str]:
    vio = []
    for key, vals in policy_values(answer).items():
        if any(v != policy[key] for v in vals):
            vio.append(key)
    return vio


# ---------------------------- MinHash near-dup ----------------------------
def _shingles(s: str, k: int = 3) -> set:
    return {s[i:i + k] for i in range(max(1, len(s) - k + 1))}


def _minhash(sh: set, num_perm: int = 64) -> Tuple[int, ...]:
    sig = []
    for i in range(num_perm):
        salt = str(i).encode()
        sig.append(min(int.from_bytes(hashlib.md5(salt + x.encode()).digest()[:8], "little") for x in sh))
    return tuple(sig)


def near_dup_groups(keys: List[str], threshold: float = 0.85, num_perm: int = 64, bands: int = 16) -> List[int]:
    """Return a 'representative index' for every key (union-find over LSH candidates)."""
    rows = num_perm // bands
    sigs = [_minhash(_shingles(k), num_perm) for k in keys]
    sh = [_shingles(k) for k in keys]
    parent = list(range(len(keys)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for b in range(bands):
        buckets = defaultdict(list)
        for i, sig in enumerate(sigs):
            buckets[sig[b * rows:(b + 1) * rows]].append(i)
        for idx in buckets.values():
            if len(idx) < 2:
                continue
            base = idx[0]
            for j in idx[1:]:
                inter = len(sh[base] & sh[j])
                union = len(sh[base] | sh[j]) or 1
                if inter / union >= threshold:
                    parent[find(j)] = find(base)
    return [find(i) for i in range(len(keys))]


# ------------------------------- pipeline --------------------------------
def clean(rows: List[Dict], min_answer_len: int = 20, near_dup: bool = True) -> Tuple[List[Dict], Dict]:
    stats = {"raw": len(rows)}

    # 1) empty / PII
    rows = [r for r in rows if r.get("instruction", "").strip() and r.get("output", "").strip()]
    stats["after_empty"] = len(rows)
    for r in rows:
        r["instruction"] = mask_pii(r["instruction"])
        r["output"] = mask_pii(r["output"])

    # 2) short answers (“好的”“可以的亲”“请联系人工”)
    rows = [r for r in rows if len(r["output"]) >= min_answer_len]
    stats["after_short"] = len(rows)

    # 3) contradiction against current SOP
    kept, conflict = [], 0
    for r in rows:
        if policy_violations(r["output"]):
            conflict += 1
            continue
        kept.append(r)
    rows = kept
    stats["after_policy_conflict"] = len(rows)

    # 4) same normalized query+context but conflicting answers: keep the first SOP-consistent one
    groups = defaultdict(list)
    for r in rows:
        groups[(normalize_query(r["instruction"]), r.get("input", ""))].append(r)
    rows = [g[0] for g in groups.values()]
    stats["after_exact_dedup"] = len(rows)

    # 5) MinHash near-dup on (normalized query + input)
    if near_dup and rows:
        keys = [normalize_query(r["instruction"]) + "|" + r.get("input", "") for r in rows]
        rep = near_dup_groups(keys)
        seen, uniq = set(), []
        for r, g in zip(rows, rep):
            if g in seen:
                continue
            seen.add(g)
            uniq.append(r)
        rows = uniq
    stats["after_near_dedup"] = len(rows)
    stats["final"] = len(rows)
    return rows, stats
