"""Sparse retrieval: BM25 (rank_bm25) + jieba with a finance user dictionary."""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import numpy as np

FIN_TERMS = [
    "营业收入", "归母净利润", "扣非净利润", "毛利率", "净利率", "净息差", "不良贷款率", "拨备覆盖率", "市场份额", "市占率",
    "装车量", "装机量", "出货量", "产能利用率", "稼动率", "新签订单", "直销收入占比", "创新药收入占比", "研发投入",
    "盈利预测", "每股收益", "同比增长", "环比提升", "前三季度", "第三季度", "三季度", "以旧换新", "加计扣除",
    "核心一级资本充足率", "资本充足率", "超长期特别国债", "育儿补贴", "海外收入占比", "液态奶收入",
]
STOP = set("的了和是在与及或等为对于其中我们公司这那个吗呢啊吧多少什么怎么是否请问？?，,。、：:（）()《》“”")
_jieba_ready = False


def _init_jieba(extra_terms: Sequence[str] = ()):
    global _jieba_ready
    import jieba

    if not _jieba_ready:
        jieba.setLogLevel(60)
        for t in list(FIN_TERMS) + list(extra_terms):
            jieba.add_word(t, freq=200000)
        _jieba_ready = True
    return jieba


def tokenize(text: str) -> List[str]:
    jieba = _init_jieba()
    toks = []
    for t in jieba.lcut_for_search(text):
        t = t.strip().lower()
        if not t or t in STOP or re.fullmatch(r"[\W_]+", t):
            continue
        toks.append(t)
    return toks


class BM25Index:
    def __init__(self, texts: List[str], k1: float = 1.5, b: float = 0.75, extra_terms: Sequence[str] = ()):
        from rank_bm25 import BM25Okapi

        _init_jieba(extra_terms)
        self.bm25 = BM25Okapi([tokenize(t) for t in texts], k1=k1, b=b)

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx]
