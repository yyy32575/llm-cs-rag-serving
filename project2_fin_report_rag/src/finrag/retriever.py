"""Hybrid retrieval: dense (FAISS) + sparse (BM25) fused by RRF or weighted min-max scores."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bm25 import BM25Index
from .chunker import Chunk
from .dense import Embedder, VectorIndex
from .query_rewrite import rewrite


def rrf_fuse(rank_lists: Sequence[List[int]], k: int = 60, weights: Optional[Sequence[float]] = None) -> List[Tuple[int, float]]:
    weights = weights or [1.0] * len(rank_lists)
    s: Dict[int, float] = {}
    for w, lst in zip(weights, rank_lists):
        for r, idx in enumerate(lst):
            s[idx] = s.get(idx, 0.0) + w / (k + r + 1)
    return sorted(s.items(), key=lambda x: -x[1])


def weighted_fuse(dense: List[Tuple[int, float]], sparse: List[Tuple[int, float]], alpha: float = 0.5):
    def mm(xs):
        if not xs:
            return {}
        v = np.array([x[1] for x in xs])
        lo, hi = v.min(), v.max()
        return {i: (s - lo) / (hi - lo + 1e-9) for i, s in xs}

    d, sp = mm(dense), mm(sparse)
    keys = set(d) | set(sp)
    return sorted(((i, alpha * d.get(i, 0) + (1 - alpha) * sp.get(i, 0)) for i in keys), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(self, chunks: List[Chunk], embedder: Optional[Embedder] = None, index_kind: str = "flat",
                 title_prefix: bool = True, extra_terms: Sequence[str] = ()):
        self.chunks = chunks
        texts = [c.embed_text(title_prefix) for c in chunks]
        self.bm25 = BM25Index(texts, extra_terms=extra_terms)
        self.embedder = embedder
        if embedder is not None:
            self.vecs = embedder.encode_docs(texts)
            self.index = VectorIndex(self.vecs.shape[1], kind=index_kind,
                                     nlist=max(4, int(np.sqrt(len(chunks))))).build(self.vecs)

    def dense(self, q: str, k: int) -> List[Tuple[int, float]]:
        qv = self.embedder.encode_queries([q])
        D, I = self.index.search(qv, k)
        return [(int(i), float(d)) for d, i in zip(D[0], I[0]) if i >= 0]

    def sparse(self, q: str, k: int) -> List[Tuple[int, float]]:
        return self.bm25.search(q, k)

    def retrieve(self, q: str, mode: str = "hybrid", k: int = 10, cand_k: int = 50, fusion: str = "rrf",
                 alpha: float = 0.5, use_rewrite: bool = True) -> List[Tuple[int, float]]:
        q2 = rewrite(q) if use_rewrite else q
        if mode == "dense":
            return self.dense(q2, k)
        if mode == "bm25":
            return self.sparse(q2, k)
        d, s = self.dense(q2, cand_k), self.sparse(q2, cand_k)
        if fusion == "rrf":
            return rrf_fuse([[i for i, _ in d], [i for i, _ in s]])[:k]
        return weighted_fuse(d, s, alpha)[:k]
