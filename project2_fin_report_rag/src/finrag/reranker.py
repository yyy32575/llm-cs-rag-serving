"""Cross-encoder reranker (bge-reranker-v2-m3 / bge-reranker-base)."""
from __future__ import annotations

import math
from typing import List, Optional


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: Optional[str] = None, max_length: int = 512,
                 batch_size: int = 16):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        self.bs = batch_size

    def score(self, query: str, passages: List[str]) -> List[float]:
        """Relevance in [0, 1]（sentence-transformers 对单标签 CrossEncoder 默认已做 sigmoid；否则这里补做）。
        归一化分数用于“低置信度拒答”阈值。"""
        if not passages:
            return []
        raw = self.model.predict([(query, p) for p in passages], batch_size=self.bs, show_progress_bar=False)
        out = [float(s) for s in list(raw)]
        if any(s < 0 or s > 1 for s in out):
            out = [1 / (1 + math.exp(-s)) for s in out]
        return out
