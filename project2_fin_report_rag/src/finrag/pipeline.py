"""End-to-end RAG pipeline: rewrite -> hybrid recall (top-20) -> rerank (top-5) -> threshold -> generate -> verify."""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from .chunker import Chunk, chunk_blocks
from .generator import REFUSAL, check_constraints, verify_citations
from .retriever import HybridRetriever
from .utils import read_jsonl


class RAGPipeline:
    def __init__(self, chunks: List[Chunk], embedder=None, reranker=None, generator=None, cfg: Optional[Dict] = None):
        self.cfg = {"mode": "hybrid", "fusion": "rrf", "recall_k": 20, "final_k": 5, "rerank": True,
                    "refuse_threshold": 0.15, "constraint_check": True, "use_rewrite": True, "title_prefix": True,
                    "index": "flat", **(cfg or {})}
        self.chunks = chunks
        self.retriever = HybridRetriever(chunks, embedder, index_kind=self.cfg["index"],
                                         title_prefix=self.cfg["title_prefix"])
        self.reranker, self.generator = reranker, generator

    @classmethod
    def from_parsed(cls, parsed_jsonl: str, chunk_size=512, overlap=50, table_protect=True, **kw):
        return cls(chunk_blocks(read_jsonl(parsed_jsonl), chunk_size, overlap, table_protect), **kw)

    def retrieve(self, q: str) -> Dict:
        c = self.cfg
        t0 = time.perf_counter()
        mode = c["mode"] if self.retriever.embedder is not None else "bm25"
        cands = self.retriever.retrieve(q, mode=mode, k=c["recall_k"], fusion=c["fusion"], use_rewrite=c["use_rewrite"])
        t1 = time.perf_counter()
        idxs = [i for i, _ in cands]
        scores = [None] * len(idxs)
        if c["rerank"] and self.reranker is not None and idxs:
            rs = self.reranker.score(q, [self.chunks[i].embed_text(c["title_prefix"]) for i in idxs])
            order = sorted(range(len(idxs)), key=lambda j: -rs[j])
            idxs, scores = [idxs[j] for j in order], [rs[j] for j in order]
        t2 = time.perf_counter()
        return {"idxs": idxs, "rerank_scores": scores, "recall_ms": (t1 - t0) * 1000, "rerank_ms": (t2 - t1) * 1000}

    def answer(self, q: str) -> Dict:
        r = self.retrieve(q)
        k = self.cfg["final_k"]
        top = [self.chunks[i] for i in r["idxs"][:k]]
        best = r["rerank_scores"][0] if r["rerank_scores"] and r["rerank_scores"][0] is not None else None
        t0 = time.perf_counter()
        missing = check_constraints(q, top) if (self.cfg["constraint_check"] and top) else []
        if not top or (best is not None and best < self.cfg["refuse_threshold"]) or missing:
            ans, ver, refused = REFUSAL, {"cited": [], "unsupported_numbers": [], "verified": True}, True
        elif self.generator is None:
            ans, ver, refused = "（未配置生成模型，仅返回检索结果）", {"cited": [], "unsupported_numbers": [], "verified": True}, False
        else:
            ans = self.generator(q, top)
            ver = verify_citations(ans, top)
            refused = False
        gen_ms = (time.perf_counter() - t0) * 1000
        cited_ids = [top[i - 1].chunk_id for i in ver["cited"]]
        return {"query": q, "answer": ans, "refused": refused, "verified": ver["verified"],
                "missing_constraints": missing,
                "unsupported_numbers": ver["unsupported_numbers"], "cited_chunk_ids": cited_ids,
                "top_score": best, "sources": [{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "page": c.page,
                                                 "text": c.text[:300]} for c in top],
                "latency_ms": {"recall": round(r["recall_ms"], 1), "rerank": round(r["rerank_ms"], 1),
                               "generate": round(gen_ms, 1)}}


def build_default_pipeline(root: str, with_generator: bool = True) -> RAGPipeline:
    """Build from env/config — used by the Streamlit demo and the FastAPI service (project 3)."""
    from .dense import Embedder
    from .utils import load_config

    cfg = load_config(os.path.join(root, "configs", "default.yaml"))
    # 环境变量可覆盖（CPU 演示：EMB_MODEL=BAAI/bge-small-zh-v1.5 RERANK_MODEL=BAAI/bge-reranker-base）
    cfg["embedding_model"] = os.environ.get("EMB_MODEL", cfg["embedding_model"])
    cfg["reranker_model"] = os.environ.get("RERANK_MODEL", cfg.get("reranker_model"))
    cfg["llm_model"] = os.environ.get("LLM_MODEL", cfg.get("llm_model"))
    cfg["llm_base_url"] = os.environ.get("LLM_BASE_URL", cfg.get("llm_base_url"))
    emb = Embedder(cfg["embedding_model"], cache_dir=os.path.join(root, "data", "cache"))
    rer = None
    if cfg.get("reranker_model"):
        from .reranker import Reranker

        rer = Reranker(cfg["reranker_model"])
    gen = None
    if with_generator:
        from .generator import Generator

        gen = Generator(cfg.get("llm_model"), cfg.get("llm_base_url"))
    return RAGPipeline.from_parsed(os.path.join(root, "data", "parsed", "layout.jsonl"), cfg["chunk_size"],
                                   cfg["overlap"], cfg["table_protect"], embedder=emb, reranker=rer, generator=gen,
                                   cfg=cfg["pipeline"])


if __name__ == "__main__":  # quick manual test
    import sys

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = build_default_pipeline(root, with_generator=False)
    print(json.dumps(p.answer(sys.argv[1] if len(sys.argv) > 1 else "宁川动力2025年三季度装机量是多少"),
                     ensure_ascii=False, indent=2))
