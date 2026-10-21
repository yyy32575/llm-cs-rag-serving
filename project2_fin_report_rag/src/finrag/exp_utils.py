"""Shared helpers for experiment scripts."""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

from .chunker import chunk_blocks
from .evaluation import retrieval_metrics
from .retriever import HybridRetriever
from .utils import read_jsonl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_EMB = os.environ.get("EMB_MODEL", "BAAI/bge-large-zh-v1.5")
DEFAULT_RERANK = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")


def load_qa(path: Optional[str] = None) -> List[Dict]:
    return read_jsonl(path or os.path.join(ROOT, "data", "eval", "qa.jsonl"))


def load_blocks(parser: str = "layout") -> List[Dict]:
    return read_jsonl(os.path.join(ROOT, "data", "parsed", f"{parser}.jsonl"))


def get_embedder(name: Optional[str] = None):
    from .dense import Embedder

    return Embedder(name or DEFAULT_EMB, cache_dir=os.path.join(ROOT, "data", "cache"))


def run_retrieval(qa, chunks, embedder, modes=("dense", "bm25", "hybrid"), k=20, use_rewrite=True,
                  title_prefix=True, fusion="rrf", alpha=0.5, index_kind="flat") -> Dict[str, Dict]:
    r = HybridRetriever(chunks, embedder, index_kind=index_kind, title_prefix=title_prefix)
    out = {}
    for m in modes:
        t0 = time.perf_counter()
        ranked = [[i for i, _ in r.retrieve(q["query"], mode=m, k=k, fusion=fusion, alpha=alpha,
                                            use_rewrite=use_rewrite)] for q in qa]
        ms = (time.perf_counter() - t0) * 1000 / max(1, len(qa))
        met = retrieval_metrics(qa, ranked, chunks)
        met["latency_ms_per_query"] = round(ms, 2)
        out[m] = met
    return out, r


def md_table(rows: List[Dict], cols: List[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def build_chunks(parser="layout", size=512, overlap=50, table_protect=True):
    return chunk_blocks(load_blocks(parser), size, overlap, table_protect)
