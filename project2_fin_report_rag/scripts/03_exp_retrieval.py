"""Experiment W6: retrieval routes (dense / BM25 / hybrid RRF / hybrid weighted) + query rewrite + title prefix.

  python scripts/03_exp_retrieval.py --chunk_size 512 --overlap 50
Outputs results/exp_retrieval.json/.md (含按问题类型拆分的 Recall@5)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finrag.exp_utils import ROOT, build_chunks, get_embedder, load_qa, md_table, run_retrieval  # noqa: E402
from finrag.utils import write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--emb", default=None)
    args = ap.parse_args()
    qa, emb = load_qa(), get_embedder(args.emb)
    chunks = build_chunks("layout", args.chunk_size, args.overlap, True)
    settings = [
        ("纯向量(FAISS)", dict(modes=("dense",), use_rewrite=False)),
        ("纯BM25", dict(modes=("bm25",), use_rewrite=False)),
        ("混合RRF", dict(modes=("hybrid",), use_rewrite=False, fusion="rrf")),
        ("混合加权α=0.3", dict(modes=("hybrid",), use_rewrite=False, fusion="weighted", alpha=0.3)),
        ("混合加权α=0.5", dict(modes=("hybrid",), use_rewrite=False, fusion="weighted", alpha=0.5)),
        ("混合加权α=0.7", dict(modes=("hybrid",), use_rewrite=False, fusion="weighted", alpha=0.7)),
        ("纯向量+Query改写", dict(modes=("dense",), use_rewrite=True)),
        ("纯BM25+Query改写", dict(modes=("bm25",), use_rewrite=True)),
        ("混合RRF+Query改写", dict(modes=("hybrid",), use_rewrite=True, fusion="rrf")),
        ("混合RRF+改写-去标题前缀", dict(modes=("hybrid",), use_rewrite=True, fusion="rrf", title_prefix=False)),
    ]
    rows, raw = [], {}
    types = ["事实", "事实-年份混淆", "同义改写", "表格", "对比", "汇总", "政策"]
    for name, kw in settings:
        res, _ = run_retrieval(qa, chunks, emb, **kw)
        m = list(res.values())[0]
        raw[name] = m
        row = {"方案": name, "R@1": m["recall@1"], "R@3": m["recall@3"], "R@5": m["recall@5"], "R@10": m["recall@10"],
               "MRR": m["mrr"], "ms/query": m["latency_ms_per_query"]}
        for t in types:
            row[t] = m["by_type"].get(t, {}).get("recall@5", "")
        rows.append(row)
        print(name, row)
    write_json(os.path.join(ROOT, "results", "exp_retrieval.json"),
               {"embedding": emb.model_name, "chunk_size": args.chunk_size, "overlap": args.overlap, "results": raw})
    with open(os.path.join(ROOT, "results", "exp_retrieval.md"), "w", encoding="utf-8") as f:
        f.write(f"# 多路召回实验（embedding={emb.model_name}, chunk={args.chunk_size}/{args.overlap}）\n\n")
        f.write(md_table(rows, ["方案", "R@1", "R@3", "R@5", "R@10", "MRR", "ms/query"]) + "\n\n")
        f.write("## 按问题类型 Recall@5\n\n" + md_table(rows, ["方案"] + types) + "\n")


if __name__ == "__main__":
    main()
