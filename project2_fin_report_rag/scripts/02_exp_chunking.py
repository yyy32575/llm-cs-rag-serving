"""Experiment W5: parser x chunk_size x overlap x table protection  ->  Recall@k for dense / BM25 / hybrid.

  EMB_MODEL=BAAI/bge-large-zh-v1.5 python scripts/02_exp_chunking.py
Outputs results/exp_chunking.json + results/exp_chunking.md
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
    ap.add_argument("--sizes", default="256,512,1024")
    ap.add_argument("--overlaps", default="50,100")
    ap.add_argument("--emb", default=None)
    args = ap.parse_args()
    qa = load_qa()
    emb = get_embedder(args.emb)
    rows = []
    grid = []
    for parser in ("naive", "layout"):
        for size in map(int, args.sizes.split(",")):
            for ov in map(int, args.overlaps.split(",")):
                grid.append((parser, size, ov, True))
    grid += [("layout", 512, 50, False)]  # 关闭表格保护的对照
    for parser, size, ov, tp in grid:
        chunks = build_chunks(parser, size, ov, tp)
        res, _ = run_retrieval(qa, chunks, emb)
        for mode, m in res.items():
            rows.append({"parser": parser, "chunk_size": size, "overlap": ov, "table_protect": tp, "mode": mode,
                         "n_chunks": len(chunks), "R@1": m["recall@1"], "R@3": m["recall@3"], "R@5": m["recall@5"],
                         "R@10": m["recall@10"], "MRR": m["mrr"], "表格R@5": m["by_type"].get("表格", {}).get("recall@5")})
        print(parser, size, ov, tp, {k: v["recall@5"] for k, v in res.items()})
    write_json(os.path.join(ROOT, "results", "exp_chunking.json"), {"embedding": emb.model_name, "rows": rows})
    cols = ["parser", "chunk_size", "overlap", "table_protect", "mode", "n_chunks", "R@1", "R@3", "R@5", "R@10", "MRR",
            "表格R@5"]
    with open(os.path.join(ROOT, "results", "exp_chunking.md"), "w", encoding="utf-8") as f:
        f.write(f"# 切块实验（embedding={emb.model_name}）\n\n" + md_table(rows, cols) + "\n")


if __name__ == "__main__":
    main()
