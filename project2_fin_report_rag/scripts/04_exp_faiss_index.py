"""Experiment W6: FAISS index comparison (Flat / IVF / HNSW): recall vs Flat, latency, build time, memory.

真实语料 chunk 只有几千条时三种索引差别很小，所以额外把向量库扩到 --scale 条（真实向量 + 高斯扰动副本），
模拟 200-300 份研报 × 多版本切块后的库规模，结论才有意义。

  python scripts/04_exp_faiss_index.py --scale 200000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finrag.dense import VectorIndex  # noqa: E402
from finrag.exp_utils import ROOT, build_chunks, get_embedder, load_qa, md_table  # noqa: E402
from finrag.utils import write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=200000)
    ap.add_argument("--emb", default=None)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    qa, emb = load_qa(), get_embedder(args.emb)
    chunks = build_chunks("layout", 512, 50, True)
    base = emb.encode_docs([c.embed_text() for c in chunks])
    rng = np.random.default_rng(0)
    reps = []
    while sum(len(r) for r in reps) + len(base) < args.scale:
        noisy = base + rng.normal(0, 0.03, base.shape).astype("float32")
        reps.append(noisy / np.linalg.norm(noisy, axis=1, keepdims=True))
    xb = np.vstack([base] + reps)[:max(args.scale, len(base))].astype("float32")
    xq = emb.encode_queries([q["query"] for q in qa])
    print("db size", xb.shape)

    gt_index = VectorIndex(xb.shape[1], "flat").build(xb)
    _, gt = gt_index.search(xq, args.k)
    nlist = int(4 * np.sqrt(len(xb)))
    configs = [("Flat", dict(kind="flat")),
               (f"IVF(nlist={nlist},nprobe=8)", dict(kind="ivf", nlist=nlist, nprobe=8)),
               (f"IVF(nlist={nlist},nprobe=32)", dict(kind="ivf", nlist=nlist, nprobe=32)),
               ("HNSW(M=32,ef=64)", dict(kind="hnsw", hnsw_m=32, ef_search=64)),
               ("HNSW(M=32,ef=128)", dict(kind="hnsw", hnsw_m=32, ef_search=128))]
    rows = []
    for name, kw in configs:
        t0 = time.perf_counter()
        idx = VectorIndex(xb.shape[1], **kw).build(xb)
        build = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(3):
            _, I = idx.search(xq, args.k)
        lat = (time.perf_counter() - t0) / 3 / len(xq) * 1000
        rec = np.mean([len(set(I[i]) & set(gt[i])) / args.k for i in range(len(xq))])
        rows.append({"索引": name, f"Recall@{args.k}(vs Flat)": round(float(rec), 4), "查询延迟ms": round(lat, 3),
                     "构建耗时s": round(build, 2), "内存MB": round(idx.memory_mb(), 1)})
        print(rows[-1])
    write_json(os.path.join(ROOT, "results", "exp_faiss_index.json"), {"db_size": len(xb), "dim": int(xb.shape[1]),
                                                                        "rows": rows})
    with open(os.path.join(ROOT, "results", "exp_faiss_index.md"), "w", encoding="utf-8") as f:
        f.write(f"# FAISS 索引对比（库规模 {len(xb)} 条，dim={xb.shape[1]}，单线程 CPU）\n\n"
                + md_table(rows, list(rows[0].keys())) + "\n")


if __name__ == "__main__":
    main()
