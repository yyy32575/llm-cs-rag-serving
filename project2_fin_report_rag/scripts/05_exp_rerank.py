"""Experiment W7: reranking (hybrid Top20 -> cross-encoder -> Top5) + refusal-threshold sweep.

  RERANK_MODEL=BAAI/bge-reranker-v2-m3 python scripts/05_exp_rerank.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finrag.evaluation import retrieval_metrics  # noqa: E402
from finrag.exp_utils import DEFAULT_RERANK, ROOT, build_chunks, get_embedder, load_qa, md_table  # noqa: E402
from finrag.reranker import Reranker  # noqa: E402
from finrag.retriever import HybridRetriever  # noqa: E402
from finrag.utils import write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank_model", default=DEFAULT_RERANK)
    ap.add_argument("--emb", default=None)
    ap.add_argument("--recall_k", type=int, default=20)
    args = ap.parse_args()
    qa, emb = load_qa(), get_embedder(args.emb)
    chunks = build_chunks("layout", 512, 50, True)
    r = HybridRetriever(chunks, emb)
    rr = Reranker(args.rerank_model)

    base_ranked, rer_ranked, top_scores, rer_ms = [], [], [], []
    for q in qa:
        idxs = [i for i, _ in r.retrieve(q["query"], mode="hybrid", k=args.recall_k)]
        base_ranked.append(idxs)
        t0 = time.perf_counter()
        sc = rr.score(q["query"], [chunks[i].embed_text() for i in idxs])
        rer_ms.append((time.perf_counter() - t0) * 1000)
        order = sorted(range(len(idxs)), key=lambda j: -sc[j])
        rer_ranked.append([idxs[j] for j in order])
        top_scores.append(max(sc) if sc else 0.0)

    m0, m1 = retrieval_metrics(qa, base_ranked, chunks), retrieval_metrics(qa, rer_ranked, chunks)
    rows = [{"方案": f"混合召回Top{args.recall_k}直接取前k", "R@1": m0["recall@1"], "R@3": m0["recall@3"],
             "R@5": m0["recall@5"], "MRR": m0["mrr"], "额外延迟ms": 0},
            {"方案": f"混合召回Top{args.recall_k}→{os.path.basename(args.rerank_model)}重排", "R@1": m1["recall@1"],
             "R@3": m1["recall@3"], "R@5": m1["recall@5"], "MRR": m1["mrr"],
             "额外延迟ms": round(sum(rer_ms) / len(rer_ms), 1)}]
    # refusal threshold sweep (answerable vs unanswerable top rerank score)
    sweep = []
    for th in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        unans = [s for s, q in zip(top_scores, qa) if not q["evidence"]]
        ans = [s for s, q in zip(top_scores, qa) if q["evidence"]]
        sweep.append({"阈值": th, "无答案题拒答率": round(sum(s < th for s in unans) / max(1, len(unans)), 4),
                      "有答案题误拒率": round(sum(s < th for s in ans) / max(1, len(ans)), 4)})
    types = sorted({q["type"] for q in qa if q["evidence"]})
    by_type = [{"类型": t, "重排前R@1": m0["by_type"][t]["recall@1"], "重排后R@1": m1["by_type"][t]["recall@1"],
                "重排前R@5": m0["by_type"][t]["recall@5"], "重排后R@5": m1["by_type"][t]["recall@5"]} for t in types]
    write_json(os.path.join(ROOT, "results", "exp_rerank.json"),
               {"embedding": emb.model_name, "reranker": args.rerank_model, "rows": rows, "by_type": by_type,
                "threshold_sweep": sweep})
    with open(os.path.join(ROOT, "results", "exp_rerank.md"), "w", encoding="utf-8") as f:
        f.write(f"# 重排实验（embedding={emb.model_name}, reranker={args.rerank_model}）\n\n")
        f.write(md_table(rows, list(rows[0].keys())) + "\n\n## 分类型\n\n" + md_table(by_type, list(by_type[0].keys())))
        f.write("\n\n## 拒答阈值扫描（rerank Top1 分数）\n\n" + md_table(sweep, list(sweep[0].keys())) + "\n")
    for x in rows + sweep:
        print(x)


if __name__ == "__main__":
    main()
