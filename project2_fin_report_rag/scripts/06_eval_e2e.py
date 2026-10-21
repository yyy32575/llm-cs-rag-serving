"""Experiment W7/W8: end-to-end QA (retrieve -> rerank -> refuse/generate -> verify) + bad-case export.

  # 用项目三部署的 vLLM 服务（OpenAI 兼容）
  LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=Qwen/Qwen3-8B python scripts/06_eval_e2e.py --backend openai
  # 无 GPU：本地小模型（Qwen3-0.6B，CPU 可跑，效果差但链路完整）
  python scripts/06_eval_e2e.py --backend hf --hf_model ./models/Qwen3-0.6B
Ablations: --no_rerank  --no_verify_refuse(阈值=0)  --final_k 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finrag.evaluation import answer_metrics, is_refusal  # noqa: E402
from finrag.exp_utils import DEFAULT_RERANK, ROOT, build_chunks, get_embedder, load_qa  # noqa: E402
from finrag.pipeline import RAGPipeline  # noqa: E402
from finrag.utils import norm_text, write_json, write_jsonl  # noqa: E402


def classify_badcase(q, a):
    if not q["evidence"]:
        return None if is_refusal(a["answer"]) else "幻觉(无答案题未拒答)"
    if a["refused"] or is_refusal(a["answer"]):
        hit = any(norm_text(ev["must"]) in norm_text(s["text"]) for ev in q["evidence"] for s in a["sources"])
        return "误拒答(证据已召回)" if hit else "空召回→拒答"
    ok = all(norm_text(ev["must"]) in norm_text(a["answer"]) for ev in q["evidence"])
    if ok:
        return None
    hit = any(norm_text(ev["must"]) in norm_text(s["text"]) and s["doc_id"] in ev["doc_ids"]
              for ev in q["evidence"] for s in a["sources"])
    if not hit:
        return "召回失败(证据不在Top-k)"
    return "生成错误(证据在但答错)" if a["verified"] else "生成幻觉(数字无出处)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["openai", "hf", "none"], default="openai")
    ap.add_argument("--hf_model", default=os.path.join(ROOT, "models", "Qwen3-0.6B"))
    ap.add_argument("--rerank_model", default=DEFAULT_RERANK)
    ap.add_argument("--emb", default=None)
    ap.add_argument("--no_rerank", action="store_true")
    ap.add_argument("--no_llm_refusal", action="store_true", help="拒答决策前置：生成 prompt 不给模型拒答出口")
    ap.add_argument("--no_constraint", action="store_true", help="关闭查询硬约束校验（只用 rerank 阈值拒答）")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--final_k", type=int, default=5)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    qa = load_qa()
    if args.limit:
        qa = qa[:args.limit]
    chunks = build_chunks("layout", 512, 50, True)
    reranker = None
    if not args.no_rerank:
        from finrag.reranker import Reranker

        reranker = Reranker(args.rerank_model)
    gen = None
    if args.backend == "openai":
        from finrag.generator import Generator

        gen = Generator()
    elif args.backend == "hf":
        from finrag.generator import HFGenerator

        gen = HFGenerator(args.hf_model, device="cpu")
    if gen is not None and args.no_llm_refusal:
        gen.llm_refusal = False
    pipe = RAGPipeline(chunks, get_embedder(args.emb), reranker, gen,
                       cfg={"rerank": not args.no_rerank, "refuse_threshold": args.threshold, "final_k": args.final_k,
                            "constraint_check": not args.no_constraint})
    outs, t0 = [], time.time()
    for i, q in enumerate(qa):
        a = pipe.answer(q["query"])
        a.update({"qid": q["qid"], "type": q["type"], "gold": q["answer"]})
        a["badcase"] = classify_badcase(q, a)
        outs.append(a)
        if i % 10 == 0:
            print(i, q["query"], "->", a["answer"][:80])
    m = answer_metrics(qa, outs, {c.chunk_id: c for c in chunks})
    lat = {k: round(sum(o["latency_ms"][k] for o in outs) / len(outs), 1) for k in ("recall", "rerank", "generate")}
    from collections import Counter

    m.update({"latency_ms_avg": lat, "n": len(qa), "unsupported_number_rate":
              round(sum(bool(o["unsupported_numbers"]) for o in outs) / len(outs), 4),
              "badcase_types": dict(Counter(o["badcase"] for o in outs if o["badcase"])),
              "backend": args.backend, "rerank": not args.no_rerank, "threshold": args.threshold,
              "total_seconds": round(time.time() - t0, 1)})
    write_json(os.path.join(ROOT, "results", f"e2e_{args.tag}.json"), m)
    write_jsonl(os.path.join(ROOT, "results", f"e2e_{args.tag}_answers.jsonl"), outs)
    print(m)


if __name__ == "__main__":
    main()
