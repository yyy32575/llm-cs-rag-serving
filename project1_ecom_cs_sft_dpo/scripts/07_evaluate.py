"""Step 7: evaluate a model (or the rule baseline) on the held-out test set.

  # 线上规则引擎 baseline
  python scripts/07_evaluate.py --backend rule --exp_name baseline_rule
  # 原始底座 zero-shot
  python scripts/07_evaluate.py --backend vllm --model Qwen/Qwen3-8B --exp_name base_zeroshot
  # SFT / DPO adapter
  python scripts/07_evaluate.py --backend vllm --model Qwen/Qwen3-8B --adapter outputs/sft_qlora_r16 --exp_name sft_r16
  # 部署后的服务（项目三，OpenAI 兼容接口）
  python scripts/07_evaluate.py --backend openai --model cs-model --base_url http://localhost:8000/v1 --exp_name awq_int4
  # 已有预测文件直接打分
  python scripts/07_evaluate.py --pred_file results/preds_sft_r16.jsonl --exp_name sft_r16

Outputs: results/preds_<exp>.jsonl, results/metrics_<exp>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ecs.io_utils import read_jsonl, write_json, write_jsonl  # noqa: E402
from ecs.metrics import aggregate, evaluate_item, llm_judge  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["rule", "reference", "hf", "vllm", "openai"], default="rule")
    ap.add_argument("--model", default=os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--quantization", default=None, help="vllm backend: awq / gptq / None")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--test_file", default=os.path.join(ROOT, "data", "test", "test.jsonl"))
    ap.add_argument("--pred_file", default=None)
    ap.add_argument("--exp_name", default="baseline_rule")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8, help="hf backend batch size")
    ap.add_argument("--judge_model", default=None, help="可选：LLM-as-judge 模型名（OpenAI 兼容，读 JUDGE_BASE_URL）")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = read_jsonl(args.test_file)
    if items and "meta" in items[0]:  # SFT-format file (e.g. data/sft/val.jsonl) -> test schema
        items = [{"id": r["id"], "instruction": r["instruction"], "input": r.get("input", ""),
                  "history": r.get("history", []), "reference": r["output"], "keypoints": r["meta"]["keypoints"],
                  "forbidden": r["meta"]["forbidden"], "transfer": r["meta"]["transfer"],
                  "intent": r["meta"]["intent"], "domain": r["meta"]["domain"],
                  "subset": r["meta"].get("subset", "normal")} for r in items if "keypoints" in r.get("meta", {})]
    if args.limit:
        items = items[:args.limit]

    t0 = time.time()
    if args.pred_file:
        preds = {p["id"]: p["prediction"] for p in read_jsonl(args.pred_file)}
        outputs = [preds[it["id"]] for it in items]
    elif args.backend == "rule":
        from ecs.rule_bot import RuleBot

        outputs = [o[0] for o in RuleBot().generate(items)]
    elif args.backend == "reference":  # sanity check: metric upper bound on gold answers
        outputs = [it["reference"] for it in items]
    else:
        from ecs.generation import get_generator

        gen = get_generator(args.backend, args.model, args.adapter, base_url=args.base_url,
                            quantization=args.quantization, load_4bit=args.load_4bit, batch_size=args.batch_size)
        outputs = [o[0] for o in gen.generate(items, n=1, temperature=0.0, max_new_tokens=args.max_new_tokens)]
    gen_sec = time.time() - t0

    results = [evaluate_item(o, it) for o, it in zip(outputs, items)]
    if args.judge_model:
        from openai import OpenAI

        client = OpenAI(base_url=os.environ.get("JUDGE_BASE_URL"), api_key=os.environ.get("JUDGE_API_KEY", "EMPTY"))
        for r, o, it in zip(results, outputs, items):
            s = llm_judge(client, args.judge_model, it["instruction"], it["reference"], o)
            if s is not None:
                r["satisfaction"] = s

    metrics = aggregate(results, items)
    metrics.update({"exp_name": args.exp_name, "backend": args.backend, "model": args.model,
                    "adapter": args.adapter, "generation_seconds": round(gen_sec, 1)})
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    write_jsonl(os.path.join(ROOT, "results", f"preds_{args.exp_name}.jsonl"),
                [{"id": it["id"], "instruction": it["instruction"], "input": it["input"], "history": it["history"],
                  "subset": it["subset"], "domain": it["domain"], "reference": it["reference"],
                  "prediction": o, **r} for o, it, r in zip(outputs, items, results)])
    write_json(os.path.join(ROOT, "results", f"metrics_{args.exp_name}.json"), metrics)
    print(json.dumps(metrics["overall"], ensure_ascii=False, indent=2))
    print("by_subset:", json.dumps({k: v["accuracy"] for k, v in metrics["by_subset"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
