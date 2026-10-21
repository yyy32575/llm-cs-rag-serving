"""Step 4: sample 2-3 candidate answers per query from the SFT model (for preference annotation).

  python scripts/04_gen_candidates.py --backend vllm --adapter outputs/sft_qlora_r16 --n_queries 2500 --n 3
Output: data/dpo/candidates.jsonl  {id, instruction, input, history, meta, responses:[...]}
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ecs.generation import get_generator  # noqa: E402
from ecs.io_utils import read_jsonl, write_jsonl  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="vllm", choices=["hf", "vllm", "openai"])
    ap.add_argument("--model", default=os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--adapter", default=os.path.join(ROOT, "outputs", "sft_qlora_r16"))
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--inp", default=os.path.join(ROOT, "data", "sft", "train.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "dpo", "candidates.jsonl"))
    ap.add_argument("--n_queries", type=int, default=2500)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = read_jsonl(args.inp)
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.n_queries]
    gen = get_generator(args.backend, args.model, args.adapter, base_url=args.base_url)
    outs = gen.generate(rows, n=args.n, temperature=args.temperature, top_p=args.top_p)
    n = write_jsonl(args.out, [{**r, "responses": o} for r, o in zip(rows, outs)])
    print(f"wrote {n} queries x {args.n} responses -> {args.out}")


if __name__ == "__main__":
    main()
