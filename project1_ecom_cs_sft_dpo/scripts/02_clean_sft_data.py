"""Step 2: clean raw SFT data and split train / val.

  python scripts/02_clean_sft_data.py --inp data/raw/sft_raw.jsonl --out_dir data/sft
Outputs
  data/sft/train.jsonl, data/sft/val.jsonl           (alpaca format, also used by LLaMA-Factory)
  data/sft/clean_stats.json                           (每条规则过滤了多少，面试时可直接报)
  data/sft/removed_samples.jsonl                      (被过滤样本留档，便于抽检)
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ecs.clean import clean  # noqa: E402
from ecs.io_utils import read_jsonl, write_json, write_jsonl  # noqa: E402


def main():
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=os.path.join(here, "..", "data", "raw", "sft_raw.jsonl"))
    ap.add_argument("--out_dir", default=os.path.join(here, "..", "data", "sft"))
    ap.add_argument("--min_answer_len", type=int, default=20)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--no_near_dup", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = read_jsonl(args.inp)
    kept, stats = clean([dict(r) for r in raw], args.min_answer_len, near_dup=not args.no_near_dup)
    kept_ids = {r["id"] for r in kept}
    removed = [r for r in raw if r["id"] not in kept_ids]

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    n_val = max(1, int(len(kept) * args.val_ratio))
    val, train = kept[:n_val], kept[n_val:]

    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val)
    write_jsonl(os.path.join(args.out_dir, "removed_samples.jsonl"), removed)
    stats.update({
        "train": len(train), "val": len(val),
        "removed_by_noise_tag": dict(Counter((r.get("meta") or {}).get("noise") or "duplicate" for r in removed)),
        "kept_noise_leak": dict(Counter((r.get("meta") or {}).get("noise") or "clean" for r in kept)),
        "train_domain": dict(Counter((r.get("meta") or {}).get("domain", "unk") for r in train)),
    })
    write_json(os.path.join(args.out_dir, "clean_stats.json"), stats)
    for k, v in stats.items():
        print(f"{k:>24}: {v}")


if __name__ == "__main__":
    main()
