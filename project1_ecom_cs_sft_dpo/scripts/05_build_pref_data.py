"""Step 5: build preference pairs (chosen / rejected) for DPO.

标注规则（客服场景）：
  chosen   = 给出明确政策依据 + 下一步具体操作路径（+ 必要的共情）
  rejected = 只说“请联系人工客服” / 政策错误或过期 / 复读 / 答非所问 / 含糊敷衍

Modes
  --mode candidates : 用 SFT 模型采样的 2-3 个候选（04 产出）自动打分；分差不够的进入人工标注队列 CSV
  --mode synthetic  : 冷启动 / 对照实验用，用金标答案作 chosen，规则扰动生成 rejected（5 类典型差回复）
  --annotated_csv   : 合并人工标注结果（列: id, chosen_idx, rejected_idx, skip）
  --noise_ratio     : 噪声实验，随机交换 x% 的 chosen/rejected（10%/20%/30%）

  python scripts/05_build_pref_data.py --mode candidates --inp data/dpo/candidates.jsonl
  python scripts/05_build_pref_data.py --mode synthetic --n_pairs 2000
  python scripts/05_build_pref_data.py --mode synthetic --n_pairs 2000 --noise_ratio 0.3 --tag noise30
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ecs.clean import POLICY_PATTERNS  # noqa: E402
from ecs.io_utils import read_jsonl, write_json, write_jsonl  # noqa: E402
from ecs.metrics import evaluate_item  # noqa: E402
from ecs.sop import OUTDATED_POLICY  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRANSFER_ONLY = ["亲，这个问题建议您联系人工客服咨询哦。", "您好，请联系人工客服处理。", "这边帮您转人工客服，请稍等。",
                 "亲亲，具体请咨询人工客服呢~"]
VAGUE = ["亲，这个要看具体情况的，您可以在订单页面看一下相关规则哦。", "您好，一般是可以的，具体以页面显示为准。",
         "这个不太好说呢，建议您再等等看。"]


def to_item(r):
    m = r.get("meta", {})
    return {"keypoints": m.get("keypoints", []), "forbidden": m.get("forbidden", []),
            "transfer": m.get("transfer", False), "domain": m.get("domain")}


def score(resp, item):
    e = evaluate_item(resp, item)
    s = 2.0 * e["coverage"] + 0.3 * e["satisfaction"]
    s -= 2.0 * e["forbidden"] + 2.0 * e["repetitive"]
    if e["transfer"] and not item["transfer"]:
        s -= 1.5
    if item["transfer"] and not e["transfer"]:
        s -= 1.5
    return s


def to_outdated(ans):
    out = ans
    for key, pats in POLICY_PATTERNS.items():
        for p in pats:
            def _rep(m, key=key):
                g = m.group(0)
                a, b = m.span(1)
                a -= m.start(0)
                b -= m.start(0)
                return g[:a] + str(OUTDATED_POLICY[key]) + g[b:]
            out = re.sub(p, _rep, out)
    return out if out != ans else None


def perturb(r, pool, rng):
    gold = r["output"]
    kinds = ["transfer_only", "vague", "outdated", "repetitive", "off_topic"]
    weights = [0.35, 0.2, 0.2, 0.1, 0.15]
    if r.get("meta", {}).get("transfer"):  # 该转人工的样本，反例是“不转人工还瞎承诺”
        return "亲，没问题的，我们一定按您要求的三倍赔偿给您，请放心。", "over_promise"
    for _ in range(5):
        k = rng.choices(kinds, weights)[0]
        if k == "transfer_only":
            return rng.choice(TRANSFER_ONLY), k
        if k == "vague":
            return rng.choice(VAGUE), k
        if k == "outdated":
            o = to_outdated(gold)
            if o:
                return o, k
        if k == "repetitive":
            first = re.split(r"[。！]", gold)[0]
            return "。".join([first] * 3) + "。", k
        if k == "off_topic":
            other = rng.choice(pool)
            if other.get("meta", {}).get("intent") != r.get("meta", {}).get("intent"):
                return other["output"], k
    return rng.choice(TRANSFER_ONLY), "transfer_only"


def pair_row(r, chosen, rejected, kind):
    return {"id": r["id"], "instruction": r["instruction"], "input": r.get("input", ""),
            "history": r.get("history", []), "chosen": chosen, "rejected": rejected,
            "meta": {**r.get("meta", {}), "rejected_type": kind}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["candidates", "synthetic"], default="synthetic")
    ap.add_argument("--inp", default=None)
    ap.add_argument("--annotated_csv", default=None)
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "data", "dpo"))
    ap.add_argument("--tag", default="clean")
    ap.add_argument("--n_pairs", type=int, default=2000)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--noise_ratio", type=float, default=0.0)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    pairs, queue = [], []
    if args.mode == "synthetic":
        rows = read_jsonl(args.inp or os.path.join(ROOT, "data", "sft", "train.jsonl"))
        rows = [r for r in rows if "keypoints" in r.get("meta", {})]
        rng.shuffle(rows)
        for r in rows[:args.n_pairs]:
            rej, kind = perturb(r, rows, rng)
            pairs.append(pair_row(r, r["output"], rej, kind))
    else:
        rows = read_jsonl(args.inp or os.path.join(ROOT, "data", "dpo", "candidates.jsonl"))
        ann = {}
        if args.annotated_csv:
            with open(args.annotated_csv, encoding="utf-8-sig") as f:
                for a in csv.DictReader(f):
                    if str(a.get("skip", "")).strip() not in ("1", "true", "True"):
                        ann[a["id"]] = (int(a["chosen_idx"]), int(a["rejected_idx"]))
        for r in rows:
            resps = [x for x in r["responses"] if x]
            if r["id"] in ann:
                c, j = ann[r["id"]]
                pairs.append(pair_row(r, resps[c], resps[j], "human"))
                continue
            item = to_item(r)
            scored = sorted(((score(x, item), x) for x in resps), key=lambda t: t[0], reverse=True)
            if len(scored) >= 2 and scored[0][0] - scored[-1][0] >= args.margin:
                pairs.append(pair_row(r, scored[0][1], scored[-1][1], "auto_score"))
            elif scored and scored[0][0] < 1.0:  # 全部候选都差：用金标答案兜底作 chosen
                pairs.append(pair_row(r, r["output"], scored[0][1], "gold_vs_model"))
            else:
                queue.append(r)
        if queue:
            p = os.path.join(args.out_dir, "to_annotate.csv")
            os.makedirs(args.out_dir, exist_ok=True)
            with open(p, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "instruction", "input", "resp_0", "resp_1", "resp_2", "chosen_idx", "rejected_idx",
                            "skip"])
                for r in queue:
                    rs = (r["responses"] + ["", "", ""])[:3]
                    w.writerow([r["id"], r["instruction"], r.get("input", ""), *rs, "", "", ""])
            print(f"{len(queue)} ambiguous queries -> {p} (人工标注后用 --annotated_csv 合并)")
        pairs = pairs[:args.n_pairs]

    n_noise = int(len(pairs) * args.noise_ratio)
    for i in rng.sample(range(len(pairs)), n_noise):
        p = pairs[i]
        p["chosen"], p["rejected"] = p["rejected"], p["chosen"]
        p["meta"]["noisy"] = True

    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_ratio))
    suffix = "" if args.tag == "clean" else f"_{args.tag}"
    write_jsonl(os.path.join(args.out_dir, f"val{suffix}.jsonl"), pairs[:n_val])
    write_jsonl(os.path.join(args.out_dir, f"train{suffix}.jsonl"), pairs[n_val:])
    stats = {"mode": args.mode, "pairs": len(pairs), "train": len(pairs) - n_val, "val": n_val,
             "noise_ratio": args.noise_ratio, "noisy_pairs": n_noise, "annotation_queue": len(queue),
             "rejected_type": dict(Counter(p["meta"]["rejected_type"] for p in pairs))}
    write_json(os.path.join(args.out_dir, f"pref_stats{suffix}.json"), stats)
    print(stats)


if __name__ == "__main__":
    main()
