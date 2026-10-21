"""Step 1: construct raw SFT data + held-out test set.

Data sources (all merged into alpaca format {instruction, input, output, history, meta}):
  1. SOP template expansion (this script)          -> 业务规则模板扩写，覆盖退换货/物流/商品参数/投诉/售后/优惠
  2. Public datasets (optional, --public_jsonl)     -> 先用 scripts/00_convert_public.py 转成同格式
  3. LLM paraphrase augmentation (optional)         -> --llm_augment N  使用 OpenAI 兼容接口改写用户问法

Raw data deliberately contains real-world noise (duplicates, “好的亲”式短回复, 过期政策答案)
so that step 2 cleaning is meaningful and its statistics can be reported.

Example:
  python scripts/01_build_sft_data.py --n_raw 12000 --n_test 400
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ecs.io_utils import read_jsonl, write_json, write_jsonl  # noqa: E402
from ecs.sop import (INTENT_BY_NAME, INTENTS, OUTDATED_POLICY, POLICY, POLICY_VERSION,  # noqa: E402
                     attach_item, augment_query, render_input, render_query)

SHORT_REPLIES = ["好的亲", "可以的", "请联系人工客服", "稍等", "亲亲在的哦", "不支持哦", "已反馈", "收到~"]
CHANGE_MIND = {"train": ["算了不退了，能换个尺码吗", "那我不退了，换货可以吗", "还是换一个吧，不退了"],
               "test": ["不退了不退了，帮我换大一码", "我改主意了，换个码数吧"]}
FOLLOW_ADDR = {"train": ["那能改地址吗", "顺便问下地址还能改吗"], "test": ["那我想换个收货地址", "地址还来得及改吗"]}


def sample_intent(rng):
    weights = [i.weight for i in INTENTS]
    return rng.choices(INTENTS, weights=weights, k=1)[0]


def make_single(rng, split="train", policy=POLICY, aug=True, item_p=0.85):
    it = sample_intent(rng)
    slots = attach_item(it, it.slot_fn(rng), rng, item_p)
    q = render_query(it, slots, rng, split)
    if aug and split == "train":
        q = augment_query(q, rng)
    res = it.build(slots, rng, policy)
    return it, slots, q, res


def make_multi_turn(rng, split="train", policy=POLICY):
    """Two multi-turn patterns: (a) 退货->突然变卦要换货  (b) 查物流->追问改地址"""
    if rng.random() < 0.5:
        first = INTENT_BY_NAME["return_no_reason"]
        s1 = {"category": "服装", "days": str(rng.randint(1, 7))}
        q1 = render_query(first, s1, rng, split)
        a1 = first.build(s1, rng, policy)["answer"]
        second = INTENT_BY_NAME["exchange_size"]
        s2 = dict(s1)
        q2 = rng.choice(CHANGE_MIND[split])
        r2 = second.build(s2, rng, policy)
        r2["answer"] = "好的，已为您取消退货意向~" + r2["answer"]
        kind = "multi_turn_change"
        it, slots = second, s2
    else:
        first = INTENT_BY_NAME["track"]
        s1 = {"status": rng.choice(["未发货", "已发货", "运输中"]), "courier": rng.choice(["顺丰", "中通", "圆通"])}
        q1 = render_query(first, s1, rng, split)
        a1 = first.build(s1, rng, policy)["answer"]
        second = INTENT_BY_NAME["change_address"]
        s2 = dict(s1)
        q2 = rng.choice(FOLLOW_ADDR[split])
        r2 = second.build(s2, rng, policy)
        kind = "multi_turn_follow"
        it, slots = second, s2
    return it, slots, [[q1, a1]], q2, r2, kind


def to_row(idx, it, slots, q, res, history=None, source="sop_template", noise=None, subset="normal"):
    return {
        "id": f"sft-{idx:06d}",
        "instruction": q,
        "input": render_input(it, slots),
        "output": res["answer"],
        "history": history or [],
        "meta": {"intent": it.name, "domain": it.domain, "keypoints": res["keypoints"],
                 "forbidden": res["forbidden"], "transfer": res["transfer"], "policy_version": POLICY_VERSION,
                 "source": source, "noise": noise, "subset": subset},
    }


def subset_of(it, slots):
    if it.name == "complain_escalate":
        return "escalate"
    d = slots.get("days")
    if d is not None:
        lim = {"price_protect": POLICY["price_protect_days"], "return_quality": POLICY["quality_exchange_days"]}.get(
            it.name, POLICY["no_reason_days"])
        if int(d) > lim:
            return "beyond_window"
    return "normal"


def build_raw(n, rng, short_ratio, conflict_ratio, mt_ratio, item_p=0.85):
    rows = []
    for i in range(n):
        r = rng.random()
        if r < mt_ratio:
            it, slots, hist, q, res, kind = make_multi_turn(rng)
            rows.append(to_row(i, it, slots, q, res, history=hist, subset=kind))
            continue
        policy, noise = POLICY, None
        if r < mt_ratio + conflict_ratio:
            policy, noise = OUTDATED_POLICY, "outdated_policy"
        it, slots, q, res = make_single(rng, "train", policy, item_p=item_p)
        if noise is None and rng.random() < short_ratio:
            res = dict(res, answer=rng.choice(SHORT_REPLIES))
            noise = "short_reply"
        rows.append(to_row(i, it, slots, q, res, noise=noise, subset=subset_of(it, slots)))
    return rows


def build_test(n, rng):
    rows, seen = [], set()
    tries = 0
    while len(rows) < n and tries < n * 50:
        tries += 1
        if rng.random() < 0.12:
            it, slots, hist, q, res, kind = make_multi_turn(rng, split="test")
            subset = kind
        else:
            it, slots, q, res = make_single(rng, "test", POLICY, aug=False)
            hist, subset = [], subset_of(it, slots)
        inp = render_input(it, slots)
        key = (q, inp, str(hist))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"id": f"test-{len(rows):05d}", "instruction": q, "input": inp, "history": hist,
                     "reference": res["answer"], "keypoints": res["keypoints"], "forbidden": res["forbidden"],
                     "transfer": res["transfer"], "intent": it.name, "domain": it.domain, "subset": subset})
    return rows


def llm_augment(rows, n, rng, model):
    """Paraphrase user queries with an OpenAI-compatible LLM (cheap & controllable)."""
    from openai import OpenAI

    client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"), api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    out = []
    for r in rng.sample(rows, min(n, len(rows))):
        prompt = ("把下面这句电商用户咨询改写成3种不同的口语化说法，保持语义完全一致，每行一句，不要编号：\n"
                  f"{r['instruction']}")
        try:
            resp = client.chat.completions.create(model=model, temperature=0.9,
                                                  messages=[{"role": "user", "content": prompt}])
            for line in (resp.choices[0].message.content or "").splitlines():
                line = line.strip(" -0123456789.、")
                if 4 <= len(line) <= 60:
                    nr = {**r, "instruction": line, "id": r["id"] + f"-aug{len(out)}",
                          "meta": {**r["meta"], "source": "llm_aug"}}
                    out.append(nr)
        except Exception as e:
            print("[llm_augment] skip:", e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_raw", type=int, default=12000)
    ap.add_argument("--n_test", type=int, default=400)
    ap.add_argument("--short_ratio", type=float, default=0.10)
    ap.add_argument("--conflict_ratio", type=float, default=0.08)
    ap.add_argument("--multi_turn_ratio", type=float, default=0.10)
    ap.add_argument("--item_p", type=float, default=0.65, help="背景信息中附带购买商品的概率（控制多样性）")
    ap.add_argument("--public_jsonl", nargs="*", default=[])
    ap.add_argument("--llm_augment", type=int, default=0)
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw = build_raw(args.n_raw, rng, args.short_ratio, args.conflict_ratio, args.multi_turn_ratio, args.item_p)
    for p in args.public_jsonl:
        pub = read_jsonl(p)
        for r in pub:
            r.setdefault("meta", {}).setdefault("source", "public")
        raw.extend(pub)
        print(f"merged public data {p}: {len(pub)}")
    if args.llm_augment:
        raw.extend(llm_augment(raw, args.llm_augment, rng, args.llm_model))

    test = build_test(args.n_test, random.Random(args.seed + 1))
    n1 = write_jsonl(os.path.join(args.out_dir, "raw", "sft_raw.jsonl"), raw)
    n2 = write_jsonl(os.path.join(args.out_dir, "test", "test.jsonl"), test)

    from collections import Counter
    stat = {
        "raw_total": n1, "test_total": n2,
        "raw_noise": dict(Counter(r["meta"].get("noise") or "clean" for r in raw)),
        "raw_domain": dict(Counter(r["meta"]["domain"] for r in raw if "domain" in r["meta"])),
        "test_subset": dict(Counter(r["subset"] for r in test)),
        "test_domain": dict(Counter(r["domain"] for r in test)),
    }
    write_json(os.path.join(args.out_dir, "raw", "build_stats.json"), stat)
    print(stat)


if __name__ == "__main__":
    main()
