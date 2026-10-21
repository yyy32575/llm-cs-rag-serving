"""Step 0 (optional): convert public e-commerce customer-service datasets into our alpaca format.

Supported inputs
  * jsonl/json with flat fields   : --q_field question --a_field answer [--ctx_field category]
  * jsonl with session messages   : --messages_field messages  (list of {role, content}); each assistant turn
                                    becomes one sample, previous turns become `history`
  * csv                           : same flat fields

Typical sources: GitHub 搜 `e-commerce customer service dataset`、JDDC 公开对话、淘宝/京东问大家人工整理的 FAQ。
Example:
  python scripts/00_convert_public.py --inp jddc.jsonl --messages_field session --out data/raw/public.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ecs.io_utils import write_jsonl  # noqa: E402

ROLE_USER = {"user", "human", "customer", "q", "买家", "顾客"}
ROLE_BOT = {"assistant", "gpt", "bot", "agent", "a", "客服", "waiter"}


def load(path):
    if path.endswith(".csv"):
        with open(path, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    with open(path, encoding="utf-8") as f:
        txt = f.read().strip()
    if txt.startswith("["):
        return json.loads(txt)
    return [json.loads(x) for x in txt.splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--q_field", default="question")
    ap.add_argument("--a_field", default="answer")
    ap.add_argument("--ctx_field", default=None)
    ap.add_argument("--messages_field", default=None)
    ap.add_argument("--max_history", type=int, default=3)
    args = ap.parse_args()

    rows, out = load(args.inp), []
    for i, r in enumerate(rows):
        if args.messages_field:
            msgs = r.get(args.messages_field) or []
            hist, pending_q = [], None
            for m in msgs:
                role = str(m.get("role") or m.get("from") or "").lower()
                text = (m.get("content") or m.get("value") or "").strip()
                if role in ROLE_USER:
                    pending_q = text if pending_q is None else pending_q + "\n" + text
                elif role in ROLE_BOT and pending_q:
                    out.append({"id": f"pub-{i}-{len(out)}", "instruction": pending_q, "input": "", "output": text,
                                "history": hist[-args.max_history:], "meta": {"source": "public", "domain": "unk"}})
                    hist.append([pending_q, text])
                    pending_q = None
        else:
            q, a = (r.get(args.q_field) or "").strip(), (r.get(args.a_field) or "").strip()
            if q and a:
                ctx = r.get(args.ctx_field, "") if args.ctx_field else ""
                out.append({"id": f"pub-{i}", "instruction": q, "input": ctx, "output": a, "history": [],
                            "meta": {"source": "public", "domain": "unk"}})
    n = write_jsonl(args.out, out)
    print(f"converted {n} samples -> {args.out}")


if __name__ == "__main__":
    main()
