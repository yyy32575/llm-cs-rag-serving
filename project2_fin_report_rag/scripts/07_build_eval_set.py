"""Helper for REAL research reports: semi-automatically build an eval set (LLM 出题 + 人工校验).

流程：随机抽取含数字的 chunk -> LLM 生成“问题/答案/证据原文” -> 导出 CSV 人工逐条校验（改错、删除、补对比/汇总题）
     -> 用 --import_csv 转成 data/eval/qa.jsonl（格式与 demo 评测集一致）。

  python scripts/07_build_eval_set.py --export data/eval/qa_draft.csv --n 150
  python scripts/07_build_eval_set.py --import_csv data/eval/qa_reviewed.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finrag.exp_utils import ROOT, build_chunks  # noqa: E402
from finrag.utils import write_jsonl  # noqa: E402

PROMPT = """根据下面的研报片段出1道问答题，要求：答案是片段中明确出现的一个数字或短语；问题要包含公司/行业名和时间。
以JSON输出：{{"query": "...", "answer": "...", "must": "答案在原文中的原样字符串", "type": "事实|表格|政策"}}
片段：{text}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=None)
    ap.add_argument("--import_csv", default=None)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "Qwen/Qwen3-8B"))
    args = ap.parse_args()
    if args.import_csv:
        rows = []
        with open(args.import_csv, encoding="utf-8-sig") as f:
            for i, r in enumerate(csv.DictReader(f)):
                if r.get("keep", "1") not in ("1", "y", "Y"):
                    continue
                docs = [d for d in re.split(r"[;；,，]", r["doc_ids"]) if d]
                ev = [{"doc_ids": docs, "must": m} for m in re.split(r"[;；]", r["must"]) if m] if r["type"] != "无答案" else []
                rows.append({"qid": f"q{i:03d}", "type": r["type"], "query": r["query"], "answer": r["answer"],
                             "evidence": ev})
        n = write_jsonl(os.path.join(ROOT, "data", "eval", "qa.jsonl"), rows)
        print("imported", n)
        return
    from openai import OpenAI

    client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
                    api_key=os.environ.get("LLM_API_KEY", "EMPTY"))
    chunks = [c for c in build_chunks("layout", 512, 50, True) if re.search(r"\d+\.\d+", c.text)]
    random.Random(0).shuffle(chunks)
    out = []
    for c in chunks[:args.n]:
        try:
            r = client.chat.completions.create(model=args.model, temperature=0.3,
                                               messages=[{"role": "user", "content": PROMPT.format(text=c.text[:800])}])
            j = json.loads(re.search(r"\{.*\}", r.choices[0].message.content, re.S).group())
            if j["must"] in c.text:
                out.append({**j, "doc_ids": c.doc_id, "page": c.page, "keep": "1", "evidence_text": c.text[:300]})
        except Exception as e:
            print("skip", e)
    with open(args.export, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["keep", "type", "query", "answer", "must", "doc_ids", "page", "evidence_text"])
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} drafts -> {args.export}（请人工校验，并手工补充 对比/汇总/无答案 题）")


if __name__ == "__main__":
    main()
