"""Step 1: parse all PDFs with the naive and the layout-aware parser, and *measure* parsing quality.

Parse-quality metric = 证据存活率：评测集每条金标证据（如 “24.8”）在对应文档解析文本中能否被完整找到。
双栏串行、表格打平、数字被换行截断都会让证据“死掉”，后面检索再强也召回不了。

  python scripts/01_parse_pdfs.py --pdf_dir data/pdfs
Outputs: data/parsed/{naive,layout}.jsonl, results/parse_quality.json
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finrag.pdf_parser import LayoutParser, NaiveParser  # noqa: E402
from finrag.utils import norm_text, read_jsonl, write_json, write_jsonl  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evidence_survival(blocks, qa):
    by_doc = defaultdict(str)
    for b in blocks:
        by_doc[b["doc_id"]] += norm_text(b["text"]) + "\n"
    tot = ok = 0
    by_type = defaultdict(lambda: [0, 0])
    for q in qa:
        for ev in q["evidence"]:
            tot += 1
            hit = any(norm_text(ev["must"]) in by_doc.get(d, "") for d in ev["doc_ids"])
            ok += hit
            by_type[q["type"]][0] += hit
            by_type[q["type"]][1] += 1
    return {"overall": round(ok / max(1, tot), 4), **{k: round(a / b, 4) for k, (a, b) in by_type.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", default=os.path.join(ROOT, "data", "pdfs"))
    ap.add_argument("--qa", default=os.path.join(ROOT, "data", "eval", "qa.jsonl"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "data", "parsed"))
    args = ap.parse_args()

    pdfs = sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf")))
    qa = read_jsonl(args.qa) if os.path.exists(args.qa) else []
    report = {}
    for parser in (NaiveParser(), LayoutParser()):
        t0 = time.time()
        blocks = []
        for p in pdfs:
            blocks += [b.to_dict() for b in parser.parse(p)]
        sec = time.time() - t0
        write_jsonl(os.path.join(args.out_dir, f"{parser.name}.jsonl"), blocks)
        report[parser.name] = {
            "pdfs": len(pdfs), "blocks": len(blocks), "table_blocks": sum(b["type"] == "table" for b in blocks),
            "seconds": round(sec, 2), "sec_per_pdf": round(sec / max(1, len(pdfs)), 3),
            "evidence_survival": evidence_survival(blocks, qa) if qa else None,
            "stats": dict(getattr(parser, "stats", {})),
        }
        print(parser.name, report[parser.name])
    write_json(os.path.join(ROOT, "results", "parse_quality.json"), report)


if __name__ == "__main__":
    main()
