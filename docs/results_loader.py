"""读取三个项目 results/ 下的实测结果，供文档填表；缺失时返回占位符，避免任何“编造数字”。"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P1 = os.path.join(ROOT, "project1_ecom_cs_sft_dpo")
P2 = os.path.join(ROOT, "project2_fin_report_rag")
P3 = os.path.join(ROOT, "project3_llm_serving")
TODO = "【待实测】"


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def pct(x, nd=1):
    return TODO if x is None else f"{x * 100:.{nd}f}%"


def num(x, nd=2):
    return TODO if x is None else f"{x:.{nd}f}"


# ---------------- project 1
def p1_metrics(exp):
    return load(os.path.join(P1, "results", f"metrics_{exp}.json"))


def p1_overall(exp, key):
    m = p1_metrics(exp)
    return None if not m else m["overall"].get(key)


def p1_subset(exp, sub):
    m = p1_metrics(exp)
    return None if not m else m["by_subset"].get(sub, {}).get("accuracy")


def p1_train(exp):
    return load(os.path.join(P1, "outputs", exp, "train_summary.json"))


def p1_clean():
    return load(os.path.join(P1, "data", "sft", "clean_stats.json")) or {}


def p1_build():
    return load(os.path.join(P1, "data", "raw", "build_stats.json")) or {}


def p1_badcase(exp):
    return load(os.path.join(P1, "results", f"badcase_{exp}.json"))


# ---------------- project 2
def p2(name):
    return load(os.path.join(P2, "results", name))


# ---------------- project 3
def p3(name):
    return load(os.path.join(P3, "results", name))


def p3_row(name, concurrency):
    """取某份压测结果中指定并发的那一行。"""
    lt = p3(name) or {}
    for r in lt.get("rows", []):
        if r.get("concurrency") == concurrency:
            return r
    return {}
