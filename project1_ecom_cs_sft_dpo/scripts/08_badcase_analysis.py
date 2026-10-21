"""Step 8: bad-case analysis (失败案例分析) + model-vs-model comparison.

  python scripts/08_badcase_analysis.py --pred results/preds_sft_r16.jsonl
  python scripts/08_badcase_analysis.py --pred results/preds_dpo_beta0.1.jsonl --compare results/preds_sft_r16.jsonl

Categories (与文档“翻车案例”一一对应):
  repetition       复读机：同一句反复说（常见原因：训练数据大量模板化重复句式）
  outdated_policy  政策过时：回答了旧版政策数字（原因：数据里混入旧 SOP，需按版本清洗）
  fact_error       事实错误：命中禁止表述（如“拆封食品可以无理由退货”）
  over_transfer    过度转人工：能答的问题推给人工（DPO 重点修复对象）
  under_transfer   该转未转：赔偿/升级投诉没有转人工（风险最高）
  context_ignored  无视背景：签收天数超过政策窗口仍承诺可退
  multi_turn_drift 多轮失败：用户“突然变卦”或追问时没跟上上下文
  keypoint_missing 要点遗漏：缺少政策依据或操作路径
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ecs.clean import policy_violations  # noqa: E402
from ecs.io_utils import read_jsonl, write_json  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["under_transfer", "repetition", "outdated_policy", "fact_error", "context_ignored", "multi_turn_drift",
         "over_transfer", "keypoint_missing"]


def classify(p, item):
    if p["accurate"]:
        return None
    if item.get("transfer") and not p["transfer"]:
        return "under_transfer"
    if p["repetitive"]:
        return "repetition"
    if policy_violations(p["prediction"]):
        return "outdated_policy"
    if p["forbidden"]:
        return "fact_error"
    if p.get("subset") == "beyond_window":
        return "context_ignored"
    if str(p.get("subset", "")).startswith("multi_turn"):
        return "multi_turn_drift"
    if p["transfer"] and not item.get("transfer"):
        return "over_transfer"
    return "keypoint_missing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--compare", default=None, help="另一个模型的预测文件（如 SFT-only），输出修复/退化样本")
    ap.add_argument("--test_file", default=os.path.join(ROOT, "data", "test", "test.jsonl"))
    ap.add_argument("--k", type=int, default=3, help="每类展示样例数")
    args = ap.parse_args()

    items = {t["id"]: t for t in read_jsonl(args.test_file)}
    preds = read_jsonl(args.pred)
    name = os.path.basename(args.pred).replace("preds_", "").replace(".jsonl", "")
    buckets = defaultdict(list)
    for p in preds:
        c = classify(p, items[p["id"]])
        if c:
            buckets[c].append(p)

    total = len(preds)
    bad = sum(len(v) for v in buckets.values())
    lines = [f"# Bad-case report: {name}", "", f"- 测试样本: {total}，失败: {bad}（{bad / max(1, total):.1%}）", "",
             "| 类别 | 数量 | 占失败比例 |", "|---|---|---|"]
    for c in ORDER:
        if buckets.get(c):
            lines.append(f"| {c} | {len(buckets[c])} | {len(buckets[c]) / max(1, bad):.1%} |")
    for c in ORDER:
        for p in buckets.get(c, [])[:args.k]:
            lines += ["", f"### [{c}] {p['id']}", f"- 背景: {p['input']}", f"- 用户: {p['instruction']}",
                      f"- 模型: {p['prediction']}", f"- 参考: {p['reference']}"]

    summary = {"exp": name, "total": total, "failed": bad, "by_type": {c: len(buckets.get(c, [])) for c in ORDER}}
    if args.compare:
        other = {p["id"]: p for p in read_jsonl(args.compare)}
        fixed = [p for p in preds if p["accurate"] and not other[p["id"]]["accurate"]]
        broken = [p for p in preds if not p["accurate"] and other[p["id"]]["accurate"]]
        win = sum(p["satisfaction"] > other[p["id"]]["satisfaction"] for p in preds)
        lose = sum(p["satisfaction"] < other[p["id"]]["satisfaction"] for p in preds)
        summary.update({"compare_with": os.path.basename(args.compare), "fixed": len(fixed), "broken": len(broken),
                        "win": win, "tie": total - win - lose, "lose": lose,
                        "fixed_types": dict(Counter(classify(other[p["id"]], items[p["id"]]) for p in fixed))})
        lines += ["", f"## 对比 {os.path.basename(args.compare)}",
                  f"- 修复 {len(fixed)} 条，退化 {len(broken)} 条；满意度 win/tie/lose = {win}/{total - win - lose}/{lose}",
                  f"- 修复的原失败类型: {summary['fixed_types']}"]
    out_md = os.path.join(ROOT, "results", f"badcase_{name}.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    write_json(os.path.join(ROOT, "results", f"badcase_{name}.json"), summary)
    print("\n".join(lines[:16]))
    print("->", out_md)


if __name__ == "__main__":
    main()
