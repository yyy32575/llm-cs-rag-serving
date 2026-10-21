"""Step 10: collect all metrics_*.json / train_summary.json into markdown tables (直接贴到文档/README).

  python scripts/10_summarize.py
"""
from __future__ import annotations

import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    lines = ["# 实验结果汇总", "", "## 测试集指标", "",
             "| 实验 | 准确率 | 自动解决率 | 满意度 | 复读率 | 转人工率 | 平均长度 |", "|---|---|---|---|---|---|---|"]
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "metrics_*.json"))):
        m = load(p)
        o = m["overall"]
        lines.append(f"| {m['exp_name']} | {o['accuracy']:.1%} | {o['auto_resolve_rate']:.1%} | {o['satisfaction']:.2f} | "
                     f"{o['repetition_rate']:.1%} | {o['transfer_rate']:.1%} | {o['avg_len']} |")
    lines += ["", "## 训练开销", "", "| 实验 | r | target_modules | QLoRA | 可训练参数 | 峰值显存GB | 耗时min | train_loss | eval_loss |",
              "|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(glob.glob(os.path.join(ROOT, "outputs", "*", "train_summary.json"))):
        s = load(p)
        if "lora_r" in s:
            lines.append(f"| {s['exp_name']} | {s['lora_r']} | {s['target_modules']} | {s['qlora']} | "
                         f"{s['trainable_params'] / 1e6:.1f}M | {s['peak_mem_gb']} | {s['train_minutes']} | "
                         f"{s['train_loss']} | {s['eval_loss']} |")
    out = os.path.join(ROOT, "results", "SUMMARY.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
