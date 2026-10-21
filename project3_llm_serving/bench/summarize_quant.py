"""Collect quantization results into the table 方案 | 模型体积 | 显存占用 | 推理速度 | 效果."""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS = [("FP16 原始", "fp16", "qwen3-8b-cs-sft-dpo"), ("AWQ INT4", "awq_int4", "qwen3-8b-cs-awq"),
        ("GPTQ INT4", "gptq_int4", "qwen3-8b-cs-gptq"), ("INT8 W8A8", "int8_w8a8", "qwen3-8b-cs-w8a8")]


def size_gb(d):
    p = os.path.join(ROOT, "models", d)
    return round(sum(os.path.getsize(f) for f in glob.glob(p + "/*.safetensors")) / 1024 ** 3, 2) if os.path.isdir(p) else None


def main():
    rows = []
    for name, tag, d in TAGS:
        lt = os.path.join(ROOT, "results", f"loadtest_quant_{tag}.json")
        mt = os.path.join(ROOT, "results", f"metrics_quant_{tag}.json")
        if not os.path.exists(lt):
            continue
        L = {r["concurrency"]: r for r in json.load(open(lt))["rows"]}
        M = json.load(open(mt))["overall"] if os.path.exists(mt) else {}
        single = L.get(1, {})
        rows.append({"方案": name, "模型体积GB": size_gb(d), "显存峰值MB": single.get("gpu_peak_mem_MB"),
                     "单请求tokens/s": round(single.get("out_tok_per_s", 0), 1), "TTFT_ms": single.get("TTFT_P50_ms"),
                     "32并发tokens/s": L.get(32, {}).get("out_tok_per_s"), "32并发P95_ms": L.get(32, {}).get("P95_ms"),
                     "准确率": M.get("accuracy"), "自动解决率": M.get("auto_resolve_rate")})
    cols = list(rows[0].keys()) if rows else []
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + \
         ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    open(os.path.join(ROOT, "results", "quant_summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
