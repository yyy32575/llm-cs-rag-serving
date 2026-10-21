"""KV Cache memory estimator (面试高频：“KV Cache 显存怎么估？”).

通用公式（MHA）：  bytes = 2(K,V) × num_layers × hidden_size × seq_len × batch × bytes_per_elem
GQA 修正（Qwen3/Llama3 等）：hidden_size 换成 num_kv_heads × head_dim —— Qwen3-8B 只有 8 个 KV head，
  比按 hidden_size 估算小 4 倍。面试时先说通用公式，再主动补一句 GQA，非常加分。

  python bench/kv_cache_calc.py --model qwen3-8b --seq_len 2048 --gpu_mem_gb 24 --weights awq
  python bench/kv_cache_calc.py --config /path/to/model/config.json
"""
from __future__ import annotations

import argparse
import json

PRESETS = {  # from official config.json
    "qwen3-8b": dict(num_layers=36, hidden=4096, heads=32, kv_heads=8, head_dim=128, params_b=8.19),
    "qwen3-4b": dict(num_layers=36, hidden=2560, heads=32, kv_heads=8, head_dim=128, params_b=4.02),
    "qwen3-14b": dict(num_layers=40, hidden=5120, heads=40, kv_heads=8, head_dim=128, params_b=14.8),
    "qwen2.5-7b": dict(num_layers=28, hidden=3584, heads=28, kv_heads=4, head_dim=128, params_b=7.62),
}
WEIGHT_BYTES = {"fp16": 2.0, "bf16": 2.0, "int8": 1.0, "awq": 0.5625, "gptq": 0.5625}  # int4 + scales/zeros ~ 4.5bit


def from_config(path):
    c = json.load(open(path))
    heads = c["num_attention_heads"]
    return dict(num_layers=c["num_hidden_layers"], hidden=c["hidden_size"], heads=heads,
                kv_heads=c.get("num_key_value_heads", heads), head_dim=c.get("head_dim", c["hidden_size"] // heads),
                params_b=None)


def kv_bytes_per_token(m, kv_dtype_bytes=2, gqa=True):
    width = m["kv_heads"] * m["head_dim"] if gqa else m["hidden"]
    return 2 * m["num_layers"] * width * kv_dtype_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b", choices=list(PRESETS))
    ap.add_argument("--config", default=None)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--kv_dtype_bytes", type=float, default=2, help="fp16=2, fp8 kv cache=1")
    ap.add_argument("--gpu_mem_gb", type=float, default=24)
    ap.add_argument("--gpu_util", type=float, default=0.90, help="vLLM --gpu-memory-utilization")
    ap.add_argument("--weights", default="fp16", choices=list(WEIGHT_BYTES))
    ap.add_argument("--activation_reserve_gb", type=float, default=1.5, help="激活/CUDA graph/碎片预留")
    a = ap.parse_args()
    m = from_config(a.config) if a.config else PRESETS[a.model]
    per_tok = kv_bytes_per_token(m, a.kv_dtype_bytes, gqa=True)
    per_tok_mha = kv_bytes_per_token(m, a.kv_dtype_bytes, gqa=False)
    total = per_tok * a.seq_len * a.batch
    print(f"模型: {a.config or a.model}  layers={m['num_layers']} hidden={m['hidden']} kv_heads={m['kv_heads']} "
          f"head_dim={m['head_dim']}")
    print(f"每 token KV: GQA {per_tok / 1024:.1f} KB  | 若按 MHA(hidden) 公式 {per_tok_mha / 1024:.1f} KB "
          f"（高估 {per_tok_mha / per_tok:.0f}x）")
    print(f"seq_len={a.seq_len}, batch={a.batch}: KV Cache = {total / 1024 ** 3:.2f} GB")
    if m.get("params_b"):
        w = m["params_b"] * WEIGHT_BYTES[a.weights]
        budget = a.gpu_mem_gb * a.gpu_util - w - a.activation_reserve_gb
        max_tokens = budget * 1024 ** 3 / per_tok
        print(f"权重({a.weights}) ≈ {w:.1f} GB；{a.gpu_mem_gb}GB×{a.gpu_util} 可用于 KV ≈ {budget:.1f} GB "
              f"≈ {max_tokens / 1e3:.0f}K tokens ≈ 最多 {int(max_tokens // a.seq_len)} 条 {a.seq_len} 长度的并发序列")


if __name__ == "__main__":
    main()
