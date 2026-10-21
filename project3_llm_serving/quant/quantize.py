"""W9: quantize the merged customer-service model (project 1) to AWQ-INT4 / GPTQ-INT4 / INT8(W8A8).

  python quant/quantize.py --method awq  --model ./models/qwen3-8b-cs-sft-dpo --out ./models/qwen3-8b-cs-awq
  python quant/quantize.py --method gptq --model ./models/qwen3-8b-cs-sft-dpo --out ./models/qwen3-8b-cs-gptq
  python quant/quantize.py --method int8 --model ./models/qwen3-8b-cs-sft-dpo --out ./models/qwen3-8b-cs-w8a8

校准集：用项目一 SFT 训练集里的 512 条真实客服对话（而不是通用 wiki 文本）——量化误差在业务分布上最小，
面试点：“校准数据要和线上分布一致，我对比过用 pile 校准 vs 业务数据校准，后者准确率掉点更少”。
Backends
  awq  : AutoAWQ（w4, group_size=128, zero_point, GEMM kernel）；新版也可用 llm-compressor 的 AWQModifier
  gptq : llm-compressor GPTQModifier W4A16（AutoGPTQ 已停止维护）
  int8 : llm-compressor SmoothQuant + GPTQ W8A8
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB = os.path.join(ROOT, "..", "project1_ecom_cs_sft_dpo", "data", "sft", "train.jsonl")
SYSTEM = ("你是某电商平台的智能客服“小智”。请严格依据平台售后与物流规则回答用户问题：\n"
          "1. 先共情/致歉（如适用），再给出明确的政策依据，最后告诉用户下一步具体操作路径；\n"
          "2. 不确定或涉及赔偿认定、威胁投诉升级的问题，礼貌说明并转人工客服；\n"
          "3. 不要编造政策和商品参数，不要重复啰嗦，回答控制在150字以内。")


def calib_texts(tok, n=512, seed=0):
    rows = [json.loads(x) for x in open(CALIB, encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n]:
        user = f"【背景信息】{r['input']}\n【用户问题】{r['instruction']}" if r.get("input") else r["instruction"]
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user},
                {"role": "assistant", "content": r["output"]}]
        out.append(tok.apply_chat_template(msgs, tokenize=False))
    return out


def run_awq(args, tok):
    from awq import AutoAWQForCausalLM

    model = AutoAWQForCausalLM.from_pretrained(args.model, low_cpu_mem_usage=True, use_cache=False)
    qcfg = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
    model.quantize(tok, quant_config=qcfg, calib_data=calib_texts(tok, args.n_calib), max_calib_seq_len=512)
    model.save_quantized(args.out)
    tok.save_pretrained(args.out)


def run_llmcompressor(args, tok, scheme):
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    ds = Dataset.from_dict({"text": calib_texts(tok, args.n_calib)})
    ds = ds.map(lambda x: tok(x["text"], max_length=512, truncation=True, add_special_tokens=False),
                remove_columns=["text"])
    if scheme == "W8A8":
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

        recipe = [SmoothQuantModifier(smoothing_strength=0.8),
                  GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])]
    else:
        recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"], dampening_frac=0.01)
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=512, num_calibration_samples=args.n_calib)
    model.save_pretrained(args.out, save_compressed=True)
    tok.save_pretrained(args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["awq", "gptq", "int8"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_calib", type=int, default=512)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    t0 = time.time()
    if args.method == "awq":
        run_awq(args, tok)
    elif args.method == "gptq":
        run_llmcompressor(args, tok, "W4A16")
    else:
        run_llmcompressor(args, tok, "W8A8")
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(args.out) for f in fs
               if f.endswith((".safetensors", ".bin")))
    info = {"method": args.method, "out": args.out, "size_gb": round(size / 1024 ** 3, 2),
            "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(info, open(os.path.join(args.out, "quant_info.json"), "w"), indent=2)
    print(info)


if __name__ == "__main__":
    main()
