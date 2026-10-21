"""Step 9: merge a LoRA adapter into the base model (for vLLM serving / AWQ & GPTQ quantization in project 3).

  python scripts/09_merge_lora.py --adapter outputs/dpo_beta0.1 --out ../project3_llm_serving/models/qwen3-8b-cs-sft-dpo
Merging runs in bf16 on CPU by default (needs ~20GB RAM for an 8B model).
"""
from __future__ import annotations

import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--sft_adapter", default=None,
                    help="如果 --adapter 是在 SFT adapter 之上继续训练的 DPO adapter，则二者本就是同一组权重，无需叠加")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map=args.device,
                                                 trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True).save_pretrained(args.out)
    print("merged model saved to", args.out)


if __name__ == "__main__":
    main()
