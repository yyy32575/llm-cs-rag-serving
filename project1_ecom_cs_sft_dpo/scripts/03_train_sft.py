"""Step 3: SFT with TRL SFTTrainer + PEFT LoRA / QLoRA.

Single 3090/4090 (24GB):
  python scripts/03_train_sft.py --qlora --lora_r 16 --target_modules q_proj,v_proj --exp_name sft_qlora_r16
LoRA bf16 (needs ~2x24GB, or a single 40GB+ card):
  accelerate launch --config_file configs/accelerate_ds_zero2.yaml scripts/03_train_sft.py --lora_r 16 --exp_name sft_lora_r16

Outputs (outputs/<exp_name>/):
  adapter_model.safetensors / tokenizer   -> LoRA adapter
  loss_history.json / loss_curve.png      -> 训练曲线（面试讲 loss 曲线用）
  train_summary.json                      -> 显存峰值、耗时、可训练参数量、最终 loss
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ecs.io_utils import filter_kwargs, read_jsonl, set_seed, write_json  # noqa: E402
from ecs.prompts import render_prompt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", default=os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--train_file", default=os.path.join(ROOT, "data", "sft", "train.jsonl"))
    ap.add_argument("--val_file", default=os.path.join(ROOT, "data", "sft", "val.jsonl"))
    ap.add_argument("--exp_name", default="sft_qlora_r16")
    ap.add_argument("--output_root", default=os.path.join(ROOT, "outputs"))
    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=None, help="默认 2*r")
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default="q_proj,v_proj",
                    help="q_proj,v_proj | all-linear | q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--qlora", action="store_true", help="4bit NF4 量化底座 + LoRA")
    # optimization
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪；loss 不降/NaN 时首先检查")
    ap.add_argument("--no_gradient_checkpointing", action="store_true")
    ap.add_argument("--flash_attn", action="store_true")
    ap.add_argument("--deepspeed", default=None)
    ap.add_argument("--report_to", default="none")
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: 只用前 N 条")
    return ap.parse_args()


def build_dataset(tokenizer, path, limit=0):
    from datasets import Dataset

    rows = read_jsonl(path)
    if limit:
        rows = rows[:limit]
    data = [{"prompt": render_prompt(tokenizer, r), "completion": r["output"]} for r in rows]
    return Dataset.from_list(data)


def main():
    args = parse_args()
    set_seed(args.seed)
    import torch
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    out_dir = os.path.join(args.output_root, args.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    train_ds = build_dataset(tok, args.train_file, args.limit)
    val_ds = build_dataset(tok, args.val_file, args.limit // 10 if args.limit else 0)
    print(train_ds[0]["prompt"] + train_ds[0]["completion"])

    model_kwargs = dict(torch_dtype=dtype, trust_remote_code=True)
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    if args.qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        if not args.deepspeed:
            model_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", 0))}
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.config.use_cache = False
    if args.qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=not args.no_gradient_checkpointing)

    tm = args.target_modules
    target_modules = "all-linear" if tm == "all-linear" else [x.strip() for x in tm.split(",") if x.strip()]
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha or 2 * args.lora_r,
                             lora_dropout=args.lora_dropout, target_modules=target_modules,
                             bias="none", task_type="CAUSAL_LM")

    cfg = dict(
        output_dir=out_dir, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio, logging_steps=10, eval_strategy="steps", evaluation_strategy="steps",
        eval_steps=args.eval_steps, save_strategy="epoch", save_total_limit=2, bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16, use_cpu=not torch.cuda.is_available() and not torch.backends.mps.is_available(),
        max_grad_norm=args.max_grad_norm, gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_len, max_seq_length=args.max_len, packing=False, completion_only_loss=True,
        report_to=args.report_to, seed=args.seed, deepspeed=args.deepspeed, run_name=args.exp_name,
        dataset_num_proc=4,
    )
    sft_args = SFTConfig(**filter_kwargs(SFTConfig, cfg))

    history = []

    class LossRecorder(TrainerCallback):
        def on_log(self, _args, state, control, logs=None, **kw):
            if logs:
                history.append({"step": state.global_step, **{k: float(v) for k, v in logs.items()
                                                               if isinstance(v, (int, float))}})

    tr_kwargs = dict(model=model, args=sft_args, train_dataset=train_ds, eval_dataset=val_ds,
                     peft_config=peft_config, callbacks=[LossRecorder()])
    params = inspect.signature(SFTTrainer.__init__).parameters
    tr_kwargs["processing_class" if "processing_class" in params else "tokenizer"] = tok
    trainer = SFTTrainer(**tr_kwargs)
    trainer.model.print_trainable_parameters()
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_out = trainer.train()
    minutes = (time.time() - t0) / 60
    eval_out = trainer.evaluate()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

    peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
    summary = {
        "exp_name": args.exp_name, "base_model": args.model_name_or_path, "qlora": args.qlora,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha or 2 * args.lora_r, "target_modules": tm,
        "lr": args.lr, "epochs": args.epochs, "train_samples": len(train_ds),
        "trainable_params": trainable, "train_loss": round(train_out.training_loss, 4),
        "eval_loss": round(eval_out.get("eval_loss", float("nan")), 4),
        "peak_mem_gb": round(peak, 2), "train_minutes": round(minutes, 1),
    }
    write_json(os.path.join(out_dir, "train_summary.json"), summary)
    write_json(os.path.join(out_dir, "loss_history.json"), history)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tr = [(h["step"], h["loss"]) for h in history if "loss" in h]
        ev = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
        plt.figure(figsize=(6, 4))
        if tr:
            plt.plot(*zip(*tr), label="train_loss")
        if ev:
            plt.plot(*zip(*ev), label="eval_loss", marker="o")
        plt.xlabel("step"), plt.ylabel("loss"), plt.title(args.exp_name), plt.legend(), plt.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    except Exception as e:  # pragma: no cover
        print("plot skipped:", e)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
