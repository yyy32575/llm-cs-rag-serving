"""Step 6: DPO on top of the SFT LoRA adapter (TRL DPOTrainer).

Default strategy (works with QLoRA on one 24GB card, no merge needed):
  load base (4bit) -> load SFT adapter twice: "policy" (trainable) + "reference" (frozen)
  -> DPOConfig(model_adapter_name="policy", ref_adapter_name="reference")
So the reference policy is exactly the SFT model, as DPO requires.

  python scripts/06_train_dpo.py --sft_adapter outputs/sft_qlora_r16 --beta 0.1 --exp_name dpo_beta0.1 --qlora
  python scripts/06_train_dpo.py --beta 0.05 --exp_name dpo_beta0.05 --qlora
  python scripts/06_train_dpo.py --train_file data/dpo/train_noise30.jsonl --exp_name dpo_noise30 --qlora
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", default=os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--sft_adapter", default=os.path.join(ROOT, "outputs", "sft_qlora_r16"))
    ap.add_argument("--train_file", default=os.path.join(ROOT, "data", "dpo", "train.jsonl"))
    ap.add_argument("--val_file", default=os.path.join(ROOT, "data", "dpo", "val.jsonl"))
    ap.add_argument("--exp_name", default="dpo_beta0.1")
    ap.add_argument("--output_root", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--beta", type=float, default=0.1, help="越大越保守（贴近参考模型），越小偏好越强但易过拟合")
    ap.add_argument("--loss_type", default="sigmoid", help="sigmoid(DPO) | ipo | hinge ...")
    ap.add_argument("--rpo_alpha", type=float, default=None, help="RPO：在 DPO loss 上叠加 chosen 的 SFT(NLL) loss，防止过优化/语言漂移")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--max_prompt_len", type=int, default=640)
    ap.add_argument("--report_to", default="none")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def build_ds(tok, path, limit=0):
    from datasets import Dataset

    rows = read_jsonl(path)
    if limit:
        rows = rows[:limit]
    return Dataset.from_list([{"prompt": render_prompt(tok, r), "chosen": r["chosen"], "rejected": r["rejected"]}
                              for r in rows])


def main():
    args = parse_args()
    set_seed(args.seed)
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from trl import DPOConfig, DPOTrainer

    out_dir = os.path.join(args.output_root, args.exp_name)
    tok = AutoTokenizer.from_pretrained(args.sft_adapter, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    kw = dict(torch_dtype=dtype, trust_remote_code=True,
              device_map={"": 0} if torch.cuda.is_available() else ({"": "mps"} if torch.backends.mps.is_available() else None))
    if args.qlora:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_use_double_quant=True,
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **kw)
    base.config.use_cache = False
    if args.qlora:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=True, adapter_name="policy")
    model.load_adapter(args.sft_adapter, adapter_name="reference")
    model.set_adapter("policy")

    train_ds, val_ds = build_ds(tok, args.train_file, args.limit), build_ds(tok, args.val_file)
    cfg = dict(output_dir=out_dir, beta=args.beta, loss_type=args.loss_type, learning_rate=args.lr, rpo_alpha=args.rpo_alpha,
               num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
               per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
               lr_scheduler_type="cosine", warmup_ratio=0.1, logging_steps=10, eval_strategy="steps",
               eval_steps=50, save_strategy="epoch", save_total_limit=1, bf16=use_bf16,
               fp16=torch.cuda.is_available() and not use_bf16, use_cpu=not torch.cuda.is_available() and not torch.backends.mps.is_available(),
               gradient_checkpointing=True,
               gradient_checkpointing_kwargs={"use_reentrant": False}, max_length=args.max_len,
               max_prompt_length=args.max_prompt_len, model_adapter_name="policy", ref_adapter_name="reference",
               report_to=args.report_to, seed=args.seed, run_name=args.exp_name, remove_unused_columns=False)
    dpo_args = DPOConfig(**filter_kwargs(DPOConfig, cfg))

    history = []

    class Rec(TrainerCallback):
        def on_log(self, _a, state, control, logs=None, **k):
            if logs:
                history.append({"step": state.global_step,
                                **{k2: float(v) for k2, v in logs.items() if isinstance(v, (int, float))}})

    tkw = dict(model=model, ref_model=None, args=dpo_args, train_dataset=train_ds, eval_dataset=val_ds,
               callbacks=[Rec()])
    params = inspect.signature(DPOTrainer.__init__).parameters
    tkw["processing_class" if "processing_class" in params else "tokenizer"] = tok
    trainer = DPOTrainer(**tkw)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = trainer.train()
    ev = trainer.evaluate()
    model.save_pretrained(out_dir, selected_adapters=["policy"])
    # PEFT saves non-default adapters into a sub-folder; flatten so that eval scripts can load it directly
    sub = os.path.join(out_dir, "policy")
    if os.path.isdir(sub):
        for f in os.listdir(sub):
            os.replace(os.path.join(sub, f), os.path.join(out_dir, f))
        os.rmdir(sub)
    tok.save_pretrained(out_dir)
    summary = {"exp_name": args.exp_name, "beta": args.beta, "loss_type": args.loss_type, "lr": args.lr,
               "rpo_alpha": args.rpo_alpha,
               "train_file": os.path.basename(args.train_file), "pairs": len(train_ds),
               "train_loss": round(out.training_loss, 4),
               "eval_rewards_accuracy": ev.get("eval_rewards/accuracies"),
               "eval_rewards_margin": ev.get("eval_rewards/margins"),
               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if torch.cuda.is_available() else 0,
               "train_minutes": round((time.time() - t0) / 60, 1)}
    write_json(os.path.join(out_dir, "train_summary.json"), summary)
    write_json(os.path.join(out_dir, "loss_history.json"), history)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
