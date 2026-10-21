#!/usr/bin/env bash
# 一键跑完项目一全部实验（单卡 3090/4090 24GB，QLoRA）。总耗时约 10~14 小时，可分段执行。
set -euo pipefail
cd "$(dirname "$0")/.."
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-8B}
export PYTHONPATH=src

stage=${1:-all}

if [[ $stage == all || $stage == data ]]; then
  python scripts/01_build_sft_data.py --n_raw 12000 --n_test 400
  python scripts/02_clean_sft_data.py
  python scripts/07_evaluate.py --backend rule --exp_name baseline_rule
fi

if [[ $stage == all || $stage == base ]]; then
  python scripts/07_evaluate.py --backend vllm --exp_name base_zeroshot
fi

if [[ $stage == all || $stage == ablation ]]; then
  # W2 消融 1：rank
  for r in 4 16 64; do
    python scripts/03_train_sft.py --qlora --lora_r $r --target_modules q_proj,v_proj --exp_name sft_qlora_r$r
    python scripts/07_evaluate.py --backend vllm --adapter outputs/sft_qlora_r$r --exp_name sft_qlora_r$r
  done
  # W2 消融 2：target_modules
  python scripts/03_train_sft.py --qlora --lora_r 16 --target_modules all-linear --exp_name sft_qlora_r16_all
  python scripts/07_evaluate.py --backend vllm --adapter outputs/sft_qlora_r16_all --exp_name sft_qlora_r16_all
  # W2 消融 3：QLoRA vs LoRA(bf16)。24GB 单卡跑 8B bf16 LoRA 很紧，建议 2 卡 ZeRO-2 或 A100/A800
  if [[ ${RUN_BF16_LORA:-0} == 1 ]]; then
    accelerate launch --config_file configs/accelerate_ds_zero2.yaml scripts/03_train_sft.py \
      --lora_r 16 --target_modules q_proj,v_proj --deepspeed configs/ds_zero2.json --exp_name sft_lora_r16_bf16
    python scripts/07_evaluate.py --backend vllm --adapter outputs/sft_lora_r16_bf16 --exp_name sft_lora_r16_bf16
  fi
fi

if [[ $stage == all || $stage == dpo ]]; then
  # W3：SFT 模型采样候选 -> 自动打分 + 人工标注队列 -> 偏好对
  python scripts/04_gen_candidates.py --backend vllm --adapter outputs/sft_qlora_r16 --n_queries 2500 --n 3
  python scripts/05_build_pref_data.py --mode candidates --n_pairs 2000
  for b in 0.05 0.1 0.5; do
    python scripts/06_train_dpo.py --qlora --beta $b --exp_name dpo_beta$b
    python scripts/07_evaluate.py --backend vllm --adapter outputs/dpo_beta$b --exp_name dpo_beta$b
  done
  # 偏好数据噪声实验
  for nr in 0.1 0.2 0.3; do
    tag=noise${nr#0.}
    python scripts/05_build_pref_data.py --mode candidates --n_pairs 2000 --noise_ratio $nr --tag $tag
    python scripts/06_train_dpo.py --qlora --beta 0.1 --train_file data/dpo/train_$tag.jsonl --exp_name dpo_$tag
    python scripts/07_evaluate.py --backend vllm --adapter outputs/dpo_$tag --exp_name dpo_$tag
  done
fi

if [[ $stage == all || $stage == report ]]; then
  python scripts/08_badcase_analysis.py --pred results/preds_sft_qlora_r16.jsonl
  python scripts/08_badcase_analysis.py --pred results/preds_dpo_beta0.1.jsonl --compare results/preds_sft_qlora_r16.jsonl
  python scripts/10_summarize.py
fi
