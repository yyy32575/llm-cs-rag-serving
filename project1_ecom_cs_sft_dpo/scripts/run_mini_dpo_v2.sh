#!/usr/bin/env bash
# 小模型 DPO 第二轮：修复第一轮“合成偏好对 + 较大学习率”导致的过优化
#   ① on-policy：用 SFT 模型自己采样的 3 个候选按规则打分成对（而不是规则扰动的合成差回复）
#   ② 学习率 2e-5 → 5e-6；③ RPO：DPO loss + chosen 的 NLL loss（rpo_alpha=1.0）
set -euo pipefail
cd "$(dirname "$0")/.."
export KMP_DUPLICATE_LIB_OK=TRUE TOKENIZERS_PARALLELISM=false
M=${MINI_MODEL:-./models/Qwen3-0.6B}
EV="--backend hf --model $M --max_new_tokens 128 --batch_size 16"
[[ -f data/dpo/candidates_mini.jsonl ]] || python scripts/04_gen_candidates.py --backend hf --model $M \
  --adapter outputs/mini_sft_r16 --n_queries 600 --n 3 --out data/dpo/candidates_mini.jsonl
python scripts/05_build_pref_data.py --mode candidates --inp data/dpo/candidates_mini.jsonl --n_pairs 2000 --tag onpolicy
run () {  # name lr rpo
  [[ -f outputs/mini_$1/train_summary.json ]] || python scripts/06_train_dpo.py --model_name_or_path $M \
    --sft_adapter outputs/mini_sft_r16 --beta 0.1 --train_file data/dpo/train_onpolicy.jsonl --val_file data/dpo/val_onpolicy.jsonl \
    --epochs 1 --batch_size 2 --grad_accum 4 --lr $2 ${3:+--rpo_alpha $3} --max_len 512 --max_prompt_len 384 --exp_name mini_$1
  python scripts/07_evaluate.py $EV --adapter outputs/mini_$1 --exp_name mini_$1
}
run dpo_onpolicy_lr5e-6 5e-6
run dpo_onpolicy_lr5e-6_rpo 5e-6 1.0
python scripts/08_badcase_analysis.py --pred results/preds_mini_dpo_onpolicy_lr5e-6_rpo.jsonl --compare results/preds_mini_sft_r16.jsonl
python scripts/10_summarize.py
