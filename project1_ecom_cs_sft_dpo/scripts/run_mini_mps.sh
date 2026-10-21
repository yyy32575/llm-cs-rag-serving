#!/usr/bin/env bash
# 无 3090 时的“小模型真实复现实验”：Qwen3-0.6B + LoRA，Apple MPS / 单张小显卡 / CPU 均可跑（约 2.5 小时）
# 目的：在完全相同的数据、评测脚本下验证 rank / target_modules / DPO / 偏好噪声的趋势；
#       8B 正式实验用 run_all.sh（3090/4090）。两者数字不可混用，文档中分别标注。
# 已完成的训练（存在 train_summary.json）会自动跳过，可断点续跑。
set -euo pipefail
cd "$(dirname "$0")/.."
export KMP_DUPLICATE_LIB_OK=TRUE TOKENIZERS_PARALLELISM=false
M=${MINI_MODEL:-./models/Qwen3-0.6B}
STEPS=${STEPS:-300}
EV="--backend hf --model $M --max_new_tokens 128 --batch_size 16"
done_() { [[ -f outputs/$1/train_summary.json ]]; }

python scripts/07_evaluate.py $EV --exp_name mini_base_zeroshot
for r in 4 16 64; do
  done_ mini_sft_r$r || python scripts/03_train_sft.py --model_name_or_path $M --lora_r $r --target_modules q_proj,v_proj \
    --max_steps $STEPS --batch_size 4 --grad_accum 1 --max_len 512 --eval_steps 100 --exp_name mini_sft_r$r
  python scripts/07_evaluate.py $EV --adapter outputs/mini_sft_r$r --exp_name mini_sft_r$r
done
done_ mini_sft_r16_all || python scripts/03_train_sft.py --model_name_or_path $M --lora_r 16 --target_modules all-linear \
  --max_steps $STEPS --batch_size 4 --grad_accum 1 --max_len 512 --eval_steps 100 --exp_name mini_sft_r16_all
python scripts/07_evaluate.py $EV --adapter outputs/mini_sft_r16_all --exp_name mini_sft_r16_all

python scripts/05_build_pref_data.py --mode synthetic --n_pairs 2000
python scripts/05_build_pref_data.py --mode synthetic --n_pairs 2000 --noise_ratio 0.3 --tag noise30
for cfg in "0.1 train.jsonl dpo_beta0.1" "0.5 train.jsonl dpo_beta0.5" "0.1 train_noise30.jsonl dpo_noise30"; do
  set -- $cfg
  done_ mini_$3 || python scripts/06_train_dpo.py --model_name_or_path $M --sft_adapter outputs/mini_sft_r16 --beta $1 \
    --train_file data/dpo/$2 --limit 600 --epochs 1 --batch_size 2 --grad_accum 4 --lr 2e-5 \
    --max_len 512 --max_prompt_len 384 --exp_name mini_$3
  python scripts/07_evaluate.py $EV --adapter outputs/mini_$3 --exp_name mini_$3
done
python scripts/08_badcase_analysis.py --pred results/preds_mini_sft_r16.jsonl
python scripts/08_badcase_analysis.py --pred results/preds_mini_dpo_beta0.1.jsonl --compare results/preds_mini_sft_r16.jsonl
python scripts/10_summarize.py
