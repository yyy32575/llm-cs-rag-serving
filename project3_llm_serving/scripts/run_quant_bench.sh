#!/usr/bin/env bash
# W9：量化对比实验一键脚本 —— 4 种精度 × (效果 + 单请求/并发速度 + 显存)
# 依赖：项目一已产出 outputs/dpo_beta0.1；GPU 24GB
set -euo pipefail
cd "$(dirname "$0")/.."
P1=../project1_ecom_cs_sft_dpo
BASE=${BASE_MODEL:-Qwen/Qwen3-8B}
mkdir -p models results

# 0) 合并 LoRA -> fp16 完整模型
[[ -d models/qwen3-8b-cs-sft-dpo ]] || python $P1/scripts/09_merge_lora.py --base $BASE \
    --adapter $P1/outputs/dpo_beta0.1 --out models/qwen3-8b-cs-sft-dpo
# 1) 量化
[[ -d models/qwen3-8b-cs-awq  ]] || python quant/quantize.py --method awq  --model models/qwen3-8b-cs-sft-dpo --out models/qwen3-8b-cs-awq
[[ -d models/qwen3-8b-cs-gptq ]] || python quant/quantize.py --method gptq --model models/qwen3-8b-cs-sft-dpo --out models/qwen3-8b-cs-gptq
[[ -d models/qwen3-8b-cs-w8a8 ]] || python quant/quantize.py --method int8 --model models/qwen3-8b-cs-sft-dpo --out models/qwen3-8b-cs-w8a8

run_one () {
  local mode=$1 path=$2 tag=$3
  MAX_LEN=4096 bash scripts/start_vllm.sh "$mode" "$path" > logs/vllm_$tag.log 2>&1 &
  local pid=$!
  until curl -s localhost:8000/v1/models >/dev/null; do sleep 5; done
  # 效果：项目一同一测试集（400 条，含困难子集）
  python $P1/scripts/07_evaluate.py --backend openai --model cs-model --base_url http://localhost:8000/v1 --exp_name quant_$tag
  cp $P1/results/metrics_quant_$tag.json results/
  # 速度：单请求（并发1）+ 并发阶梯；流式测 TTFT
  python bench/load_test.py --target openai --url http://localhost:8000/v1 --stream --concurrency 1,8,32 \
      --requests_per_level 96 --tag quant_$tag
  kill $pid; wait $pid 2>/dev/null || true; sleep 10
}
mkdir -p logs
run_one fp16 models/qwen3-8b-cs-sft-dpo fp16
run_one awq  models/qwen3-8b-cs-awq     awq_int4
run_one gptq models/qwen3-8b-cs-gptq    gptq_int4
run_one int8 models/qwen3-8b-cs-w8a8    int8_w8a8
python bench/summarize_quant.py
