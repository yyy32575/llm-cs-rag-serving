#!/usr/bin/env bash
# 启动 vLLM OpenAI 兼容服务（单卡 3090/4090 24GB）
#   bash scripts/start_vllm.sh fp16   ./models/qwen3-8b-cs-sft-dpo
#   bash scripts/start_vllm.sh awq    ./models/qwen3-8b-cs-awq
#   bash scripts/start_vllm.sh gptq   ./models/qwen3-8b-cs-gptq
#   bash scripts/start_vllm.sh int8   ./models/qwen3-8b-cs-w8a8     (llm-compressor W8A8)
set -euo pipefail
MODE=${1:-awq}
MODEL=${2:-./models/qwen3-8b-cs-awq}
PORT=${PORT:-8000}
MAX_LEN=${MAX_LEN:-4096}
GPU_UTIL=${GPU_UTIL:-0.90}
MAX_SEQS=${MAX_SEQS:-64}
EXTRA=${EXTRA:-}

case $MODE in
  fp16) QUANT="" ; DTYPE="--dtype bfloat16" ;;     # 8B bf16 权重 ~16.4GB，24GB 卡只剩 ~4GB KV Cache
  awq)  QUANT="--quantization awq_marlin" ; DTYPE="--dtype float16" ;;
  gptq) QUANT="--quantization gptq_marlin" ; DTYPE="--dtype float16" ;;
  int8) QUANT="" ; DTYPE="--dtype bfloat16" ;;     # compressed-tensors 自动识别
  *) echo "unknown mode $MODE"; exit 1 ;;
esac

# 关键参数说明（面试会问）：
#  --max-model-len        : 上下文上限，直接决定单序列 KV Cache 上限；客服场景 4096 足够
#  --gpu-memory-utilization: vLLM 预分配显存比例（权重 + KV Cache 块池）
#  --max-num-seqs         : continuous batching 同时调度的最大序列数
#  --enable-prefix-caching: 相同 system prompt 的 KV 复用，客服场景 prefill 大幅减少
#  --served-model-name    : 对外模型名，网关用它路由
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name cs-model \
  --host 0.0.0.0 --port "$PORT" $QUANT $DTYPE \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" --max-num-seqs "$MAX_SEQS" \
  --enable-prefix-caching --disable-log-requests $EXTRA
