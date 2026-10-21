#!/usr/bin/env bash
# W10：vLLM 参数对比（continuous batching 上限 / prefix caching / max_model_len），用于解释 P95 和吞吐的取舍
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:-models/qwen3-8b-cs-awq}
mkdir -p logs results
bench () {
  local tag=$1; shift
  EXTRA="$*" bash scripts/start_vllm.sh awq "$MODEL" > logs/vllm_$tag.log 2>&1 &
  local pid=$!
  until curl -s localhost:8000/v1/models >/dev/null; do sleep 5; done
  python bench/load_test.py --target openai --url http://localhost:8000/v1 --stream --concurrency 1,4,8,16,32 --tag $tag
  kill $pid; wait $pid 2>/dev/null || true; sleep 10
}
MAX_SEQS=64 bench awq_default
MAX_SEQS=8  bench awq_maxseqs8                                       # 近似“静态小 batch”
MAX_SEQS=64 bench awq_no_prefix_cache --no-enable-prefix-caching
MAX_SEQS=64 bench awq_chunked_prefill --enable-chunked-prefill --max-num-batched-tokens 2048
