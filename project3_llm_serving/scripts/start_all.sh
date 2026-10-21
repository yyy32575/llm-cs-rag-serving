#!/usr/bin/env bash
# W12：三项目串联启动（单卡 24GB）
#   vLLM(AWQ 客服模型, :8000) + RAG 服务(:8100) + 降级小模型(:8001, 可选) + Redis + 网关(:9000)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
P2=../project2_fin_report_rag

redis-server --port 6379 --daemonize yes || echo "redis 未安装：网关自动退化为进程内缓存"
GPU_UTIL=0.70 bash scripts/start_vllm.sh awq models/qwen3-8b-cs-awq > logs/vllm.log 2>&1 &
if [[ -d models/qwen3-4b-awq ]]; then
  PORT=8001 GPU_UTIL=0.18 MAX_LEN=2048 bash scripts/start_vllm.sh awq models/qwen3-4b-awq > logs/vllm_fallback.log 2>&1 &
fi
(cd $P2 && LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=cs-model NO_LLM=1 \
   uvicorn app.api:app --host 0.0.0.0 --port 8100 > ../project3_llm_serving/logs/rag.log 2>&1 &)
until curl -s localhost:8000/v1/models >/dev/null; do sleep 5; done
until curl -s localhost:8100/healthz >/dev/null; do sleep 3; done
uvicorn gateway.main:app --host 0.0.0.0 --port 9000 > logs/gateway.log 2>&1 &
sleep 3
curl -s -X POST localhost:9000/chat -H 'Content-Type: application/json' -d '{"query":"双11能叠加优惠券吗"}'; echo
curl -s -X POST localhost:9000/chat -H 'Content-Type: application/json' -d '{"query":"宁川动力2025年三季度毛利率是多少"}'; echo
echo "all services up: gateway http://localhost:9000  stats http://localhost:9000/stats"
