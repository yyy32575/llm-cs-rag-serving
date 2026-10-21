# 项目三：推理部署 —— vLLM + 量化 + FastAPI 网关（缓存/限流/熔断降级）+ 压测与故障演练

> 业务背景：把项目一的客服模型和项目二的研报 RAG 串成一个在线服务。日均 5000+ 请求、双 11 高峰 3 倍，
> 要求 P95 < 2s、32 并发不崩，GPU 扛不住时有降级方案，热门问题直接走缓存。

## 架构

```
用户 ──> FastAPI 网关(:9000)
          ├─ 单用户限流(令牌桶) ──> 缓存(Redis/本地LRU，热点预热) ──命中──> 直接返回(<1ms)
          ├─ 全局限流(只对打到 GPU 的请求计数，超限 429)
          ├─ 意图路由 ──客服──> vLLM 客服模型(AWQ-INT4, :8000)
          │          └─研报──> RAG 检索服务(:8100, 项目二) ──约束校验不过──> 直接拒答
          │                                        └─证据──> vLLM 生成
          ├─ 超时 + 熔断(错误率>30% 或 P95>SLO) ──> 降级小模型(:8001, 短 max_tokens) ──> 兜底话术+转人工
          └─ 日志(JSONL: input/output tokens、latency、status) + /stats(60s 滚动) + /metrics(Prometheus)
```

## 目录

```
project3_llm_serving/
├── gateway/        main.py(网关) cache.py ratelimit.py router.py degrade.py(熔断) metrics.py llm_client.py config.py
├── quant/quantize.py         # AWQ(AutoAWQ) / GPTQ W4A16 / INT8 W8A8(llm-compressor)，业务数据校准
├── bench/
│   ├── load_test.py          # 并发阶梯压测（1/4/8/16/32），QPS、P50/P95/P99、TTFT、tokens/s、显存
│   ├── kv_cache_calc.py      # KV Cache 显存估算（含 GQA 修正）
│   ├── chaos_drill.py        # 故障演练：上游报错 / 慢解码 / 3 倍突发 / 热点
│   ├── summarize_quant.py    # 量化对比表
│   └── locustfile.py
├── mock/mock_llm_server.py   # 无 GPU 时的 OpenAI 兼容 mock（参数化延迟 + 故障注入），只用于验证网关逻辑
├── scripts/  start_vllm.sh  run_quant_bench.sh  run_vllm_param_bench.sh  start_all.sh
└── configs/  gateway.yaml  prometheus.yml
```

## 快速开始（GPU，Linux）

```bash
pip install -r requirements.txt
# 1) 合并项目一的 LoRA 并量化
python ../project1_ecom_cs_sft_dpo/scripts/09_merge_lora.py --adapter ../project1_ecom_cs_sft_dpo/outputs/dpo_beta0.1 \
       --out models/qwen3-8b-cs-sft-dpo
bash scripts/run_quant_bench.sh            # FP16/AWQ/GPTQ/INT8 四组：效果(项目一测试集) + 速度 + 显存
# 2) 服务化 + 压测
bash scripts/start_all.sh                  # vLLM + RAG + Redis + 网关
python bench/load_test.py --target openai --url http://localhost:8000/v1 --stream --tag vllm_awq
python bench/load_test.py --target gateway --url http://localhost:9000 --hot_ratio 0.3 --tag gateway
bash scripts/run_vllm_param_bench.sh       # max-num-seqs / prefix caching / chunked prefill 对比
# 3) 故障演练（真实 vLLM 下：OOM 可用超长 max_tokens + 高并发复现；慢请求可用 --max-num-seqs 1 复现）
python bench/chaos_drill.py --primary_admin http://localhost:8000
```

## 无 GPU 验证网关

```bash
python mock/mock_llm_server.py --port 8000 --model_name cs-model --step_ms 12 --mean_out 60 &
python mock/mock_llm_server.py --port 8001 --model_name fallback-model --step_ms 8 --mean_out 40 &
REDIS_URL= uvicorn gateway.main:app --port 9000 &
python bench/chaos_drill.py
```

> mock 的延迟是参数化模拟，**不代表 GPU 真实性能**；mock 下的数字只能用来证明“熔断/降级/限流/缓存逻辑按设计工作”，
> 简历和面试里的 QPS、P95、量化对比必须用真实 vLLM 实测数据。
