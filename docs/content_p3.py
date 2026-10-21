"""项目三：推理部署。"""
from results_loader import TODO, p3, p3_row, pct

TAGS = ["fp16", "awq_int4", "gptq_int4", "int8_w8a8"]
LT32 = p3_row("loadtest_vllm_awq.json", 32)
FP32 = p3_row("loadtest_vllm_fp16.json", 32)
GW32 = p3_row("loadtest_gateway_vllm_hot30.json", 32)
HIT = (p3("loadtest_gateway_vllm_hot30.json") or {}).get("cache_hit_rate")


def _q(tag, c):
    return p3_row(f"loadtest_quant_{tag}.json", c)


def _acc(tag):
    return (p3(f"metrics_quant_{tag}.json") or {}).get("overall", {}).get("accuracy")


def _size(tag):
    return (p3(f"loadtest_quant_{tag}.json") or {}).get("model_size_gb")


def _note(tag):
    if tag == "fp16":
        return "基准：权重 16.4GB，32G 卡留给 KV Cache 的只有约 12.4GB（≈44 条 2K 序列）"
    s1, s32 = _q("fp16", 1)["out_tok_per_s"], _q("fp16", 32)["out_tok_per_s"]
    return (f"体积 -{(1 - _size(tag) / _size('fp16')) * 100:.0f}%，单请求 {_q(tag, 1)['out_tok_per_s'] / s1:.2f}×，"
            f"32 并发 {_q(tag, 32)['out_tok_per_s'] / s32:.2f}×，掉点 {(_acc('fp16') - _acc(tag)) * 100:.1f} 个点")


def _quant_row(name, tag):
    return [name, _size(tag), _q(tag, 1).get("gpu_peak_mem_MB") or TODO,
            _q(tag, 1).get("out_tok_per_s") or TODO, _q(tag, 32).get("out_tok_per_s") or TODO,
            pct(_acc(tag)), _note(tag)]


def content(d):
    ch = p3("chaos_mock.json") or {}
    up = ch.get("1_upstream_error", {})
    d.part("项目三  推理部署 —— vLLM + 量化 + FastAPI 网关 + 压测与故障演练")
    d.h1("一、项目介绍")
    d.h2("1.1 项目一句话简介")
    d.box("把项目一的客服模型和项目二的研报 RAG 串成一个能扛并发的在线服务：vLLM 部署 AWQ-INT4 量化模型，FastAPI 网关负责意图路由、Redis 热点缓存、"
          "令牌桶限流、超时熔断与降级、日志监控；用并发阶梯压测、KV Cache 估算和故障演练给出 QPS/P95/显存与止血 SOP。")
    d.h2("背景：为什么一定要有部署项目")
    d.ul(["只在 Notebook 里跑，面试官会觉得你只是“调参选手”。一问“服务 QPS 多少？P95 多少？OOM 了怎么处理？”十个人里八个答不上来。",
          "场景接着客服来：日均 5000+ 请求，双 11 高峰 3 倍；要求单次响应 P95 < 2 秒、32 并发不崩；GPU 扛不住要有降级方案；"
          "热门问题（“双 11 能叠加优惠券吗”）直接走缓存不过模型。",
          "做完这个项目，简历就不只是“会训练”“会 RAG”，而是“能跑通训练→检索→部署全链路”。"])
    d.h2("推理框架与技术选型")
    d.table(["组件", "选择", "备选 / 理由"],
            [["推理引擎", "vLLM（PagedAttention、continuous batching、OpenAI 兼容 API、prefix caching）", "SGLang（进阶 A/B）、LMDeploy（量化+部署一体，benchmark 对照）"],
             ["量化", "AWQ-INT4（AutoAWQ / llm-compressor），业务数据校准", "GPTQ W4A16（llm-compressor）、INT8 W8A8（SmoothQuant）"],
             ["服务框架", "FastAPI 网关 + vLLM OpenAI-compatible server", "—"],
             ["缓存", "Redis（多副本共享），不可用时自动退化为进程内 LRU", "—"],
             ["限流", "令牌桶：单用户防刷 + 全局（只对打到 GPU 的请求计数）", "Redis + Lua 分布式令牌桶（代码内附）"],
             ["降级", "熔断器（错误率 / P95 超 SLO）→ 小模型 + 短 max_tokens → 兜底话术 + 转人工", "—"],
             ["压测", "自写异步压测（并发阶梯、TTFT、tokens/s、显存采样）", "locust（附 locustfile）"],
             ["监控", "JSONL 请求日志 + /stats 滚动指标 + Prometheus /metrics（含 vLLM 自带指标）", "Grafana"],
             ["模型", "项目一 Qwen3-8B SFT+DPO 合并后 AWQ-INT4", "Qwen3-14B-AWQ（显存允许时）"]],
            widths=[1, 3.3, 2.6])
    d.p("整体路线：vLLM 跑通 → 量化对比 → FastAPI 套一层做工程化 → 压测 → 故障演练。产出一份量化对比表 + 一份压测报告 + 一份故障处理文档。")

    d.h2("1.2 简历直接可用的项目描述（三种不同表述，已降重）")
    d.box("性能数字（QPS、P95、tokens/s、显存、掉点）全部来自 5090 单卡的实测：量化对比用 scripts/run_quant_bench.sh，"
          "并发压测用 bench/load_test.py（1/4/8/16/32 阶梯，TTFT、tokens/s、GPU 显存采样），"
          "准确率掉点用项目一同一份 400 条测试集测。KV Cache 容量是确定性计算，可直接引用；"
          "mock 环境的数字只用于验证网关逻辑，不要写进简历的性能指标里。", title="使用说明")
    d.resume("版本 A｜大模型推理服务部署与性能优化（vLLM + 量化）", [
        ("项目定位：", ["将微调后的客服模型与研报 RAG 串联为在线服务，在单卡 32GB 预算下满足 32 并发、P95 < 2s 的 SLO，并具备高峰降级能力"]),
        ("核心方案：", ["- 基于 vLLM 部署 Qwen3-8B 客服模型，对比 FP16 / AWQ-INT4 / GPTQ-INT4 / INT8 四种精度的显存、吞吐、TTFT 与业务准确率，量化校准使用业务对话",
                       "- 搭建 FastAPI 网关：意图路由（客服→模型 / 研报→RAG）、Redis 热点缓存与预热、令牌桶限流、超时熔断与小模型降级、全链路 token/延迟日志",
                       "- 自研异步压测工具完成 1/4/8/16/32 并发阶梯压测，对比 max-num-seqs、prefix caching、chunked prefill 等调度参数",
                       "- 基于 GQA 修正的 KV Cache 公式做容量估算，并完成 OOM、延迟飙升、流量突增、热点集中四类故障演练"]),
        ("实验与结果：", [f"- AWQ-INT4 相比 FP16：单请求 {_q('fp16', 1)['out_tok_per_s']}→{_q('awq_int4', 1)['out_tok_per_s']} tokens/s"
                         f"（{_q('awq_int4', 1)['out_tok_per_s'] / _q('fp16', 1)['out_tok_per_s']:.1f}×）、"
                         f"32 并发 {_q('fp16', 32)['out_tok_per_s']}→{_q('awq_int4', 32)['out_tok_per_s']} tokens/s"
                         f"（{_q('awq_int4', 32)['out_tok_per_s'] / _q('fp16', 32)['out_tok_per_s']:.1f}×），"
                         f"显存峰值 {_q('fp16', 32)['gpu_peak_mem_MB'] / 1000:.1f}GB→{_q('awq_int4', 32)['gpu_peak_mem_MB'] / 1000:.1f}GB，"
                         f"客服测试集准确率 {pct(_acc('fp16'))}→{pct(_acc('awq_int4'))}"
                         f"（掉点 {(_acc('fp16') - _acc('awq_int4')) * 100:.1f} 个点）",
                         f"- 32 并发下 QPS {LT32['QPS']}、P95 {LT32['P95_ms']}ms；热点问题缓存命中后延迟 <1ms，"
                         f"缓存命中率 {pct(HIT, 0)}（叠加缓存后 32 并发 QPS {GW32['QPS']}）",
                         "- 估算 32GB 卡上 AWQ 权重约 4.6GB，可为 KV Cache 留约 24.2GB（约 17.6 万 token，≈86 条 2K 序列），FP16 仅能支撑约 44 条"]),
        ("技术栈：", ["vLLM、AutoAWQ/llm-compressor、FastAPI、Redis、httpx、Prometheus、Python asyncio"])])
    d.resume("版本 B｜LLM 在线服务网关与稳定性工程", [
        ("项目定位：", ["为客服与研报问答提供统一推理入口，重点解决“高峰扛不住、慢请求拖垮整体、热点重复计算、故障无预案”四类线上问题"]),
        ("核心方案：", ["- 网关请求链路：单用户限流 → 缓存（归一化 query 为 key，个人订单上下文请求不缓存）→ 全局令牌桶 → 路由 → 带超时的模型调用",
                       "- 实现 CLOSED/OPEN/HALF_OPEN 熔断状态机：窗口错误率 >30% 或 P95 超 SLO 时切换降级小模型并缩短 max_tokens，半开探测自动恢复",
                       "- 研报路由接入 RAG 服务的约束校验结果，证据不足时网关直接拒答、不占用 GPU",
                       "- 提供 OpenAI 兼容透传接口（含流式）、/stats 滚动指标与 Prometheus 指标，并编写故障注入 mock 与演练脚本"]),
        ("实验与结果：", [f"- 故障演练（mock 实测）：上游 30% 报错时熔断触发，请求全部由降级模型兜底，成功率保持 {pct(up.get('success_rate'), 0)}；"
                         f"故障清除后 {up.get('recovery_s_after_fault_cleared', TODO)}s 内自动恢复",
                         f"- 3 倍突发流量下令牌桶拦截超额请求，放行请求 P95 {(ch.get('3_burst_3x', {}) or {}).get('p95_ms', TODO)}ms；"
                         "热点问题 100% 命中缓存且不消耗限流令牌",
                         f"- 量化 + continuous batching 后 32 并发 P95 由 {FP32['P95_ms']}ms 降至 {LT32['P95_ms']}ms"]),
        ("技术栈：", ["FastAPI、asyncio、Redis、令牌桶、熔断降级、Prometheus/Grafana、vLLM"])])
    d.resume("版本 C｜客服 + 研报大模型服务化（训练→检索→部署全链路）", [
        ("项目定位：", ["打通“微调模型 → 知识检索 → 在线服务”全链路，输出可落地的部署配置、容量评估方法与故障止血 SOP"]),
        ("核心方案：", ["- 合并 LoRA 并以业务数据校准量化，用项目一同一测试集评估量化掉点，保证训练侧与部署侧指标口径一致",
                       "- 拆解 prefill（计算密集）与 decode（访存密集）瓶颈，围绕 TTFT 与 tokens/s 双目标调整批处理与缓存参数",
                       "- 设计降级优先级：缓存热点 → 量化 → 调度参数 → 小模型兜底 → 话术兜底转人工"]),
        ("实验与结果：", [f"- 单卡 32GB 下服务 32 并发，P95 {LT32['P95_ms']}ms、吞吐 {LT32['out_tok_per_s']} tokens/s；"
                         f"AWQ 相对 FP16 显存节省 "
                         f"{(1 - _q('awq_int4', 32)['gpu_peak_mem_MB'] / _q('fp16', 32)['gpu_peak_mem_MB']) * 100:.0f}%",
                         "- 形成量化对比表、压测报告、故障记录表与三项目串联架构文档"]),
        ("技术栈：", ["vLLM、AWQ/GPTQ、LMDeploy(对照)、FastAPI、Redis、locust、Prometheus"])])

    d.h2("1.3 面试直接可背的项目介绍（1-2 分钟）")
    d.box("""第三个项目是把前两个项目部署上线。场景还是电商客服：日均 5000 多请求，双 11 高峰 3 倍，要求 P95 两秒以内、32 并发不崩，GPU 扛不住要能降级，热门问题直接走缓存。
第一步是量化。我把项目一的 LoRA 合并后分别做了 AWQ-INT4、GPTQ-INT4 和 INT8，校准数据用的是业务客服对话而不是通用语料，然后在 vLLM 上对比显存、吞吐、首 token 延迟，并用项目一同一个测试集测准确率掉点，最后选了 AWQ。算一下就知道为什么：Qwen3-8B FP16 权重 16.4GB，32G 卡留给 KV Cache 的只有 12.4GB，只能撑 44 条 2K 长度的并发；AWQ 权重 4.6GB，KV Cache 有 24.2GB，能撑 86 条——同一张卡多服务一倍的人。而且 decode 是访存瓶颈，INT4 权重读得更快：32 并发 P95 从 1522ms 降到 986ms。
第二步是服务化，我用 FastAPI 写了网关：先做单用户限流，再查 Redis 缓存，缓存没命中才消耗全局令牌，然后按意图路由——客服问题走微调模型，研报问题先调项目二的 RAG 服务，如果约束校验不过就直接拒答不占 GPU。模型调用有超时和熔断，错误率超过 30% 或者 P95 超过 SLO 就切到小模型，并缩短 max_tokens，再不行就兜底话术转人工。
第三步是压测和故障演练，并发 1 到 32 阶梯压测记录 QPS、P50/P95/P99 和显存，演练了 OOM、慢请求、流量突增和热点集中。演练里我还发现一个设计问题：热点问题被全局限流误伤了，其实缓存命中几乎零成本，所以把全局限流挪到了缓存之后。""",
          title="约 2 分钟")
    d.h2("1.4 项目英文简历描述")
    d.box(f"""LLM Inference Serving & Optimization (vLLM + Quantization + Gateway)
• Served the fine-tuned Qwen3-8B customer-service model with vLLM; benchmarked FP16 vs. AWQ-INT4 vs. GPTQ-INT4 vs. INT8 on memory, throughput, TTFT and task accuracy, using in-domain calibration data.
• Built a FastAPI gateway with intent routing (chat model vs. RAG service), Redis hot-query caching with warm-up, token-bucket rate limiting, timeouts, a circuit breaker with small-model fallback, and token/latency logging with Prometheus metrics.
• Developed an async load tester (concurrency 1–32, P50/P95/P99, TTFT, tokens/s, GPU memory) and ran chaos drills (upstream errors, slow decoding, 3× traffic bursts, hot keys).
• Estimated KV-cache capacity with a GQA-aware formula: AWQ frees ~24.2 GB for KV cache on a 32 GB GPU (~86 concurrent 2K-token sequences vs. ~44 with FP16); achieved {LT32['QPS']} QPS and {LT32['P95_ms']} ms P95 at 32 concurrency ({LT32['out_tok_per_s']} output tokens/s, {LT32['gpu_peak_mem_MB'] / 1000:.1f} GB peak).
Tech: vLLM, AutoAWQ, llm-compressor, FastAPI, Redis, asyncio, Prometheus""", title="English version")

    # ======================================================================= 二
    d.h1("二、面试方法")
    d.h2("2.1 引导提问的思路、方向与话术")
    d.p("部署方向的追问更偏工程和系统，画风和训练/RAG 不一样。介绍里要主动报“数 + 公式 + 取舍”，把面试官引到你最熟的 KV Cache 估算和故障演练上。")
    d.table(["想引导的方向", "钩子话术", "准备好的另一半"],
            [["KV Cache", "“算一下就知道为什么必须量化”", "公式 2×L×(kv_heads×head_dim)×seq×batch×bytes；Qwen3-8B 每 token 144KB；GQA 比 MHA 公式少 4 倍"],
             ["量化取舍", "“校准数据用业务对话而不是通用语料”", "AWQ 保护激活大的显著通道；校准分布与线上一致掉点更小；用项目一测试集测掉点"],
             ["瓶颈分析", "“decode 阶段是访存瓶颈”", "prefill 计算密集并行处理 prompt；decode 每步读全部权重和 KV，batch 越大越摊薄"],
             ["稳定性", "“热点问题被全局限流误伤了”", "限流顺序调整：单用户限流→缓存→全局限流；演练前后对比"],
             ["降级", "“错误率或 P95 超 SLO 都会触发熔断”", "状态机、窗口、最小请求数、半开探测、降级动作的优先级"]],
            widths=[1, 2.2, 3.6])
    d.h2("2.2 追问应对、诚实边界与如何化解深挖")
    d.table(["事实", "可以怎么说", "别说"],
            [["mock 后端只验证了网关逻辑", "“网关的熔断/降级/限流逻辑先在故障注入 mock 上做了单元级演练，性能数字是在真实 vLLM 上压的”", "把 mock 的 QPS 当真实性能"],
             ["单机单卡，没有 K8s 集群", "“单机部署，多副本时缓存用 Redis 共享、限流用 Redis+Lua，扩容按压测得到的单卡容量线性估算”", "“支撑了线上千万级流量”"],
             ["没有真实双 11 流量", "“用 3 倍突发流量做了压测模拟”", "“经历过双 11”"],
             ["没写 CUDA kernel", "“用的是 vLLM/AWQ 的 Marlin kernel，理解原理但没自己写”", "“优化了 CUDA kernel”"]],
            widths=[2, 3.4, 1.6])
    d.ul(["被问“延迟砍一半从哪入手”：按收益/成本排序——缓存热 query > 量化 > speculative decoding > batch 策略优化 > 换更小模型；并说清楚每一项预期收益怎么验证。",
          "被问到没做过的（TensorRT-LLM、多卡张量并行）：讲原理 + 何时需要（单卡放不下 / 延迟要求更极致），并说明你的场景单卡 AWQ 已满足 SLO，所以没做。"])
    d.h2("2.3 面试常见 QA（45 题）")
    qa = [
        ("一句话介绍项目。", "vLLM 部署量化后的客服模型，FastAPI 网关串联 RAG，做缓存、限流、熔断降级与监控，并用压测和故障演练给出容量与 SOP。"),
        ("为什么选 vLLM？", "PagedAttention 解决 KV Cache 碎片、continuous batching 提高 GPU 利用率、OpenAI 兼容 API、支持 AWQ/GPTQ/LoRA/prefix caching，社区事实标准。"),
        ("PagedAttention 原理？", "把 KV Cache 按固定大小的 block 分页管理，序列的逻辑块映射到不连续的物理块，类似操作系统虚拟内存；消除预分配最大长度造成的内部碎片，还能在并行采样/前缀共享时共享块。"),
        ("continuous batching 是怎么回事？", "不等一批请求凑齐、也不等整批结束：每个 decode step 调度器都可以把已完成的序列移出、把新请求插入，动态合并请求，GPU 利用率显著高于静态 batching。"),
        ("Prefill 和 Decode 有什么区别？", "Prefill 一次性并行处理整个 prompt，计算密集（矩阵乘），决定 TTFT；Decode 逐 token 自回归，每步都要读全部权重和 KV Cache，访存密集，决定 TPOT/吞吐。"),
        ("P95 瓶颈通常在哪？", "decode 阶段（访存瓶颈）和排队：并发高时请求在调度队列等待，长 prompt 的 prefill 会阻塞别人的 decode（可用 chunked prefill 缓解）。"),
        ("KV Cache 显存怎么估？", "2(K,V) × num_layers × hidden × seq_len × batch × bytes。GQA 模型要把 hidden 换成 num_kv_heads × head_dim：Qwen3-8B 是 36 层、8 个 KV head、head_dim 128，每 token 2×36×1024×2B=144KB，2K×32 条约 9GB。"),
        ("为什么 GQA 能省显存？", "多个 query head 共享一组 KV head，KV Cache 与 KV head 数成正比；Qwen3-8B 32 个 Q head 只有 8 个 KV head，KV 缩小 4 倍，同显存可服务 4 倍并发。"),
        ("32GB 卡能服务多少并发？", "FP16 权重 16.4GB，按 0.9 利用率扣掉激活预留，KV 只剩约 12.4GB≈9.0 万 token，约 44 条 2K 序列；AWQ 权重约 4.6GB，KV 约 24.2GB≈17.6 万 token，约 86 条（bench/kv_cache_calc.py）。"),
        ("gpu-memory-utilization 是什么？", "vLLM 预分配给权重 + KV block 池的显存比例；设太高和其他进程抢显存会 OOM，太低浪费并发能力。"),
        ("max-num-seqs、max-model-len 怎么设？", "max-model-len 决定单序列最长 KV；客服场景 4096 够用。max-num-seqs 是同时调度的序列上限，太小吞吐低、太大单请求变慢且易抢占。用压测找 P95 和吞吐的平衡点。"),
        ("prefix caching 有什么用？", "相同前缀（system prompt）的 KV 复用，客服场景每个请求共享长 system prompt，命中后 prefill 计算显著减少、TTFT 降低。"),
        ("AWQ 原理？", "Activation-aware Weight Quantization：根据激活幅度找出约 1% 的显著权重通道，用逐通道缩放保护它们，再做 group-wise INT4 量化，无需反向传播。"),
        ("GPTQ 原理？和 AWQ 区别？", "GPTQ 逐层用二阶信息（Hessian 近似）做最优量化并补偿误差（OBQ 思想）；AWQ 基于激活的缩放，校准更快、对校准集过拟合更少。实际效果接近，vLLM 上都有 Marlin kernel。"),
        ("量化掉了多少精度？", "用项目一同一个 400 条测试集测准确率/自动解决率（run_quant_bench.sh 自动完成），报实测值；没有测过就说不出这个数，所以量化实验必须做。"),
        ("为什么 INT4 反而可能更快？", "decode 是访存瓶颈，INT4 权重只有 FP16 的 1/4，读权重的带宽压力大幅下降；配合 Marlin 等高效 kernel，单请求 tokens/s 往往更高。但大 batch 计算密集时反量化开销会显现。"),
        ("校准数据为什么用业务数据？", "量化误差取决于激活分布，用与线上一致的客服对话校准，显著通道选得更准；通用 wiki 文本校准可能保护了错的通道。"),
        ("W8A8 和 W4A16 区别？", "W4A16 只量化权重，计算时反量化为 16bit，省显存、提升 decode；W8A8 权重和激活都 8bit，可用 INT8 Tensor Core，计算密集的大 batch 场景更有优势，需要 SmoothQuant 处理激活离群值。"),
        ("KV Cache 也能量化吗？", "可以，vLLM 支持 FP8 KV Cache（--kv-cache-dtype fp8），KV 显存减半、并发翻倍，代价是轻微精度损失，需要用测试集验证。"),
        ("网关请求链路是怎样的？", "单用户限流 → 缓存查询 → 全局令牌桶（只对打到 GPU 的请求） → 意图路由 → （研报：RAG 检索 + 约束校验）→ 带超时的模型调用（熔断/降级）→ 日志与指标 → 写缓存。"),
        ("缓存 key 怎么设计？会不会答错？", "归一化 query（去语气词、标点、空白）+ 路由做 MD5；只缓存通用 FAQ，带个人订单上下文的请求不缓存，避免“A 的订单答案给了 B”。热点问题启动时预热，TTL 一天。"),
        ("为什么限流要放在缓存之后？", "演练发现热点问题被全局令牌桶误伤（命中率 100% 却有大量 429）。缓存命中几乎零成本，不应消耗 GPU 配额；所以单用户限流在前（防刷），全局限流在缓存之后。"),
        ("令牌桶和漏桶区别？", "令牌桶允许一定突发（桶容量），平均速率受限；漏桶以恒定速率流出，平滑但不允许突发。在线服务通常用令牌桶。"),
        ("多副本时限流怎么做？", "Redis + Lua 原子脚本实现分布式令牌桶（ratelimit.py 内附），或在网关层（Nginx/Envoy）做；缓存同样用 Redis 共享。"),
        ("熔断器怎么设计？", "滑动窗口统计错误率和 P95；请求数 ≥ min_requests 时，错误率 >30% 或 P95 > SLO 打开熔断，所有请求走降级；open_seconds 后半开，放一个探测请求，成功则关闭、失败重新打开。"),
        ("降级策略有哪些？优先级？", "① 切小模型（量化 4B）② 缩短 max_tokens ③ 话术兜底 + 转人工；更前置的是缓存与拒答（不占 GPU）。"),
        ("超时怎么设？", "单次 LLM 调用 8 秒总预算，降级调用用一半预算；超时立即返回兜底话术，避免请求堆积拖垮整体。流式场景还要看首 token 超时。"),
        ("OOM 了怎么办？", f"现象：进程被 kill / CUDA OOM。排查：nvidia-smi 看显存、检查 batch 与 max_model_len、是否有超长请求。修复：降 max-model-len、限制 max-num-seqs/max-num-batched-tokens、开 chunked prefill、降 gpu-memory-utilization 给其他进程留空间、换量化。网关侧熔断把流量切到降级模型。"),
        ("延迟突然飙升怎么排查？", "看 vLLM 指标：num_requests_waiting（排队）、gpu_cache_usage_perc（KV 满了会触发抢占/重算）、TTFT vs TPOT 谁变大；常见原因是长请求占满 KV Cache 或流量突增。修复：限流、chunked prefill、长短请求分流。"),
        ("请求抖动 / 输出被截断？", "抖动：GC、显存碎片、CPU 预处理瓶颈，可预热 + 固定 batch 参数；截断：max_tokens 太小或 finish_reason=length，调大 max_tokens 或流式输出。"),
        ("怎么压测？指标有哪些？", "并发阶梯 1/4/8/16/32，每档固定请求数；记录 QPS、P50/P95/P99、TTFT、输出 tokens/s、错误率、429 数、显存峰值；prompt 用真实测试集保持长度分布。"),
        ("为什么 P95 比平均值重要？", "SLO 关心尾部体验，平均值会被大量快请求掩盖；P95/P99 才反映排队和长请求的影响。"),
        ("QPS 和并发是什么关系？", "Little 定律：并发 = QPS × 平均延迟。并发上去后 QPS 先线性增长，接近 GPU 饱和后增长放缓、延迟上升，拐点就是容量上限。"),
        ("流式输出有什么好处？", "首 token 很快返回，用户感知延迟取决于 TTFT 而非总时长；网关透传 SSE，同时统计 TTFT。"),
        ("如果要把延迟砍一半，你从哪入手？", "缓存热 query（零成本）> 量化（访存瓶颈直接受益）> speculative decoding（小模型草稿+大模型验证）> batch 策略（max-num-seqs、chunked prefill）> 换更小模型。每步都用压测验证。"),
        ("speculative decoding 原理？", "小模型（或 n-gram/EAGLE 头）先生成 k 个草稿 token，大模型一次前向并行验证，接受的 token 直接输出；输出分布与大模型一致，在接受率高时显著加速 decode。"),
        ("vLLM、SGLang、LMDeploy 怎么选？", "vLLM 生态最全；SGLang 的 RadixAttention 前缀复用和结构化输出更强；LMDeploy TurboMind + AWQ 性能好、部署简单。可用同一压测脚本做 A/B。"),
        ("多 LoRA 怎么服务？", "vLLM 支持 --enable-lora 同时加载多个 adapter，请求按 model 名路由到不同 LoRA，共享底座权重；适合客服/研报两个 adapter 共用一个底座。"),
        ("监控看哪些指标？", "网关：QPS、P50/P95/P99、错误率、缓存命中率、429、熔断状态、input/output tokens；vLLM：running/waiting 请求数、KV cache 使用率、TTFT/TPOT 直方图；GPU：显存、利用率。"),
        ("日志记录什么？", "每个请求：request_id、路由、是否命中缓存、served_by（主/降级/兜底/拒答）、input_tokens、output_tokens、latency、status、降级原因；JSONL 便于统计每小时 QPS、平均延迟和错误率。"),
        ("意图路由怎么做的？准确吗？", "关键词规则优先（零延迟、可解释）：研报/财务/政策词 vs 客服词打分；生产上可换成 bge-small + 线性分类器。路由错误时 RAG 约束校验会拒答，客服模型也会说明无法处理，不会产生严重后果。"),
        ("研报路由为什么在网关就能拒答？", "RAG 服务的 retrieve_only 接口返回约束校验结果（年份/主体/指标是否在证据里），不通过网关直接返回“资料中未找到”，不占用 GPU。"),
        ("怎么验证量化后效果没坏？", "同一个测试集、同样的评测脚本、贪心解码：07_evaluate.py --backend openai 指向量化服务；对比准确率、复读率、转人工率，并抽查 badcase 是否新增。"),
        ("如何做容量规划？", "压测得到单卡在 SLO 下的最大 QPS，按峰值（日常×3）÷ 单卡 QPS ×（1+冗余 30%）估算卡数；再用 KV Cache 公式校验并发上限。"),
        ("这个项目最大的收获？", "性能问题要用数说话：先估算（KV 公式）再实测（压测）再演练（故障）；很多“优化”最便宜的是缓存和限流顺序这种系统设计，而不是改 kernel。"),
    ]
    d.qa(qa)

    # ======================================================================= 三
    d.h1("三、项目启动方法")
    d.code("""cd project3_llm_serving
pip install -r requirements.txt                       # Linux + CUDA 12.x
# W9 合并 + 量化 + 对比（自动：合并→AWQ/GPTQ/INT8→逐个起 vLLM→测准确率→压测→汇总）
bash scripts/run_quant_bench.sh                       # 结果：results/quant_summary.md
# 单独起服务
bash scripts/start_vllm.sh awq models/qwen3-8b-cs-awq # :8000，served-model-name=cs-model
# W10 服务化 + 压测
redis-server --daemonize yes
uvicorn gateway.main:app --host 0.0.0.0 --port 9000
python bench/load_test.py --target openai  --url http://localhost:8000/v1 --stream --tag vllm_awq
python bench/load_test.py --target gateway --url http://localhost:9000 --hot_ratio 0.3 --tag gateway
bash scripts/run_vllm_param_bench.sh                  # max-num-seqs / prefix caching / chunked prefill
python bench/kv_cache_calc.py --model qwen3-8b --seq_len 2048 --batch 32 --weights awq
# W11 故障演练
python bench/chaos_drill.py --gateway http://localhost:9000 --primary_admin http://localhost:8000
# W12 三项目串联
bash scripts/start_all.sh                             # vLLM + RAG(:8100) + Redis + 网关(:9000)
curl -X POST localhost:9000/chat -H 'Content-Type: application/json' -d '{"query":"双11能叠加优惠券吗"}'
curl -X POST localhost:9000/chat -H 'Content-Type: application/json' -d '{"query":"宁川动力2025年三季度毛利率是多少"}'
curl localhost:9000/stats

# 无 GPU 验证网关逻辑（mock 后端，支持故障注入 /admin/fault）
python mock/mock_llm_server.py --port 8000 --model_name cs-model --step_ms 12 --mean_out 60 &
python mock/mock_llm_server.py --port 8001 --model_name fallback-model --step_ms 8 --mean_out 40 &
REDIS_URL= uvicorn gateway.main:app --port 9000 &
python bench/chaos_drill.py""")
    d.table(["常见问题", "解决"],
            [["vLLM 启动 OOM", "降 --gpu-memory-utilization / --max-model-len；确认没有其他进程占显存"],
             ["AWQ 模型加载报 quantization 不匹配", "AWQ 用 awq_marlin、GPTQ 用 gptq_marlin；llm-compressor 产物无需指定 --quantization"],
             ["返回内容带 <think>", "请求体加 chat_template_kwargs: {enable_thinking: false}（网关和压测脚本已默认添加）"],
             ["Redis 未安装", "网关自动退化为进程内 LRU（/stats 中 cache_backend 字段可见）"],
             ["Windows 上跑 vLLM", "vLLM 仅支持 Linux，Windows 请用 WSL2 + CUDA；网关/压测脚本可在 Windows 原生运行"]],
            widths=[2.2, 4])

    # ======================================================================= 四
    d.h1("四、项目实验数据")
    d.h2("4.1 KV Cache 容量估算（确定性计算，可直接引用）")
    d.table(["模型 / 权重精度", "每 token KV", "2K×32 并发 KV", "权重显存", "32GB×0.9 可用 KV", "可服务 2K 序列数"],
            [["Qwen3-8B FP16（GQA 8 KV head）", "144 KB", "9.0 GB", "16.4 GB", "≈12.4 GB（≈90K token）", "≈44"],
             ["Qwen3-8B AWQ-INT4", "144 KB", "9.0 GB", "≈4.6 GB", "≈24.2 GB（≈176K token）", "≈86"],
             ["若误用 MHA 公式（hidden=4096）", "576 KB", "36 GB", "—", "—", "高估 4 倍"]],
            widths=[2.4, 1, 1.1, 1, 1.8, 1.2])
    d.h2("4.2 量化对比（5090 单卡实测）")
    d.small("准确率一律用项目一同一份 400 条测试集、同一评测脚本、贪心解码：FP16 的 81.5% 就是项目一 SFT+DPO(β=0.1) 的准确率，"
            "量化后的掉点才和项目一对得上。显存峰值为 32 并发下权重 + 实际 KV 占用 + 激活的实测口径。")
    d.table(["方案", "模型体积GB", "显存峰值MB", "单请求 tokens/s", "32 并发 tokens/s", "准确率(项目一测试集)", "相比 FP16"],
            [_quant_row("FP16 原始", "fp16"), _quant_row("AWQ INT4", "awq_int4"),
             _quant_row("GPTQ INT4", "gptq_int4"), _quant_row("INT8 W8A8", "int8_w8a8")],
            widths=[1.2, 1, 1, 1.1, 1.2, 1.3, 2.4], font_size=7.6)
    d.h2("4.3 并发压测（真实 vLLM，5090 单卡实测）")
    d.small(f"设置：AWQ-INT4 服务，流式输出，平均输出 62 token，每档固定请求数，记录 QPS / P50 / P95 / P99 / TTFT / 输出 tokens/s / GPU 显存峰值。"
            f"AWQ 在 32 并发下 P95 {LT32['P95_ms']}ms、QPS {LT32['QPS']}；FP16 同一档是 {FP32['P95_ms']}ms、QPS {FP32['QPS']}——"
            f"FP16 虽然也在 2s SLO 内，但显存峰值 {FP32['gpu_peak_mem_MB'] / 1000:.1f}GB、KV 余量只够 44 条 2K 序列，"
            f"换 AWQ 后显存降到 {LT32['gpu_peak_mem_MB'] / 1000:.1f}GB、P95 再降 "
            f"{(1 - LT32['P95_ms'] / FP32['P95_ms']) * 100:.0f}%，并发余量翻倍到 86 条：同样的卡多服务一倍的人，这才是量化的理由。")
    real = p3("loadtest_vllm_awq.json")
    rows = [[r["concurrency"], r["QPS"], r["P50_ms"], r["P95_ms"], r["P99_ms"], r["TTFT_P50_ms"], r["out_tok_per_s"],
             r["gpu_peak_mem_MB"]] for r in real["rows"]] if real else \
        [[c, TODO, TODO, TODO, TODO, TODO, TODO, TODO] for c in (1, 4, 8, 16, 32)]
    d.table(["并发", "QPS", "P50 ms", "P95 ms", "P99 ms", "TTFT P50", "输出 tokens/s", "显存峰值MB"], rows)
    d.p(f"网关叠加 30% 热点缓存后（results/loadtest_gateway_vllm_hot30.json）：32 并发 QPS 由 {LT32['QPS']} 提升到 {GW32['QPS']}，"
        f"缓存命中率 {pct(HIT, 0)}，命中请求不经过 GPU，延迟 <1ms 且不消耗限流令牌；"
        f"FP16 对照（results/loadtest_vllm_fp16.json）32 并发 QPS {FP32['QPS']}、P95 {FP32['P95_ms']}ms、显存峰值 {FP32['gpu_peak_mem_MB']}MB。")
    d.h2("4.4 网关逻辑验证（mock 后端，本机实测；仅验证逻辑，不代表 GPU 性能）")
    gw = p3("loadtest_mock_gateway_hot30.json")
    if gw:
        d.small("设置：mock 主模型单步 12ms、平均输出 60 token；30% 热点问题；研报问题经真实 RAG 检索服务（bge-small + reranker-base）。")
        d.table(["并发", "QPS", "P50 ms", "P95 ms", "P99 ms", "错误率", "429"],
                [[r["concurrency"], r["QPS"], r["P50_ms"], r["P95_ms"], r["P99_ms"], pct(r["error_rate"]), r["http_429"]]
                 for r in gw["rows"]])
    if ch:
        rows = []
        names = {"0_baseline": "基线（无故障）", "1_upstream_error": "上游报错（模拟 OOM/5xx）", "2_slow_decode": "解码变慢 6 倍（模拟 KV 抢占）",
                 "3_burst_3x": "3 倍突发流量（96 并发）", "4_hot_query": "热点问题集中访问"}
        for k, name in names.items():
            v = ch.get(k, {})
            br = v.get("breaker", {})
            rows.append([name, v.get("n"), pct(v.get("success_rate")), v.get("p95_ms"), str(v.get("served_by")),
                         v.get("http_429"), pct(v.get("cache_hit_rate")), (br.get("reason") or "-")])
        d.table(["演练场景", "请求数", "成功率", "P95 ms", "服务来源分布", "429", "缓存命中", "熔断原因"], rows,
                widths=[2, 0.7, 0.8, 0.8, 2.2, 0.6, 0.8, 1.6], font_size=7.3)
        d.p(f"读法：上游报错时熔断器在错误率达到 30% 阈值后打开，请求全部由降级模型兜底（成功率 {pct(up.get('success_rate'))}），"
            f"故障清除后经半开探测 {up.get('recovery_s_after_fault_cleared', TODO)} 秒恢复；解码变慢时由 P95 超 SLO 触发熔断；"
            "突发流量下令牌桶拦截超额请求保护后端；热点问题在调整限流顺序后 100% 命中缓存、不再被误伤（调整前同一场景有 54% 的请求被 429）。")
    d.h2("4.5 故障记录表")
    d.table(["故障", "现象", "排查手段", "修复方案"],
            [["OOM", "vLLM 进程被 kill / CUDA OOM，网关 5xx 激增", "nvidia-smi 看显存；检查 batch、max_model_len、超长请求", "降 max_model_len、限 max-num-seqs、开 chunked prefill、换量化；熔断切降级模型"],
             ["延迟飙升", "P95 从 1.x s 到 5 s+", "vLLM num_requests_waiting、gpu_cache_usage_perc、TTFT vs TPOT", "限流、continuous batching 参数、长短请求分流；P95 熔断降级"],
             ["流量突增", "排队、超时、错误率上升", "/stats QPS 与 429 计数", "令牌桶限流 + 缓存 + 降级；按压测容量扩容"],
             ["热点被误伤", "热点命中缓存却返回 429", "演练日志：cache_hit 与 429 同时出现", "全局限流移到缓存之后"],
             ["请求抖动", "部分请求偶发变慢", "查 GC、CPU 预处理、显存碎片", "预热 + 固定 batch 参数"],
             ["输出截断", "回答不完整", "finish_reason=length", "调大 max_tokens / 流式输出"],
             ["RAG 服务不可用", "研报类请求失败", "网关日志 rag_unavailable", "返回 503 明确提示，不让模型无证据作答"]],
            widths=[1, 1.8, 2.2, 2.8], font_size=7.6)
    d.h2("4.6 性能优化顺序（面试可直接说）")
    d.p("缓存（热点零成本）> 量化（decode 访存瓶颈直接受益）> Batching 与调度参数（max-num-seqs、prefix caching、chunked prefill）> speculative decoding > 硬件升级。"
        "每一步都要用同一压测脚本、同一 prompt 分布验证。")
