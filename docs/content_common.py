"""封面、总览、附录（通用经验）。"""
from reportlab.platypus import Paragraph, Spacer

from pdfkit import ST


def cover():
    return [Spacer(1, 150), Paragraph("大模型算法岗 · 三项目实战", ST["title"]), Spacer(1, 14),
            Paragraph("电商客服 SFT+DPO ｜ 金融研报 RAG ｜ vLLM 推理部署", ST["subtitle"]), Spacer(1, 10),
            Paragraph("项目文档：项目介绍 · 简历描述 · 面试方法 · 常见 QA · 启动方法 · 实验数据", ST["subtitle"]),
            Spacer(1, 220),
            Paragraph("源码目录：project1_ecom_cs_sft_dpo / project2_fin_report_rag / project3_llm_serving", ST["small"]),
            Paragraph("说明：文中所有实验数字均由仓库内脚本实际运行得到，并标注运行环境；"
                      "8B 训练与 vLLM 部署的指标来自 5090 单卡实测，项目一数据管线、规则基线与 0.6B 复现、项目二全部检索实验、"
                      "项目三网关逻辑验证来自本机（Apple M5 Pro）实测。", ST["small"])]


def overview(d):
    d.part("第〇部分  总览与阅读说明")
    d.h1("0.1 先读这个：数字从哪来、哪些能写进简历")
    d.box("""本文档的每一个数字都来自仓库内脚本的真实运行结果（文档构建脚本 docs/build_doc.py 直接读取各项目 results/ 下的 JSON 填表）。
分三类，写简历和面试时请严格区分：
① 【本机实测】：Apple M5 Pro（无 NVIDIA GPU）上实际跑出的数字。项目一的数据清洗统计与规则基线、项目一 Qwen3-0.6B 小规模复现、项目二全部检索/重排/拒答实验、项目三 KV Cache 计算与网关逻辑验证都属于这一类。
② 【5090 实测】：在 5090 单卡上跑 Qwen3-8B / vLLM 得到的数字（8B 的 SFT/DPO 效果、量化对比、真实 QPS/P95），
见项目一 4.4、项目二 4.6 最后一行、项目三 4.2–4.3。这批数字是简历里最硬的部分，要能说出训练耗时、显存峰值、badcase 条数这些细节。
③ 【mock 模拟】：项目三网关在 mock 后端上的压测/故障演练数字，只能证明“熔断、降级、限流、缓存的逻辑按设计工作”，不代表 GPU 性能，不要写进简历的性能指标里。""",
          title="诚实边界（非常重要）")
    d.p("这样做的原因很简单：面试官追问三层就会露馅的数字，写上去反而减分（见附录“做完之后怎么判断能不能写进简历”）。"
        "本仓库的设计目标是：你在 5090 上按脚本跑一遍，得到的每个数字都能讲清楚“怎么来的、为什么是这个值、换个设置会怎样”。")

    d.h1("0.2 三个项目是一条线")
    d.p("三个项目不是三份独立作业，而是一条完整链路：**电商客服**里你微调了模型（训练与对齐）→ **金融研报**里你做了知识检索（RAG）→ "
        "**部署上线**时把两者串到一个在线服务里（推理优化与工程化）。面试官最想看到的就是：你能不能从一个具体需求出发，把训练、检索、上线这条路走通。")
    d.code("""用户请求 ─> FastAPI 网关 ─ 限流(令牌桶) ─ 缓存(Redis, 热点预热) ─ 命中直接返回
                 │
                 ├─ 意图路由: 客服类 ──> vLLM[项目一 SFT+DPO 模型, AWQ-INT4]
                 └─ 意图路由: 研报类 ──> RAG 服务[项目二: 改写→FAISS+BM25→bge-reranker→约束校验]
                                          └─> vLLM 生成(证据绑定 + 引用校验)
                 超时/熔断 ──> 降级小模型 ──> 兜底话术 + 转人工
                 日志(input/output tokens, latency) + /stats + Prometheus""")
    d.table(["项目", "方向", "核心技术", "必须有的实验", "最能拉开差距的点"],
            [["一 电商客服", "训练与对齐", "Qwen3-8B、TRL、PEFT LoRA/QLoRA、DPO",
              "rank 消融、target_modules、QLoRA vs LoRA、SFT vs DPO、β、偏好噪声", "数据清洗统计 + 翻车案例（复读机/变卦/事实错误）"],
             ["二 研报 RAG", "检索增强", "pdfplumber 版面解析、BGE、FAISS、BM25、bge-reranker",
              "切块网格、多路召回、索引对比、重排、拒答阈值", "多栏/表格解析 + 证据级评测集 + 拒答设计"],
             ["三 推理部署", "推理优化与工程", "vLLM、AWQ/GPTQ、FastAPI、Redis、Prometheus",
              "量化对比、并发阶梯压测、参数对比、故障演练", "能报出 QPS/P95/显存 + 讲清瓶颈与止血 SOP"]],
            widths=[1.1, 1, 2.2, 2.6, 2.4])

    d.h1("0.3 执行节奏（12 周 = 3 个月）")
    d.table(["周", "项目一：训练与对齐", "项目二：RAG 系统", "项目三：推理部署"],
            [["W1", "客服数据清洗 + SFT 跑通", "—", "—"], ["W2", "LoRA 消融实验", "—", "—"],
             ["W3", "客服偏好数据 + DPO", "—", "—"], ["W4", "客服 badcase + 文档", "—", "—"],
             ["W5", "—", "研报 PDF 解析 + 切块 + baseline", "—"], ["W6", "—", "金融 QA 评测集 + 多路召回实验", "—"],
             ["W7", "—", "重排 + 故障处理", "—"], ["W8", "—", "Demo + 评测文档", "—"],
             ["W9", "—", "—", "客服模型量化实验"], ["W10", "—", "—", "服务化 + 压测"],
             ["W11", "—", "—", "工程化 + 故障演练"], ["W12", "—", "—", "客服 + 研报三项目串联 + 总文档"]],
            widths=[0.6, 2, 2, 2])
    d.p("前 1 个月补基础（Transformer、训练闭环，建议先花 2 天读 nanoGPT 源码），中间 3 个月做这三个项目，后 2 个月刷题 + 面试。"
        "算力：默认单卡 5090（32GB）。三个项目都按“不吃算力”设计：8B 用 QLoRA 单卡可训，推理用 AWQ-INT4 单卡可服务。")

    d.h1("0.4 仓库结构与交付物")
    d.code("""LLM_Projects/
├── project1_ecom_cs_sft_dpo/   # 项目一：数据构造/清洗 → SFT(LoRA/QLoRA) → 偏好数据 → DPO → 评测 → badcase → 合并
├── project2_fin_report_rag/    # 项目二：演示语料 → 版面解析 → 切块 → 混合检索 → 重排 → 约束生成 → 评测 → Demo/API
├── project3_llm_serving/       # 项目三：量化 → vLLM → 网关(缓存/限流/路由/熔断降级/监控) → 压测 → 故障演练
├── docs/                       # 本文档的生成脚本（python docs/build_doc.py 可随实验结果重新生成）
└── 项目文档.pdf""")


def appendix(d):
    d.part("附录  通用经验：踩坑、简历自检、开源仓库")
    d.h1("A.1 我踩过的坑，你们别再踩了")
    d.ol(["**只跑通不做实验**——面试官问“你做了什么优化”，张嘴说不出东西，氛围直接冷下来。每个项目至少 3 组对比实验。",
          "**用公开 demo 数据还不改**——面试官一眼看出来，问“数据哪来的”就尴尬。本仓库的数据管线允许你接入公开数据、自己扩写、自己清洗，"
          "你要能报出“原始多少条、每条规则过滤多少、最后多少条”。",
          "**三个项目互不相干**——做了训练、RAG、部署但串不起来，整体感全无。本仓库项目三的网关直接调用项目一的模型和项目二的检索服务。",
          "**只有代码没有文档**——你能跑不代表面试官能看到你的思考过程，文档（实验表、badcase、故障记录）是你的证据。",
          "**贪多嚼不烂**——做 5 个浅的不如 3 个深的，面试不看数量看深度。",
          "**不准备追问**——项目做了但讲的时候结结巴巴，和没做区别不大。第二章的追问清单逐条过一遍。"])
    d.h1("A.2 做完之后怎么判断能不能写进简历（5 条自检）")
    d.table(["自检项", "标准", "本仓库对应的证据"],
            [["至少 3 组对比实验", "不是“跑通了”，是“试过几种方案知道哪个好”", "各项目第四章实验表"],
             ["能说出口的数字", "“效果不错”是废话，“准确率从 58% 到 82%”才是有效信息", "results/*.json，均可复现"],
             ["有翻车记录", "面试官最想听你遇到问题怎么解决", "badcase 报告、故障演练记录"],
             ["能讲满 5 分钟不卡壳", "背景→问题→方案→实验→结果→复盘", "1.3 面试稿"],
             ["追问三层接得住", "为什么→换个方案会怎样→线上出问题怎么办", "2.2 追问应对 + 2.3 QA"]],
            widths=[1.3, 3, 2])
    d.p("五条里不到三条的项目，说实话别往简历上放，写上去被问到反而减分。")
    d.h1("A.3 讲 badcase 的万能结构")
    d.box("出了什么问题（现象+数据）→ 为什么（定位过程，看了哪些信号）→ 试了什么（至少两个方案，一个没用一个有用）→ 解决没（修复前后数字）→ 复盘（以后怎么提前避免）",
          title="四步法")
    d.h1("A.4 开源仓库汇总")
    d.table(["仓库", "地址", "用途 / 建议精读部分"],
            [["LLaMA-Factory", "github.com/hiyouga/LLaMA-Factory", "SFT+DPO 全流程参考，数据格式规范、LoRA 配置、对齐算法实现（本仓库 configs/ 提供对照配置）"],
             ["TRL", "github.com/huggingface/trl", "SFTTrainer / DPOTrainer，自写训练脚本的规范参考"],
             ["PEFT", "github.com/huggingface/peft", "LoRA / QLoRA 实现，读 lora/layer.py"],
             ["Transformers", "github.com/huggingface/transformers", "模型加载 / 推理 / chat template"],
             ["Qwen3", "github.com/QwenLM/Qwen3", "基座模型文档 + 微调最佳实践（thinking 开关）"],
             ["nanoGPT", "github.com/karpathy/nanoGPT", "理解训练闭环，建议做项目前花 2 天读源码"],
             ["RAGFlow", "github.com/infiniflow/ragflow", "生产级 RAG，重点读 DeepDoc PDF 解析 + 检索融合 + 评测体系"],
             ["Langchain-Chatchat", "github.com/chatchat-space/Langchain-Chatchat", "中文 RAG 经典参考：中文切块、向量库适配、混合检索"],
             ["QAnything", "github.com/netease-youdao/QAnything", "2-stage retrieval（Embedding 粗排 + Rerank 精排）与 rerank 阈值设计"],
             ["FAISS / FlagEmbedding", "github.com/facebookresearch/faiss ｜ github.com/FlagOpen/FlagEmbedding", "向量检索 / BGE Embedding + Reranker"],
             ["LlamaIndex / Milvus / GraphRAG", "run-llama/llama_index ｜ milvus-io/milvus ｜ microsoft/graphrag", "快速 baseline / 生产级向量库 / 多跳问答进阶"],
             ["vLLM", "github.com/vllm-project/vllm", "推理事实标准，重点读 PagedAttention + continuous batching 调度器"],
             ["LMDeploy / SGLang", "InternLM/lmdeploy ｜ sgl-project/sglang", "量化+部署一体化 benchmark 对照 / 进阶 A/B 对比"],
             ["AutoAWQ / AutoGPTQ / llm-compressor", "casper-hansen/AutoAWQ ｜ AutoGPTQ/AutoGPTQ ｜ vllm-project/llm-compressor", "AWQ / GPTQ / W8A8 量化（AutoGPTQ 已停更，GPTQ 推荐 llm-compressor）"]],
            widths=[1.5, 2.6, 3.6])
    d.p("说法建议：“我参考了 RAGFlow 的文档解析思路 / LLaMA-Factory 的数据模板，但训练脚本、检索链路和评测是自己基于 TRL / FAISS 实现的”，"
        "比“我用 LLaMA-Factory 跑的”更容易被认可，也是事实（本仓库训练、检索、网关代码均为自写）。")
    d.h1("A.5 三个项目合在一起时的简历排版建议")
    d.ul(["每个项目按“项目定位 → 核心方案（3-4 条）→ 实验与结果（2-3 个数字）→ 技术栈”四段写，和参考简历格式一致。",
          "三个项目的数字不要互相矛盾：项目三量化后的准确率必须用项目一同一测试集测，“掉点 <1%”要和项目一的绝对值对得上。",
          "项目名可以按目标岗位微调：投训练岗突出项目一、投 RAG/应用岗突出项目二、投推理/infra 岗突出项目三，顺序也随之调整。",
          "同一份简历里三个项目用不同的“降重表述”（第一章 1.2 每个项目给了三版），避免句式雷同。"])
