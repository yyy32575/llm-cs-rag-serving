# 大模型算法岗三项目（源码 + 项目文档）

| 目录 | 项目 | 一句话 |
|---|---|---|
| `project1_ecom_cs_sft_dpo/` | 电商客服 SFT + LoRA 消融 + DPO | Qwen3-8B 后训练：数据清洗 → QLoRA → 偏好数据 → DPO → 分桶评测 |
| `project2_fin_report_rag/` | 金融研报 RAG | 多栏 PDF 版面解析 → 表格保护切块 → FAISS+BM25 混合召回 → 重排 → 约束拒答与引用校验 |
| `project3_llm_serving/` | 推理部署 | vLLM + AWQ 量化 → FastAPI 网关（缓存/限流/路由/熔断降级/监控）→ 压测 → 故障演练 |
| `项目文档.pdf` | 文档 | 项目介绍、简历描述（三版降重 + 英文）、面试稿、面试方法、每项目 45 题 QA、启动方法、实验数据 |
| `docs/` | 文档生成脚本 | `python docs/build_doc.py` —— 从各项目 `results/` 读取实测结果重新生成 PDF |

三个项目是一条线：项目三的网关调用项目一的模型（合并 LoRA 后量化）与项目二的检索服务（`/rag/query`）。

## 环境

- Python 3.10+；正式实验需要 Linux + 单卡 RTX 5090 32GB（vLLM / bitsandbytes 仅支持 Linux，Windows 请用 WSL2）。
- 无 NVIDIA GPU 也能运行：项目一数据管线/规则基线/0.6B 小规模复现（Apple MPS 或 CPU）、项目二全部检索实验（CPU）、项目三网关 + mock 后端。
- 各项目目录下有 `requirements.txt` 与 `README.md`，启动步骤详见《项目文档.pdf》每个项目的第三章。

## 实验数字说明

文档中的数字分三类：【5090 实测】8B 训练与 vLLM 部署指标，可直接写进简历；【本机实测】项目一数据管线与 0.6B 复现、
项目二全部检索实验、项目三网关逻辑验证；【mock 模拟】只用于验证网关逻辑，不代表 GPU 性能。
详见文档第〇部分“诚实边界”。重新跑实验后执行 `python docs/build_doc.py` 即可刷新文档里的表格。
