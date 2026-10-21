# 项目一：电商客服问答 —— Qwen3-8B 的 SFT + LoRA 消融 + DPO 偏好对齐

> 业务背景：电商平台日均 5000+ 客服咨询（退换货、物流、商品参数、售后投诉）。现有关键词规则引擎在新问法上的自动解决率只有约 30%~40%，其余全部转人工。
> 目标：让 7B/8B 级模型接住高频问题，把自动解决率提升到 70%+，同时保证政策准确、少转人工、不复读。

## 目录结构

```
project1_ecom_cs_sft_dpo/
├── src/ecs/
│   ├── sop.py          # 业务 SOP 唯一事实源：20 个意图 × 6 大业务域，政策数字、商品库、问法模板(train/test 隔离)
│   ├── prompts.py      # 训练/推理/评测统一的 system prompt + chat template（Qwen3 关闭 thinking）
│   ├── clean.py        # 数据清洗：PII 脱敏、短回复过滤、政策冲突检测、精确去重 + MinHash 近重复
│   ├── metrics.py      # 自动评测：准确率/自动解决率/满意度/复读率/转人工率（含否定感知的禁止表述匹配）
│   ├── rule_bot.py     # 线上规则引擎 baseline（关键词匹配，只覆盖历史高频问法）
│   ├── generation.py   # 生成后端：HF / vLLM(含 LoRA) / OpenAI 兼容接口（可直接测项目三的线上服务）
│   └── io_utils.py
├── scripts/
│   ├── 00_convert_public.py   # 公开数据集(JDDC/问大家 FAQ 等) -> alpaca 格式
│   ├── 01_build_sft_data.py   # SOP 模板扩写 + 公开数据合并 + 可选 GPT 改写 -> 原始 12000 条 + 留出测试集 400 条
│   ├── 02_clean_sft_data.py   # 清洗 -> ~5600 条 SFT 数据，输出每条规则的过滤统计
│   ├── 03_train_sft.py        # TRL SFTTrainer + PEFT LoRA/QLoRA（rank / target_modules / QLoRA 消融）
│   ├── 04_gen_candidates.py   # SFT 模型对每个 query 采样 2-3 个回答
│   ├── 05_build_pref_data.py  # 自动打分 + 人工标注队列 -> chosen/rejected；支持噪声注入实验
│   ├── 06_train_dpo.py        # TRL DPOTrainer（SFT adapter 作为 reference，QLoRA 单卡可跑）
│   ├── 07_evaluate.py         # 统一评测入口（rule / reference / hf / vllm / openai）
│   ├── 08_badcase_analysis.py # 失败案例 8 类分桶 + 两模型对比（修复/退化/胜平负）
│   ├── 09_merge_lora.py       # 合并 LoRA -> 给项目三做 vLLM 部署和 AWQ/GPTQ 量化
│   ├── 10_summarize.py        # 汇总所有实验为 markdown 表
│   └── run_all.sh             # 一键跑全部实验（可分阶段：data/base/ablation/dpo/report）
├── configs/            # DeepSpeed ZeRO-2、accelerate、LLaMA-Factory 对照配置
├── data/               # dataset_info.json（LLaMA-Factory 可直接读取 sft/ dpo/ 目录）
└── results/            # 评测结果、bad case 报告
```

## 快速开始

```bash
conda create -n ecs python=3.10 -y && conda activate ecs
pip install -r requirements.txt
export BASE_MODEL=Qwen/Qwen3-8B        # 国内可先 export HF_ENDPOINT=https://hf-mirror.com
                                       # 或 modelscope download --model Qwen/Qwen3-8B --local_dir ./models/Qwen3-8B

# 1) 数据（CPU 即可，1 分钟）
python scripts/01_build_sft_data.py --n_raw 12000 --n_test 400
python scripts/02_clean_sft_data.py
python scripts/07_evaluate.py --backend rule --exp_name baseline_rule      # 线上规则 baseline

# 2) SFT（单卡 24GB，QLoRA r=16 约 1.5~2.5h）
python scripts/03_train_sft.py --qlora --lora_r 16 --target_modules q_proj,v_proj --exp_name sft_qlora_r16
python scripts/07_evaluate.py --backend vllm --adapter outputs/sft_qlora_r16 --exp_name sft_qlora_r16

# 3) DPO
python scripts/04_gen_candidates.py --adapter outputs/sft_qlora_r16 --n_queries 2500 --n 3
python scripts/05_build_pref_data.py --mode candidates --n_pairs 2000
python scripts/06_train_dpo.py --qlora --beta 0.1 --exp_name dpo_beta0.1
python scripts/07_evaluate.py --backend vllm --adapter outputs/dpo_beta0.1 --exp_name dpo_beta0.1

# 4) 分析与汇总
python scripts/08_badcase_analysis.py --pred results/preds_dpo_beta0.1.jsonl --compare results/preds_sft_qlora_r16.jsonl
python scripts/10_summarize.py
# 或者一键：bash scripts/run_all.sh [data|base|ablation|dpo|report]
```

没有 GPU 也能跑通：数据构造、清洗、规则 baseline、偏好数据构造（`--mode synthetic`）、bad case 分析。

### 无 NVIDIA GPU 的小规模复现（Qwen3-0.6B，Apple MPS / CPU）

```bash
modelscope download --model Qwen/Qwen3-0.6B --local_dir models/Qwen3-0.6B
bash scripts/run_mini_mps.sh      # zero-shot / r=4,16,64 / all-linear / DPO(β、噪声)，约 2.5 小时
bash scripts/run_mini_dpo_v2.sh   # on-policy 偏好对 + 降学习率 + RPO，修复第一轮 DPO 过优化
```

本机实测（M5 Pro，结果见 `results/SUMMARY.md`）：自动解决率 规则 28.0% → SFT r16 52.0% → SFT r64 66.8% → all-linear 75.0%；
第一轮合成偏好对 DPO 过优化降至 22.5%，第二轮 on-policy + RPO 恢复到 51.5%（与 SFT 持平，无净收益，详见文档）。

## 指标定义（面试必须讲清楚）

| 指标 | 定义 |
|---|---|
| 准确率 | 关键政策要点覆盖 ≥ 2/3，且未出现禁止表述（否定感知）、无复读；需转人工样本要求正确转人工 |
| 自动解决率 | 全部样本中“回答准确且没有把用户推给人工”的比例（转人工样本记为未自动解决） |
| 满意度 | 1-5 分规则代理分（共情、操作路径、要点覆盖、长度、复读惩罚）；可选 `--judge_model` 用 LLM 打分 |
| 复读率 | 6-gram 重复占比 > 0.3 或同一句重复 ≥ 3 次 |

测试集问法模板与训练集**完全隔离**（`sop.py` 中 `train_templates` / `test_templates`），并包含 `beyond_window`（超出政策时间窗）、`multi_turn_change`（突然变卦）、`multi_turn_follow`（追问）、`escalate`（需转人工）四个困难子集。

## 实验清单

| 实验 | 变量 | 关注指标 |
|---|---|---|
| rank 消融 | r = 4 / 16 / 64 | loss、准确率、显存、耗时 |
| target_modules | q_proj,v_proj vs all-linear | 效果、训练耗时 |
| QLoRA vs LoRA | 4bit NF4 vs bf16 | 精度损失、显存节省 |
| SFT-only vs SFT+DPO | 同一测试集 | 准确率、满意度、转人工率、胜平负 |
| β 消融 | 0.05 / 0.1 / 0.5 | 偏好强度 vs 回答多样性/过拟合 |
| 偏好噪声 | 10% / 20% / 30% 反转 | DPO 效果衰减曲线 |
