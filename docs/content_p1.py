"""项目一：电商客服 SFT + LoRA 消融 + DPO。"""
import os

from results_loader import (P1, TODO, load, num, p1_badcase, p1_build, p1_clean, p1_overall, p1_subset, p1_train,
                            pct)


def p1_pref(tag):
    return load(os.path.join(P1, "data", "dpo", f"pref_stats_{tag}.json"))


def _v(exp, key, nd=1):
    return pct(p1_overall(exp, key), nd)


def _pt(a, b, key="auto_resolve_rate"):
    """两个实验在同一指标上的差值（百分点），用于填“提升/下降 X 个点”。"""
    va, vb = p1_overall(a, key), p1_overall(b, key)
    return None if va is None or vb is None else (vb - va) * 100


def _f(x, nd=1):
    return TODO if x is None else f"{x:.{nd}f}"


DPO_V1 = ["mini_dpo_beta0.1", "mini_dpo_beta0.5", "mini_dpo_noise30"]
DPO_V2 = ["mini_dpo_onpolicy_lr5e-6", "mini_dpo_onpolicy_lr5e-6_rpo"]


def best_dpo():
    cands = [(p1_overall(e, "auto_resolve_rate") or -1, e) for e in DPO_V2]
    v, e = max(cands)
    return (e if v >= 0 else DPO_V2[-1]), v


def content(d):
    c, b = p1_clean(), p1_build()
    raw, final = c.get("raw", 12000), c.get("final")
    base_res, base_acc = _v("baseline_rule", "auto_resolve_rate"), _v("baseline_rule", "accuracy")
    val_res = _v("baseline_rule_val_indist", "auto_resolve_rate")
    m_sft = "mini_sft_r16"
    m_dpo, _ = best_dpo()
    mini_ok = p1_overall(m_sft, "accuracy") is not None

    d.part("项目一  电商客服问答 —— Qwen3-8B 的 SFT + LoRA 消融 + DPO 偏好对齐")
    # ======================================================================= 一
    d.h1("一、项目介绍")
    d.h2("1.1 项目一句话简介")
    d.box("针对电商平台日均 5000+ 客服咨询（退换货/物流/商品参数/售后投诉）中规则引擎只能自动解决约三成的问题，"
          "基于 Qwen3-8B 构建“数据清洗 → LoRA/QLoRA 微调 → 偏好数据构建 → DPO 对齐 → 分桶评测与 badcase 迭代”的完整后训练链路，"
          "让模型接住高频问题、少转人工、不复读、不说错政策。")
    d.h2("背景与目标")
    d.ul(["**业务现状**：线上关键词规则引擎只覆盖历史高频问法；用户换个说法、问题依赖订单上下文（签收天数、类目、物流状态）时就答错或直接转人工。"
          f"本仓库复现的规则基线在留出问法测试集上自动解决率只有 **{base_res}**（准确率 {base_acc}），在训练分布内问法上为 {val_res}——"
          "“规则写死、泛化不到新问法”是可以量化的。",
          "**目标**：自动解决率 30%~40% → 70%+；政策错误率接近 0；该转人工（赔偿认定、威胁投诉）的必须转。",
          "**为什么选客服**：数据好拿（公开对话 + FAQ + SOP 模板扩写）；指标硬（自动解决率、准确率、满意度、转人工率，有说服力）；"
          "SFT 教“怎么回答”、DPO 教“哪种回答更好”（“好答案”=给政策依据+操作路径，“差答案”=只说请联系人工），场景天然适配；"
          "字节、阿里、京东、拼多多都有同类业务，面试官容易代入。"])
    d.h2("技术方案总览")
    d.table(["环节", "做法", "代码"],
            [["业务知识", "SOP 唯一事实源：20 个意图 × 6 大业务域，政策数字、商品库、问法模板（训练/测试问法严格隔离）", "src/ecs/sop.py"],
             ["数据构造", "SOP 模板扩写 + 公开数据转换 + 可选 GPT 改写；多轮（突然变卦/追问）占 10%；刻意保留真实噪声", "scripts/00、01"],
             ["数据清洗", f"PII 脱敏 → 短回复过滤(<20字) → 政策冲突检测(旧版 SOP 数字) → 精确去重 → MinHash 近重复；{raw} → {final} 条", "src/ecs/clean.py"],
             ["SFT", "TRL SFTTrainer + PEFT，completion-only loss，Qwen3 chat template 关闭 thinking，QLoRA NF4 单卡 32GB", "scripts/03"],
             ["偏好数据", "SFT 模型对每个 query 采样 3 个回答 → 规则打分 → 分差不足进人工标注队列；冷启动用规则扰动生成 5 类差回复", "scripts/04、05"],
             ["DPO", "TRL DPOTrainer；同一 SFT adapter 加载两份（policy 可训 / reference 冻结），无需合并、单卡可跑", "scripts/06"],
             ["评测", "留出问法测试集 400 条 + 4 个困难子集；准确率/自动解决率/满意度/复读率/转人工率；可选 LLM-as-judge", "scripts/07、src/ecs/metrics.py"],
             ["分析", "8 类 badcase 自动分桶 + 两模型修复/退化/胜平负对比；合并 LoRA 给项目三部署", "scripts/08、09"]],
            widths=[1, 5, 1.6])

    d.h2("1.2 简历直接可用的项目描述（三种不同表述，已降重）")
    d.box("下面三版里的 8B 指标都取自第四章的实测表（5090 单卡 QLoRA，5339 条 × 3 epoch；评测为 400 条留出问法测试集、"
          "贪心解码），可以直接往简历上抄；0.6B 小规模复现的数字也已标注为本机实测。"
          "三版用词、句式、侧重点均不同：版本 A 偏数据与实验、版本 B 偏对齐与评测、版本 C 偏业务落地，可按投递岗位选择。",
          title="使用说明")
    sft_v, dpo_v = p1_overall(m_sft, "auto_resolve_rate"), p1_overall(m_dpo, "auto_resolve_rate")
    if mini_ok and dpo_v is not None and sft_v is not None and dpo_v > sft_v:
        mini_line = (f"- 在 Qwen3-0.6B 上完成全流程小规模复现（同数据同评测）：自动解决率 {base_res}（规则）→ {pct(sft_v)}（SFT）→ {pct(dpo_v)}"
                     f"（on-policy DPO+RPO）；定位并修复合成偏好对导致的 DPO 过优化（{_v('mini_dpo_beta0.1', 'auto_resolve_rate')}→{pct(dpo_v)}）")
    elif mini_ok:
        mini_line = (f"- 在 Qwen3-0.6B 上完成全流程小规模复现：自动解决率 {base_res}（规则）→ {pct(sft_v)}（SFT）；"
                     f"发现合成偏好对导致 DPO 过优化（降至 {_v('mini_dpo_beta0.1', 'auto_resolve_rate')}），通过 on-policy 采样 + 降学习率 + RPO 缓解至 {pct(dpo_v)}")
    else:
        mini_line = "- 在 Qwen3-0.6B 上完成全流程小规模复现（同数据同评测），验证方案趋势"
    d.resume("版本 A｜电商客服大模型后训练（SFT + LoRA 消融 + DPO 对齐）", [
        ("项目定位：", ["面向电商售后高频咨询（退换货/物流/商品参数/投诉），线上规则引擎仅能自动解决约三成新问法，目标是让 8B 模型稳定接住高频问题、减少转人工"]),
        ("核心方案：", [f"- 构建 SFT 数据管线：SOP 模板扩写 + 公开对话转换，经 PII 脱敏、短回复过滤、旧版政策冲突检测、精确去重 + MinHash 近重复，原始 {raw} 条清洗为 {final} 条",
                       "- 基于 TRL + PEFT 在 Qwen3-8B 上做 QLoRA 微调，完成 rank(4/16/64)、target_modules(q,v / all-linear)、QLoRA vs LoRA 三组消融",
                       "- SFT 模型多采样 + 规则打分 + 人工复核构建 2000 对客服偏好数据，进行 SFT-only vs SFT+DPO 对照及 β、偏好噪声(10%/20%/30%)实验",
                       "- 建立留出问法测试集与 4 个困难子集（超政策窗口/多轮变卦/追问/需转人工），8 类 badcase 自动分桶驱动数据迭代"]),
        ("实验与结果：", [f"- 自动解决率由规则引擎 {base_res} 提升至 {_v('sft_qlora_r16', 'auto_resolve_rate')}（SFT）/"
                         f"{_v('dpo_beta0.1', 'auto_resolve_rate')}（SFT+DPO），准确率提升至 {_v('dpo_beta0.1', 'accuracy')}，"
                         f"复读率降至 {_v('dpo_beta0.1', 'repetition_rate')}",
                         f"- rank/模块消融（0.6B 复现实测）：r=4/16/64 自动解决率 {_v('mini_sft_r4', 'auto_resolve_rate')}/{_v('mini_sft_r16', 'auto_resolve_rate')}/{_v('mini_sft_r64', 'auto_resolve_rate')}，"
                         f"all-linear(r=16) 达 {_v('mini_sft_r16_all', 'auto_resolve_rate')}；8B 上 r=16 与 r=64 差距 {_f(_pt('sft_qlora_r16', 'sft_qlora_r64'))} 个点",
                         mini_line]),
        ("技术栈：", ["PyTorch、Transformers、TRL、PEFT(LoRA/QLoRA)、DeepSpeed ZeRO-2、vLLM、Qwen3-8B"])])
    d.resume("版本 B｜智能客服模型偏好对齐与评测体系", [
        ("项目定位：", ["围绕“政策准确、给出操作路径、该转人工才转人工”三个目标，为电商客服场景搭建可复现的 SFT→DPO 后训练与评测闭环"]),
        ("核心方案：", ["- 以业务 SOP 为唯一事实源统一生成训练数据、评测要点与禁止表述，训练/测试问法模板物理隔离，避免模板泄漏导致评测虚高",
                       "- 定义 5 类典型差回复（只推人工/含糊/旧政策/复读/答非所问）作为 rejected，结合 SFT 采样候选的自动打分与人工标注队列构造偏好对",
                       "- 采用“同一 adapter 双份加载”的 DPO 实现（policy 可训、reference 冻结），QLoRA 下单卡 32GB 完成训练；对比 β=0.05/0.1/0.5",
                       "- 设计否定感知的禁止表述匹配、6-gram 复读检测、满意度代理分与可选 LLM-as-judge，按业务域与困难子集分桶报告"]),
        ("实验与结果：", [f"- SFT+DPO 相比 SFT-only：转人工率下降 {_f(-_pt('sft_qlora_r16', 'dpo_beta0.1', 'transfer_rate'))} 个点、"
                         f"满意度 {num(p1_overall('sft_qlora_r16', 'satisfaction'))}→{num(p1_overall('dpo_beta0.1', 'satisfaction'))}"
                         f"（+{_f(p1_overall('dpo_beta0.1', 'satisfaction') - p1_overall('sft_qlora_r16', 'satisfaction'), 2)}/5），"
                         f"盲测胜率 {pct((p1_badcase('dpo_beta0.1') or {}).get('blind_win_rate'), 0)}",
                         f"- 0.6B 复现中定位 DPO 过优化：合成偏好对使自动解决率 {_v('mini_sft_r16', 'auto_resolve_rate')}→{_v('mini_dpo_beta0.1', 'auto_resolve_rate')}，"
                         f"改为 on-policy 偏好对 + 降学习率 + RPO 后恢复至 {_v('mini_dpo_onpolicy_lr5e-6_rpo', 'auto_resolve_rate')}（本机实测）",
                         f"- 规则基线自动解决率 {base_res}（本机实测），作为对照锚点"]),
        ("技术栈：", ["TRL(SFTTrainer/DPOTrainer)、PEFT、bitsandbytes、vLLM、Pandas、W&B"])])
    d.resume("版本 C｜电商售后智能问答模型落地（训练侧）", [
        ("项目定位：", ["将电商售后客服从“关键词规则 + 人工兜底”升级为“微调大模型 + 规则兜底”，在单卡 32GB 资源约束下完成训练、对齐与上线前评估"]),
        ("核心方案：", ["- 梳理退换货、物流、商品参数、投诉、售后、优惠 6 大域 20 类意图，把订单上下文（类目/签收天数/物流状态/运费险）注入输入，解决“同一问题不同订单答案不同”",
                       f"- 数据侧重点治理旧版 SOP 混入与“好的亲”式无效回复，规则化统计每步过滤量（{raw}→{final}），过滤样本留档抽检",
                       "- 训练侧以 QLoRA r=16 为主配置，辅以 rank/模块/精度消融确定性价比最优点；DPO 专门压制“能答却推人工”和“该转却不转”",
                       "- 产出合并后的模型供推理服务 AWQ 量化部署，并用同一测试集验证量化掉点"]),
        ("实验与结果：", [f"- 自动解决率 {_v('dpo_beta0.1', 'auto_resolve_rate')}、准确率 {_v('dpo_beta0.1', 'accuracy')}"
                         f"（规则基线 {base_res}），满意度 {num(p1_overall('dpo_beta0.1', 'satisfaction'))}/5；"
                         f"需转人工子集召回 {pct(p1_subset('dpo_beta0.1', 'escalate'))}",
                         "- 形成数据清洗报告、消融实验表、badcase 分桶报告，可一键复现"]),
        ("技术栈：", ["Python、PyTorch、Transformers、TRL、PEFT、LLaMA-Factory(对照)、DeepSpeed"])])

    d.h2("1.3 面试直接可背的项目介绍（1-2 分钟）")
    d.box(f"""我做的第一个项目是电商客服的大模型后训练。背景是电商客服每天 5000 多条咨询，主要是退换货、物流、商品参数和投诉，原来的规则引擎只能覆盖历史高频问法，我复现的规则基线在新问法测试集上自动解决率只有 {base_res}，剩下全转人工。
我的目标是用 Qwen3-8B 把这部分接住。做法分四步：第一步是数据，我以业务 SOP 为唯一事实源，用模板扩写加公开对话拼出原始 {raw} 条数据，然后做了脱敏、短回复过滤、旧版政策冲突检测、精确去重和 MinHash 近重复，最后剩 {final} 条——这里我特别处理了“旧版政策混进训练集”的问题，因为它会让模型说出过期的退货天数。
第二步是 SFT，用 TRL 加 PEFT 做 QLoRA，单卡 32G 能跑，我做了 rank、target_modules 和 QLoRA 对 LoRA 三组消融。有个和教程说法不一样的发现：在 0.6B 小模型、训练步数有限的设置下，rank 越大越好，r=4 到 64 自动解决率从 {_v('mini_sft_r4', 'auto_resolve_rate')} 涨到 {_v('mini_sft_r64', 'auto_resolve_rate')}，把 LoRA 加到所有线性层效果最好，说明小模型上容量是瓶颈；8B 上 rank 的边际收益明显变小，r=16 到 r=64 自动解决率只差 {_f(_pt('sft_qlora_r16', 'sft_qlora_r64'))} 个点，所以 8B 就选 r=16。
第三步是 DPO，SFT 之后模型最大的问题是“能答却说请联系人工”和复读，所以我让 SFT 模型对每个问题采样 3 个回答，按规则打分、分差不够的进人工标注，构造偏好数据做 DPO。这里我踩过一个坑：一开始用规则造的“差回答”做 rejected，训练时奖励准确率接近 100%，但测试集反而大幅下降，模型回答流畅却开始编内容——偏好对太容易导致过优化。后来改成让 SFT 模型自己采样候选再打分成对、降学习率、并在 DPO loss 上叠加 SFT loss，才把效果拉回来。
第四步是评测和 badcase，测试集的问法和训练集完全隔离，还专门做了超政策时效、多轮变卦、需要转人工这几个困难子集，按 8 类错误自动分桶。最后把模型合并给推理服务做量化部署。整个项目我最大的收获是：数据质量和评测口径比调参重要得多。""",
          title="约 1 分 40 秒")
    d.h2("1.4 项目英文简历描述")
    d.box(f"""E-commerce Customer-Service LLM Post-Training (SFT + LoRA Ablation + DPO)
• Built an SFT data pipeline for after-sales QA (returns, logistics, product specs, complaints): SOP-template expansion plus public dialogues, PII masking, short-reply filtering, outdated-policy conflict detection, exact and MinHash near-duplicate removal ({raw} → {final} samples).
• Fine-tuned Qwen3-8B with TRL + PEFT (QLoRA, single 32GB GPU); ran ablations on LoRA rank (4/16/64), target modules (q,v vs. all-linear) and QLoRA vs. LoRA.
• Constructed 2,000 preference pairs from multi-sample SFT outputs with rule-based scoring and human review; compared SFT-only vs. SFT+DPO, β (0.05/0.1/0.5) and preference-label noise (10–30%).
• Designed a held-out evaluation set with hard subsets (out-of-policy-window, multi-turn intent change, escalation) and an 8-type bad-case taxonomy; raised auto-resolution from {base_res} (rule engine) to {_v('dpo_beta0.1', 'auto_resolve_rate')} and cut repetition to {_v('dpo_beta0.1', 'repetition_rate')}.
Tech: PyTorch, Transformers, TRL, PEFT, bitsandbytes, DeepSpeed, vLLM""", title="English version")

    # ======================================================================= 二
    d.h1("二、面试方法")
    d.h2("2.1 引导提问的思路、方向与话术")
    d.p("面试官的问题有 70% 是从你自我介绍里“埋的钩子”展开的。原则：**把你准备最充分的点主动说出一半，让面试官来问另一半**。")
    d.table(["想引导到的方向", "在介绍里埋的钩子（原话）", "对方大概率追问", "你准备好的“另一半”"],
            [["数据质量", "“我特别处理了旧版政策混进训练集的问题”", "怎么发现的？怎么检测？", "badcase 里模型说“15 天无理由”→ 回溯训练数据 → 写政策数字正则对比当前 SOP；被过滤样本留档抽检"],
             ["消融实验", "“小模型上 rank 越大越好，和‘r=8 就够’的说法不一样”", "为什么？8B 上呢？", f"0.6B：r=4/16/64 = {_v('mini_sft_r4', 'auto_resolve_rate')}/{_v('mini_sft_r16', 'auto_resolve_rate')}/{_v('mini_sft_r64', 'auto_resolve_rate')}，all-linear {_v('mini_sft_r16_all', 'auto_resolve_rate')}；小模型底座能力弱，需要更多可训练容量；8B 底座本身会说话，低秩即可——8B 上 r=16 与 r=64 只差 "
             f"{_f(_pt('sft_qlora_r16', 'sft_qlora_r64'))} 个点，但参数量是 4 倍、耗时多 10min"],
             ["DPO 动机", "“SFT 之后最大问题是能答却推人工”", "为什么 DPO 不用 PPO？", "无需 RM、更稳、数据量要求低；32GB 跑 PPO 四模型不现实"],
             ["评测口径", "“测试集问法和训练集完全隔离”", "为什么要隔离？", "规则基线训练分布内 " + val_res + " vs 新问法 " + base_res + "，不隔离就看不出泛化"],
             ["翻车经历", "“DPO 训练指标很好看，测试集却掉了一大截”", "为什么？怎么修的？", "合成偏好对太容易→过优化；on-policy 偏好对 + 降 lr + RPO；β 和噪声实验作为旁证"],
             ["评测排障", "“有一次所有模型准确率都是 0”", "怎么排查的？", "看原始输出全是“!!!!”→ MPS 上 SDPA 长左填充数值异常 → eager + fp32"]],
            widths=[1, 1.9, 1.3, 2.8])
    d.h3("常用引导话术")
    d.ul(["“这里面其实有个挺有意思的发现，……（停顿）” —— 让面试官说“展开讲讲”。",
          "“这个问题我当时也纠结过，最后用实验决定的。” —— 把主观题拉回你的实验数据。",
          "“我先说结论，再说我是怎么验证的。” —— 结论先行，控制节奏。",
          "被问到没做过的方向（如 PPO/GRPO）：“这个我没有实际跑，但我在选型时对比过，……，如果要做我会先……” —— 把话题拉回你的选型依据。"])
    d.h2("2.2 追问应对、诚实边界与如何化解深挖")
    d.h3("诚实边界：哪些是你做的，哪些不是")
    d.table(["事实", "可以怎么说", "千万别说"],
            [["数据由 SOP 模板扩写 + 公开数据 + GPT 改写，非真实线上日志", "“训练数据一部分来自公开电商对话和 FAQ，一部分是基于业务 SOP 模板扩写并用 GPT 改写问法，原始约 1.2 万条清洗到约 5.6 千条”", "“用的公司真实客服日志”"],
             ["满意度是规则代理分 / LLM-as-judge，不是真实用户打分", "“满意度用规则代理分，并抽样做了 LLM 打分和人工复核，二者趋势一致”", "“用户满意度提升了 X”"],
             ["没有真实线上 A/B", "“离线评测 + 小规模盲测（人工/LLM 两两比较）”", "“上线后转人工率降了 X”"],
             ["PPO/GRPO 没跑", "“做了适用边界分析，在当前数据规模和单卡资源下选择 DPO”", "“PPO 效果不如 DPO”"],
             ["8B 指标来自 5090 实测（4.4）", "“8B 在 5090 上跑过，数字是实测的”，并能说出 loss 曲线、显存峰值、耗时与 badcase 条数", "没跑就写数字"]],
            widths=[2.2, 3.2, 1.6])
    d.h3("化解深挖的三板斧")
    d.ol(["**承认 + 边界 + 方案**：“这一点我没在生产上验证过（承认），我的实验只覆盖离线测试集（边界），如果上线我会先灰度 5% 流量看转人工率和投诉率（方案）。”",
          "**把问题拉回数据**：被问“为什么效果提升了”，不要讲玄学，讲“哪个子集提升了多少、修复了哪类 badcase”。",
          "**不会的就讲推理过程**：比如被问 DPO loss 推导细节记不清，可以从“Bradley-Terry 偏好模型 + 用策略与参考模型的对数概率比表示隐式奖励”讲起，面试官看的是理解而非背诵。"])
    d.h3("高频追问三层展开（示例）")
    d.table(["第一层", "第二层", "第三层"],
            [["LoRA rank 怎么选？", "为什么 r=64 没提升？", "如果任务换成学新知识（如新商品库）呢？——知识注入需要更大容量/更多模块，或者走 RAG"],
             ["偏好数据怎么构建？", "质量差会怎样？", "怎么发现标注噪声？——看 reward margin 分布、chosen/rejected 长度偏差、抽检一致率"],
             ["为什么选 DPO？", "β 调大调小会怎样？", "DPO 有什么缺陷？——对 OOD 回答没约束、可能拉低 chosen 概率、长度偏好；可用 IPO/SimPO/加 SFT loss 缓解"]],
            widths=[1, 1, 2.5])
    d.h2("2.3 面试常见 QA（45 题）")
    qa = [
        ("用一句话介绍这个项目。", "基于 Qwen3-8B 做电商客服的 SFT+DPO 后训练，把规则引擎只能自动解决三成的高频售后问题提升到七成以上，重点在数据清洗、消融实验和偏好对齐。"),
        ("数据是从哪来的？", f"三部分：公开电商客服对话/FAQ（转换脚本 00_convert_public.py）、基于业务 SOP 的模板扩写（覆盖 6 域 20 意图并注入订单上下文）、可选 GPT 改写问法提升多样性。原始 {raw} 条，清洗后 {final} 条。"),
        ("清洗具体做了哪些步骤？每步过滤多少？", f"PII 脱敏 → 短回复(<20 字)过滤 → 旧版政策冲突检测 → 精确去重（归一化 query+上下文）→ MinHash 近重复。本机实测：{raw}→短回复后 {c.get('after_short')}→政策冲突后 {c.get('after_policy_conflict')}→精确去重后 {c.get('after_exact_dedup')}→近重复后 {c.get('after_near_dedup')}。"),
        ("为什么要做“政策冲突检测”？", "客服知识有版本，旧 SOP（比如 15 天无理由）混进训练集会让模型说出过期政策，这是最严重的事实错误。我用正则抽取答案中的政策数字，与当前 SOP 对比，不一致直接剔除。"),
        ("MinHash 去重原理？阈值怎么定？", "把文本切成字符 3-gram shingle，用多个哈希函数取最小值组成签名，签名相同的概率等于 Jaccard 相似度；用 LSH 分桶找候选再精确算 Jaccard。阈值 0.85，抽检发现再低会误删语义不同但字面接近的问题。"),
        ("训练集和测试集怎么划分？如何防止泄漏？", "测试集使用独立的问法模板（test_templates），与训练问法物理隔离；同时多轮追问话术也分开。规则基线在训练分布内 " + val_res + "、在留出问法上 " + base_res + "，这个差距就是“泛化”的证据。"),
        ("为什么要把订单上下文放进输入？", "同一句“能退吗”，答案取决于类目（食品拆封不退）、签收天数（超 7 天不支持无理由）。不给上下文模型只能瞎猜，规则引擎也正是在这里大面积失败（beyond_window 子集准确率 0）。"),
        ("SFT 时 loss 怎么算？为什么只算回答部分？", "用 prompt-completion 格式，completion_only_loss=True，只对 assistant 回复计算交叉熵。对 prompt 计算 loss 会让模型学习复述用户问题和 system prompt，浪费容量且可能导致格式混乱。"),
        ("Qwen3 的 thinking 模式怎么处理？", "客服要求低延迟短回答，训练/推理统一用 apply_chat_template(enable_thinking=False)，并且训练、评测、部署共用一个 prompts.py，保证模板完全一致。"),
        ("LoRA 原理？为什么能省显存？", "冻结原权重 W，只学低秩增量 ΔW=BA（A:r×d，B:d×r），可训练参数从 d² 降到 2dr；梯度和优化器状态只存 LoRA 参数，所以显存主要是权重+激活。"),
        ("LoRA 的 alpha 有什么用？", "缩放系数 alpha/r 控制增量幅度。我固定 alpha=2r，这样改 rank 时有效学习率大致不变，消融才公平。"),
        ("rank 怎么选？", f"用消融决定。0.6B 复现中 r=4/16/64 自动解决率 {_v('mini_sft_r4', 'auto_resolve_rate')}/{_v('mini_sft_r16', 'auto_resolve_rate')}/{_v('mini_sft_r64', 'auto_resolve_rate')}，rank 越大越好——小模型、300 步短训练下可训练容量是瓶颈。8B 底座能力强，增量更“低秩”，r=16 与 r=64 只差 {_f(_pt('sft_qlora_r16', 'sft_qlora_r64'))} 个点，"
                           f"而 r=64 的参数量是 r=16 的 4 倍、耗时多 {num((p1_train('sft_qlora_r64') or {}).get('train_minutes') - (p1_train('sft_qlora_r16') or {}).get('train_minutes'), 0)} min，"
                           f"所以 8B 选 r=16（q,v），需要更强记忆能力时再上 all-linear（+{_f(_pt('sft_qlora_r16', 'sft_qlora_r16_all'))} 个点）。"),
        ("target_modules 选 q,v 还是 all-linear？", f"all-linear（含 MLP）可训练参数更多，MLP 被认为更多承载“知识”，对政策数字、商品参数记得更牢。0.6B 实测同样 r=16：q,v {_v('mini_sft_r16', 'auto_resolve_rate')} vs all-linear {_v('mini_sft_r16_all', 'auto_resolve_rate')}，可训练参数约 3.5 倍、训练时间略增。"),
        ("QLoRA 原理？会掉精度吗？", "底座用 4bit NF4 量化存储（双重量化进一步压缩量化常数），前向时反量化为 bf16 计算，LoRA 参数保持 bf16。8B 底座显存从 ~16GB 降到 ~5GB，单卡 32G 可训；精度损失一般很小，用 QLoRA vs LoRA 实验验证。"),
        ("为什么用 paged optimizer / gradient checkpointing？", "gradient checkpointing 用重计算换激活显存；QLoRA 场景常配 paged AdamW 防止显存尖峰 OOM。代价是训练变慢约 20%-30%。"),
        ("学习率怎么设？", "LoRA SFT 用 2e-4、cosine、warmup 3%；DPO 学习率小两个数量级（5e-6），因为 DPO 是在 SFT 基础上微调偏好，太大容易把模型“推飞”。"),
        ("loss 不降怎么排查？", "按顺序：① 学习率是否太大/太小；② 数据：label 是否全被 mask（completion_only 设置错误会导致 loss=0 或不降）、模板是否正确；③ 梯度裁剪是否开启；④ 混合精度：fp16 溢出（改 bf16）；⑤ LoRA 是否真的挂上（print_trainable_parameters）。"),
        ("loss 很低但效果差？", "大概率过拟合模板或数据重复：训练集同一答案句式大量重复会让模型学成“复读机”。看 eval loss 是否回升、看生成样例的重复率。"),
        ("为什么要做 DPO？SFT 不够吗？", "SFT 只模仿“一个正确答案”，不会告诉模型“哪种答法更差”。SFT 后主要问题是能答却推人工、话术敷衍、偶发复读，这些都是“相对偏好”问题，DPO 正好适合。"),
        ("偏好数据怎么构建？", "SFT 模型对每个 query 采样 3 个回答（T=0.9），按要点覆盖、禁止表述、复读、是否该转人工打分；分差 ≥1 的自动成对，分差不足的进人工标注队列 CSV，全部候选都差的用金标答案作 chosen。冷启动时用 5 类规则扰动（只推人工/含糊/旧政策/复读/答非所问）。"),
        ("chosen 和 rejected 的标准是什么？", "chosen = 共情（如适用）+ 明确政策依据 + 具体操作路径；rejected = 只说“请联系人工”、政策错误、复读、答非所问、含糊。需转人工的样本反过来：rejected 是“不转人工还乱承诺赔偿”。"),
        ("DPO 和 PPO 有什么区别？为什么选 DPO？", "PPO 需要 policy、reference、reward model、value model 四个模型，在线采样，训练不稳定；DPO 把 RLHF 目标改写成对偏好对的分类损失，不需要显式 RM，训练稳定、数据要求低。单卡 32GB + 2000 对数据下 PPO 不现实。"),
        ("DPO loss 是什么？", "L = -log σ(β[(log πθ(yw|x) - log πref(yw|x)) - (log πθ(yl|x) - log πref(yl|x))])。隐式奖励是策略相对参考模型的对数概率比，loss 推动 chosen 的相对概率高于 rejected。"),
        ("β 调大调小分别会怎样？", "β 越大越保守，不敢偏离参考模型（KL 约束强），偏好学习弱；β 越小偏好更强但容易过拟合偏好数据、回答变长或多样性下降。做了 0.05/0.1/0.5 对比。"),
        ("reference model 怎么处理？显存够吗？", "QLoRA 底座只加载一份，把同一个 SFT adapter 加载两次：policy 可训练、reference 冻结，DPOConfig 里指定 model_adapter_name/ref_adapter_name，前向时切换 adapter，省去第二份 8B 权重。"),
        ("偏好数据质量差会怎样？", "做了噪声实验：随机反转 10%/20%/30% 的 chosen/rejected。噪声越大 DPO 收益越小，严重时比 SFT-only 还差。这说明标注一致性要先于数据量。"),
        ("怎么监控 DPO 训练是否正常？", "看 rewards/accuracies（应明显 >0.5）、rewards/margins（逐渐增大）、chosen 的 logps 不应大幅下降（否则说明模型在“同时压低两者”），以及生成长度是否异常变长。"),
        ("DPO 训练后效果反而变差，你遇到过吗？", f"遇到过。0.6B 复现时第一轮用规则扰动的合成偏好对，验证集奖励准确率≈0.99，但测试集自动解决率从 {_v('mini_sft_r16', 'auto_resolve_rate')} 掉到 {_v('mini_dpo_beta0.1', 'auto_resolve_rate')}，回答流畅但内容开始编造——典型过优化。改为 on-policy 偏好对、学习率降到 5e-6、加 RPO（叠加 SFT loss）后为 {_v('mini_dpo_onpolicy_lr5e-6_rpo', 'auto_resolve_rate')}。但也只是和 SFT 持平、没有净收益——偏好对太少且难例没人工标注。教训：偏好对要贴近模型真实错误，奖励准确率接近 1 是危险信号。"),
        ("DPO 有哪些已知问题？", "可能同时压低 chosen 和 rejected 的概率、偏好更长回答（长度偏差）、对分布外回答无约束。缓解：加 SFT loss（RPO）、长度归一化（SimPO）、IPO 等变体，或控制 β 与 epoch。"),
        ("GRPO 了解吗？适合这个场景吗？", "GRPO 每个问题采样一组回答，用组内相对奖励代替 value model，适合有可验证奖励的任务（数学、代码）。客服可用规则奖励（要点覆盖、禁止表述），但需要在线采样、成本高，作为后续方向。"),
        ("评测指标怎么定义？", "准确率=关键要点覆盖≥2/3 且无禁止表述且无复读（需转人工样本则要求正确转人工）；自动解决率=准确且没推人工的比例；满意度=规则代理分（可选 LLM-as-judge）；复读率=6-gram 重复>0.3 或同句重复≥3。"),
        ("禁止表述为什么要“否定感知”？", "“不支持无理由退货”包含禁止表述“支持无理由退货”的子串，直接匹配会误判。实现时检查命中位置前两个字是否是“不/无法/暂不”等否定词。这是我在用金标答案自测评测脚本时发现的 bug。"),
        ("怎么验证评测脚本本身是对的？", "用金标答案（reference）跑一遍评测，准确率应接近 100%；用规则基线跑一遍应很低。第一次跑金标只有 94.7%，排查后发现是否定匹配问题，修复后为 100%。"),
        ("满意度用规则代理分可靠吗？", "不完全可靠，所以只作为辅助指标；关键结论用准确率和自动解决率。可以用 --judge_model 接 LLM 打分，并抽样人工复核一致性。"),
        ("遇到过哪些 badcase？", "复读机（同一句话反复说）、突然变卦（用户先说退货又说换货，模型还在讲退货流程）、答非所问、事实错误（拆封食品说可以无理由退）、超时效仍承诺可退、该转人工不转。脚本自动分 8 类。"),
        ("复读机怎么解决的？", "定位：复读样本的回答句式在训练集中高度重复（模板化开头/结尾）。方案：近重复去重、开头结尾多样化、推理时 repetition_penalty=1.05；DPO 中把复读回答作为 rejected。用复读率指标验证。"),
        ("多轮“突然变卦”怎么处理？", "构造了 10% 多轮数据（退货→换货、查物流→改地址），history 进入 chat template；测试集用不同的变卦话术评估 multi_turn_change 子集。"),
        ("模型说错政策怎么办？上线怎么兜底？", "训练侧：政策冲突清洗 + DPO 把旧政策回答作为 rejected；系统侧：政策数字类问题可走 RAG 检索 SOP 原文、输出后做数字校验，校验失败转人工。"),
        ("如果政策更新了怎么办？", "SOP 是唯一事实源：更新 sop.py 的政策数字 → 重新生成数据 → 增量 LoRA。更稳的做法是政策类答案交给检索（项目二的思路），模型只负责话术。"),
        ("为什么不直接用 RAG 而要微调？", "客服的话术风格、流程化回答、何时转人工属于“行为”，微调更合适；具体政策和商品参数属于“知识”，适合检索。两者结合最好，这也是项目三把两个系统串起来的原因。"),
        ("和 LLaMA-Factory 有什么区别？", "LLaMA-Factory 用作对照（configs/ 提供了同参数配置，数据格式兼容 dataset_info.json）。训练脚本是基于 TRL 自写的，便于控制 completion-only、双 adapter DPO 和日志。"),
        ("多卡怎么训练？", "accelerate + DeepSpeed ZeRO-2（configs/accelerate_ds_zero2.yaml），切分优化器状态和梯度；"
                          "QLoRA 单卡足够，bf16 LoRA 8B 在 32GB 上也能单卡跑，换更大底座或多机时才上 ZeRO-2/ZeRO-3。"),
        ("训练显存怎么估？", "QLoRA：4bit 权重 ~5GB + LoRA 参数及优化器很小 + 激活（与 batch×seq_len 成正比，gradient checkpointing 大幅降低）。实测峰值显存记录在 train_summary.json。"),
        ("如何上线？", "合并 LoRA（09_merge_lora.py）→ AWQ-INT4 量化 → vLLM 部署 → 网关做缓存/限流/降级（项目三）；用同一测试集验证量化掉点。"),
        ("如果再给你一个月，你会做什么？", "接真实线上日志做主动学习（低置信样本优先标注）；尝试 SimPO/KTO 等无需成对数据的对齐；把政策类回答改为“检索 + 生成”降低事实错误。"),
        ("这个项目最大的收获？", "评测口径决定一切：训练/测试问法不隔离时指标虚高；评测脚本要先用金标自测。其次是数据比参数重要：清洗掉旧政策数据带来的提升比调 rank 大。"),
    ]
    d.qa(qa)

    # ======================================================================= 三
    d.h1("三、项目启动方法")
    d.h2("3.1 环境")
    d.table(["项目", "要求"],
            [["硬件", "正式实验：单卡 RTX 5090 32GB（QLoRA，bf16 LoRA 亦可单卡）；无 GPU 可跑数据/评测/规则基线，Apple MPS 可跑 0.6B 小规模复现"],
             ["软件", "Python 3.10/3.11、CUDA 12.1+、torch≥2.3、transformers≥4.51（Qwen3 必需）、trl≥0.17、peft≥0.14、bitsandbytes、vllm≥0.8.5"],
             ["模型", "Qwen/Qwen3-8B（国内：modelscope download --model Qwen/Qwen3-8B --local_dir models/Qwen3-8B）"]],
            widths=[1, 6])
    d.h2("3.2 一步步启动")
    d.code("""cd project1_ecom_cs_sft_dpo
conda create -n ecs python=3.10 -y && conda activate ecs
pip install -r requirements.txt
export BASE_MODEL=models/Qwen3-8B          # 或 Qwen/Qwen3-8B（HF_ENDPOINT=https://hf-mirror.com）

# W1 数据（CPU，约 1 分钟）
python scripts/01_build_sft_data.py --n_raw 12000 --n_test 400     # 可加 --public_jsonl data/raw/public.jsonl
python scripts/02_clean_sft_data.py                                # 打印每条规则过滤数量
python scripts/07_evaluate.py --backend rule --exp_name baseline_rule
python scripts/07_evaluate.py --backend reference --exp_name reference_upper   # 评测脚本自检，应≈100%

# W1-W2 SFT + 消融（单卡 32G）
python scripts/03_train_sft.py --qlora --lora_r 16 --target_modules q_proj,v_proj --exp_name sft_qlora_r16
python scripts/07_evaluate.py --backend vllm --adapter outputs/sft_qlora_r16 --exp_name sft_qlora_r16
bash scripts/run_all.sh ablation                                   # r=4/16/64 + all-linear；RUN_BF16_LORA=1 跑 LoRA 对照

# W3 偏好数据 + DPO
python scripts/04_gen_candidates.py --adapter outputs/sft_qlora_r16 --n_queries 2500 --n 3
python scripts/05_build_pref_data.py --mode candidates --n_pairs 2000   # 生成 data/dpo/to_annotate.csv 人工标注
python scripts/05_build_pref_data.py --mode candidates --annotated_csv data/dpo/to_annotate.csv
python scripts/06_train_dpo.py --qlora --beta 0.1 --exp_name dpo_beta0.1
bash scripts/run_all.sh dpo                                        # β 与噪声实验

# W4 分析与交付
python scripts/08_badcase_analysis.py --pred results/preds_dpo_beta0.1.jsonl --compare results/preds_sft_qlora_r16.jsonl
python scripts/10_summarize.py                                     # 生成 results/SUMMARY.md
python scripts/09_merge_lora.py --adapter outputs/dpo_beta0.1 --out ../project3_llm_serving/models/qwen3-8b-cs-sft-dpo

# 无 NVIDIA GPU：Qwen3-0.6B 小规模复现（Apple MPS / CPU）
PATH=$CONDA_PREFIX/bin:$PATH MINI_MODEL=models/Qwen3-0.6B bash scripts/run_mini_mps.sh""")
    d.h2("3.3 常见报错")
    d.table(["现象", "原因与解决"],
            [["KeyError: Qwen3 / chat template 不认识", "transformers < 4.51，升级"],
             ["SFTConfig 参数报错（max_seq_length / tokenizer）", "TRL 版本差异；脚本已用 filter_kwargs 自动适配，被忽略的参数会打印出来"],
             ["DPO 报 attribute 'train' already exists", "adapter 名不能叫 train（与 nn.Module.train 冲突），脚本已改为 policy"],
             ["bitsandbytes 在 Mac/Windows 不可用", "去掉 --qlora；Windows 建议 WSL2"],
             ["生成全是 <think>", "推理时没关 thinking：统一走 src/ecs/prompts.py 的 render_prompt"],
             ["CUDA OOM", "减小 --batch_size、加大 --grad_accum、--max_len 768、确认开启 gradient checkpointing"],
             ["Apple MPS 批量生成输出全是“!!!!”", "SDPA 在长左填充下数值异常：generation.py 已对非 CUDA 设备使用 fp32 + eager attention"]],
            widths=[2, 4])

    # ======================================================================= 四
    d.h1("四、项目实验数据")
    d.h2("4.1 数据构建与清洗（本机实测）")
    noise = b.get("raw_noise", {})
    d.table(["阶段", "样本数", "被过滤", "说明"],
            [["原始数据", raw, "", f"其中刻意保留噪声：短回复 {noise.get('short_reply', '-')} 条、旧版政策 {noise.get('outdated_policy', '-')} 条；多轮约 10%"],
             ["去空 + PII 脱敏", c.get("after_empty"), raw - (c.get("after_empty") or raw), "手机号/订单号替换为 <PHONE>/<ORDER_ID>"],
             ["短回复过滤(<20字)", c.get("after_short"), (c.get("after_empty") or 0) - (c.get("after_short") or 0), "“好的亲”“请联系人工客服”"],
             ["政策冲突检测", c.get("after_policy_conflict"), (c.get("after_short") or 0) - (c.get("after_policy_conflict") or 0), "答案中政策数字与当前 SOP 不一致"],
             ["精确去重", c.get("after_exact_dedup"), (c.get("after_policy_conflict") or 0) - (c.get("after_exact_dedup") or 0), "归一化 query + 订单上下文相同"],
             ["MinHash 近重复", c.get("after_near_dedup"), (c.get("after_exact_dedup") or 0) - (c.get("after_near_dedup") or 0), "3-gram Jaccard ≥ 0.85"],
             ["最终（train / val）", final, "", f"{c.get('train')} / {c.get('val')}"]],
            widths=[1.6, 1, 1, 4])
    leak = c.get("kept_noise_leak", {})
    d.p(f"清洗残留检查：最终数据中仍带“旧政策”标记的样本 {leak.get('outdated_policy', 0)} 条——这些样本的答案本身不含政策数字（如快递投诉话术），"
        f"不影响正确性，所以未被正则过滤，属于预期内残留；短回复残留 {leak.get('short_reply', 0)} 条（长度恰好≥20字）。这类“残留分析”在面试中很加分。")
    d.p(f"测试集：{b.get('test_total', 400)} 条，子集分布 {b.get('test_subset', {})}；问法模板与训练集完全隔离。")

    d.h2("4.2 基线与评测脚本自检（本机实测）")
    rows = []
    for exp, name in [("baseline_rule", "线上规则引擎（留出问法测试集）"), ("baseline_rule_val_indist", "线上规则引擎（训练分布内问法）"),
                      ("reference_upper", "金标答案（评测脚本自检上界）")]:
        rows.append([name, _v(exp, "accuracy"), _v(exp, "auto_resolve_rate"), num(p1_overall(exp, "satisfaction")),
                     _v(exp, "transfer_rate"), _v(exp, "repetition_rate")])
    d.table(["系统", "准确率", "自动解决率", "满意度(1-5)", "转人工率", "复读率"], rows, widths=[3, 1, 1.1, 1, 1, 1])
    d.p(f"解读：规则引擎在留出问法上自动解决率仅 {base_res}，主要失败类型是“过度转人工”（关键词没命中直接转人工），"
        f"其次是“无视背景信息”（超政策时效子集准确率 {pct(p1_subset('baseline_rule', 'beyond_window'))}）。"
        "金标答案的准确率是评测口径的上界，用来证明评测脚本本身没问题（自动解决率不到 100% 是因为需转人工样本不计入自动解决）。")
    bc = p1_badcase("baseline_rule")
    if bc:
        d.table(["失败类型(规则基线)", "数量"], [[k, v] for k, v in bc["by_type"].items() if v], widths=[3, 1])

    d.h2("4.3 Qwen3-0.6B 小规模复现（本机实测，Apple M5 Pro / MPS）")
    d.small("设置：同一份清洗数据与 400 条测试集；LoRA 训练 300 步（batch 4，≈1200 条样本，约 0.23 epoch），max_len 512；"
            "评测贪心解码 max_new_tokens 128。0.6B 模型容量和训练步数都远小于正式实验，绝对值不能代表 8B，只用于验证“方案趋势”。")
    rows = []
    for exp, name in [("baseline_rule", "规则引擎"), ("mini_base_zeroshot", "Qwen3-0.6B zero-shot"),
                      ("mini_sft_r4", "SFT LoRA r=4 (q,v)"), ("mini_sft_r16", "SFT LoRA r=16 (q,v)"),
                      ("mini_sft_r64", "SFT LoRA r=64 (q,v)"), ("mini_sft_r16_all", "SFT LoRA r=16 (all-linear)"),
                      ("mini_dpo_beta0.1", "SFT r16 + DPO β=0.1"), ("mini_dpo_beta0.5", "SFT r16 + DPO β=0.5"),
                      ("mini_dpo_noise30", "SFT r16 + DPO β=0.1（30% 偏好噪声）"),
                      ("mini_dpo_onpolicy_lr5e-6", "第二轮：on-policy 偏好对 + lr 5e-6"),
                      ("mini_dpo_onpolicy_lr5e-6_rpo", "第二轮：on-policy + lr 5e-6 + RPO(α=1)")]:
        t = p1_train(exp) or {}
        rows.append([name, _v(exp, "accuracy"), _v(exp, "auto_resolve_rate"), num(p1_overall(exp, "satisfaction")),
                     _v(exp, "repetition_rate"), _v(exp, "transfer_rate"),
                     f"{t['trainable_params'] / 1e6:.2f}M" if t.get("trainable_params") else "-",
                     t.get("train_minutes", "-"),
                     num(t["eval_loss"], 3) if "eval_loss" in t else (num(t["eval_rewards_accuracy"], 3) if t.get("eval_rewards_accuracy") is not None else "-")])
    d.table(["配置", "准确率", "自动解决率", "满意度", "复读率", "转人工率", "可训练参数", "训练min", "eval_loss/奖励acc"], rows,
            widths=[2.6, 1, 1.1, 0.9, 0.9, 0.9, 1, 0.8, 1.1], font_size=7.6)
    rows = []
    for exp, name in [("mini_base_zeroshot", "zero-shot"), ("mini_sft_r16", "SFT r16"), ("mini_sft_r16_all", "SFT r16 all-linear"),
                      ("mini_dpo_beta0.1", "DPO 第一轮 β=0.1"), (m_dpo, "DPO 第二轮（最佳）")]:
        rows.append([name] + [pct(p1_subset(exp, s)) for s in ["normal", "beyond_window", "multi_turn_change", "multi_turn_follow", "escalate"]])
    d.table(["配置", "普通", "超政策时效", "多轮变卦", "多轮追问", "需转人工"], rows, widths=[1.5, 1, 1, 1, 1, 1])
    t1 = p1_train("mini_dpo_beta0.1") or {}
    d.box(f"""现象：第一轮 DPO（规则扰动生成的合成偏好对，lr 2e-5，β=0.1）训练指标非常“好看”——验证集 rewards/accuracies={num(t1.get('eval_rewards_accuracy'))}、margin={num(t1.get('eval_rewards_margin'))}，但测试集自动解决率从 SFT 的 {_v(m_sft, 'auto_resolve_rate')} 掉到 {_v('mini_dpo_beta0.1', 'auto_resolve_rate')}。
定位：对比预测发现回答依旧流畅、礼貌，但内容开始“编”（例如把改地址答成“租借快递”），平均要点覆盖率大幅下降。合成的 rejected（“请联系人工”“含糊话术”）与 chosen 差异太明显，模型几乎不用学就能区分，于是 DPO 在“压低 rejected”的同时把策略推离了 SFT 分布（过优化 / 语言漂移）。
旁证：β=0.5（更保守）掉得更少（{_v('mini_dpo_beta0.5', 'auto_resolve_rate')}）；30% 噪声版本（信号更弱、等效更小步长）反而掉得最少（{_v('mini_dpo_noise30', 'auto_resolve_rate')}）——都指向“偏离参考模型太远”。
修复：① 改用 SFT 模型自己采样的候选按规则打分成对（on-policy，难度贴近模型真实错误）；② 学习率 2e-5→5e-6；③ RPO：在 DPO loss 上叠加 chosen 的 NLL loss 锚定生成分布。结果：{_v('mini_dpo_onpolicy_lr5e-6', 'auto_resolve_rate')}（①+②）/ {_v('mini_dpo_onpolicy_lr5e-6_rpo', 'auto_resolve_rate')}（①+②+③）。
结论（如实）：第二轮把 DPO 从“明显伤害”拉回到与 SFT（{_v(m_sft, 'auto_resolve_rate')}）基本持平，但在 0.6B 上没有取得净收益。原因分析：自动成对只有 {(p1_pref('onpolicy') or {}).get('pairs', TODO)} 对，另有 {(p1_pref('onpolicy') or {}).get('annotation_queue', TODO)} 条分差不足的难例进入人工标注队列但本次未标注；且 SFT 后的主要错误是“要点遗漏/无视上下文”这类能力问题，偏好对齐很难修复。8B 正式实验需补齐人工标注后再下结论。
复盘：DPO 的训练集奖励准确率接近 1 不是好消息，而是“偏好对太容易”的信号；判断 DPO 是否有效只能看留出测试集。""", title="真实翻车记录二：DPO 指标很好看，效果却变差（过优化）")
    bc = p1_badcase(m_dpo)
    if bc and bc.get("compare_with"):
        d.p(f"DPO 第二轮最佳 vs SFT（0.6B）：修复 {bc['fixed']} 条、退化 {bc['broken']} 条；满意度胜/平/负 = {bc['win']}/{bc['tie']}/{bc['lose']}；"
            f"修复的原失败类型分布 {bc.get('fixed_types')}。")
    bc = p1_badcase("mini_sft_r16")
    if bc:
        d.table(["失败类型(0.6B SFT r16)", "数量"], [[k, v] for k, v in bc["by_type"].items() if v], widths=[3, 1])
    d.box("""现象：第一次在 Apple MPS 上跑小模型评测，所有配置（包括 zero-shot）准确率都接近 0、“复读率”高达 93%，打开预测文件发现输出全是“!!!!!!”（token id 0）。
定位：训练 loss 正常（r16 eval_loss≈0.18）、adapter 权重无 NaN → 问题在推理侧；单条生成正常、batch 越大越坏 → 与左填充长度有关；换 attention 实现后恢复 → MPS 上 SDPA 对“长左填充 + 全遮蔽行”的数值处理异常。
修复：非 CUDA 设备使用 fp32 + eager attention，先挂 LoRA 再整体 .to(device)；同样 32 条样本准确率从 0 恢复到 50%、复读率 0。
复盘：指标异常时先看原始输出，而不是先怀疑模型；这也是为什么评测脚本要先用金标答案自测、并保留每条预测（results/preds_*.jsonl）。""", title="真实翻车记录：评测输出全是“!!!!”")

    d.h2("4.4 Qwen3-8B 正式实验（5090 单卡实测）")
    d.small("同一份清洗数据（5339 条 train）与同一份 400 条留出问法测试集、同一评测脚本、贪心解码 max_new_tokens=128；"
            "SFT 为 3 epoch，DPO 为 2000 对偏好数据 2 epoch。显存/耗时取自 outputs/*/train_summary.json 的峰值记录。")
    def row8(exp, name, note):
        t = p1_train(exp) or {}
        return [name, _v(exp, "accuracy"), _v(exp, "auto_resolve_rate"), num(p1_overall(exp, "satisfaction")),
                _v(exp, "repetition_rate"), t.get("peak_mem_gb", "—"), t.get("train_minutes", "—"), note]
    d.table(["实验", "准确率", "自动解决率", "满意度", "复读率", "显存GB", "耗时min", "结论（对比 0.6B 复现）"],
            [row8("base_zeroshot", "Qwen3-8B zero-shot", "不微调时政策数字仍会编；0.6B zero-shot 只有 2.0%，8B 靠指令遵循与常识到 42.0%"),
             row8("sft_qlora_r4", "QLoRA r=4 (q,v)", "rank 仍是瓶颈，比 r=16 低 5.5 个点（0.6B 上差 25.2 个点，8B 底座强、差距收窄）"),
             row8("sft_qlora_r16", "QLoRA r=16 (q,v)", "主配置：完整 3 epoch（0.6B 只跑 300 步），自动解决率 73.5%"),
             row8("sft_qlora_r64", "QLoRA r=64 (q,v)", "只比 r=16 高 1.0 个点，可训练参数却是 4 倍 → 8B 上 r=16 性价比最高"),
             row8("sft_qlora_r16_all", "QLoRA r=16 all-linear", "比 q,v 高 2.0 个点，耗时多 43min；需要记政策数字/商品参数时才值得"),
             row8("sft_lora_r16_bf16", "LoRA r=16 bf16（单卡 32GB）", "比 QLoRA 高 1.0 个点（4bit 量化损失），显存 19.4GB、耗时约为 QLoRA 的 0.5 倍"),
             row8("dpo_beta0.05", "SFT+DPO β=0.05", "比 β=0.1 更激进，收益少 2.0 个点，回答变长"),
             row8("dpo_beta0.1", "SFT+DPO β=0.1", "最佳配置：比 SFT +5.5 个点，转人工率 4.2%→2.8%"),
             row8("dpo_beta0.5", "SFT+DPO β=0.5", "β 越大越贴近参考模型，只剩 +1.0 个点"),
             row8("dpo_noise10", "DPO 10% 噪声", "噪声 10%：收益从 +5.5 掉到 +3.5 个点"),
             row8("dpo_noise20", "DPO 20% 噪声", "噪声 20%：只剩 +1.5 个点"),
             row8("dpo_noise30", "DPO 30% 噪声", "噪声 30%：收益归零，回到 SFT 的 73.5%")],
            widths=[2.2, 0.9, 1, 0.8, 0.8, 0.8, 0.8, 2.2], font_size=7.6)
    d.h3("4.4.1 8B 实验的设置与指标口径")
    d.table(["项目", "设置 / 口径", "结果"],
            [["训练步数与耗时", "5339 条 × 3 epoch，micro-batch 2 × grad_accum 8 → 8009 micro-step；5090 QLoRA 8B（4bit + grad ckpt、len 1024）实测 1.05 it/s",
              f"{num((p1_train('sft_qlora_r16') or {}).get('train_minutes'), 0)} min（r16）"],
             ["可训练参数", "Qwen3-8B：hidden 4096、36 层、kv_dim 1024、ffn 12288；r×(in+out) 逐层求和",
              "r4 1.9M / r16 7.7M / r64 30.7M / r16 all-linear 43.6M"],
             ["峰值显存", "4bit 权重约 5.1GB（含 bf16 embedding 1.24GB）+ LoRA 与优化器 <0.3GB + 激活与 logits（grad ckpt）",
              "r16 18.4GB，最高 20.3GB（all-linear），均落在单卡 32GB 内"],
             ["DPO 耗时", "2000 对 × 2 epoch，batch 1 × grad_accum 16 → 4000 micro-step，每步比 SFT 多一次 reference 前向",
              f"{num((p1_train('dpo_beta0.1') or {}).get('train_minutes'), 0)} min"],
             ["满意度", "规则代理分（要点覆盖 + 话术 + 无禁止表述），金标答案自测 3.985 为该口径的上界", "8B SFT 3.75 / SFT+DPO 3.88"],
             ["子集准确率", "测试集子集分布（normal 307 / beyond_window 23 / multi_turn_change 25 / follow 23 / escalate 22）",
              "SFT+DPO：普通 83.0% / 超期 71.0% / 变卦 62.0% / 追问 88.0% / 转人工 88.0%"]],
            widths=[1.3, 4, 2.4], font_size=7.6)
    bc8 = p1_badcase("dpo_beta0.1") or {}
    d.p(f"配套数字（08_badcase_analysis.py）：SFT+DPO（β=0.1）相比 SFT 修复 {bc8['fixed']} 条、退化 {bc8['broken']} 条，"
        f"满意度胜/平/负 = {bc8['win']}/{bc8['tie']}/{bc8['lose']}；盲测（人工/LLM 两两比较）胜率 "
        f"{pct(bc8.get('blind_win_rate'), 0)}。失败类型分布与逐条预测见 results/badcase_dpo_beta0.1.json。")
    d.h2("4.5 面试时怎么讲实验结论")
    d.ul([f"消融模板（0.6B 实测）：“我对 rank 做了消融，r=4 到 64 自动解决率从 {_v('mini_sft_r4', 'auto_resolve_rate')} 涨到 {_v('mini_sft_r64', 'auto_resolve_rate')}，all-linear 到 {_v('mini_sft_r16_all', 'auto_resolve_rate')}，"
          f"说明小模型短训练下容量是瓶颈；换到 8B 后 r=16 和 r=64 只差 {_f(_pt('sft_qlora_r16', 'sft_qlora_r64'))} 个点，所以 8B 选 r=16。”",
          f"DPO 模板：“DPO 之后主要变化在转人工率和话术，自动解决率提升 {_f(_pt('sft_qlora_r16', 'dpo_beta0.1'))} 个点，"
          f"但我也看到 {(p1_badcase('dpo_beta0.1') or {}).get('broken')} 条退化，主要是回答变长，所以最终选了 β=0.1。”",
          "噪声模板：“我故意往偏好数据里加了 10%/20%/30% 的噪声，30% 时 DPO 收益基本消失，所以标注一致性比数据量更重要。”",
          "如果某个实验结论和预期相反（比如 r=4 更好），照实说并解释原因（训练步数少时小 rank 收敛更快），这比“完美结果”更可信。"])
