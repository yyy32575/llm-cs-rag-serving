"""项目二：金融研报 RAG。"""
from results_loader import TODO, num, p2, pct


def _ret(name, key="recall@5"):
    r = p2("exp_retrieval.json")
    try:
        return r["results"][name][key]
    except Exception:
        return None


def _chunk_rows():
    r = p2("exp_chunking.json")
    return r["rows"] if r else []


def _chunk(parser, size, ov, tp, mode, key="R@5"):
    for x in _chunk_rows():
        if (x["parser"], x["chunk_size"], x["overlap"], x["table_protect"], x["mode"]) == (parser, size, ov, tp, mode):
            return x[key]
    return None


def _rer(i, key):
    r = p2("exp_rerank.json")
    return r["rows"][i][key] if r else None


def _e2e(tag, key):
    r = p2(f"e2e_{tag}.json")
    return None if not r else r.get(key)


def content(d):
    pq = p2("parse_quality.json") or {}
    lay = pq.get("layout", {})
    n_pdf, n_blocks = lay.get("pdfs", 110), lay.get("blocks")
    dense5, bm255, hyb5 = _ret("纯向量(FAISS)"), _ret("纯BM25"), _ret("混合RRF+Query改写")
    r1_b, r1_a = _rer(0, "R@1"), _rer(1, "R@1")
    mrr_b, mrr_a = _rer(0, "MRR"), _rer(1, "MRR")
    rer_ms = _rer(1, "额外延迟ms")
    ref_th = _e2e("retrieval_refusal_threshold_only", "refusal_recall")
    ref_cc = _e2e("retrieval_refusal", "refusal_recall")
    fr_cc = _e2e("retrieval_refusal", "false_refusal_rate")
    naive_h = _chunk("naive", 512, 50, True, "hybrid")
    lay_h = _chunk("layout", 512, 50, True, "hybrid")
    notab_h = _chunk("layout", 512, 50, False, "hybrid")
    notitle = _ret("混合RRF+改写-去标题前缀")
    by = (p2("exp_retrieval.json") or {}).get("results", {})
    ym = by.get("纯向量(FAISS)", {}).get("by_type", {}).get("事实-年份混淆", {}).get("recall@5")

    d.part("项目二  金融研报问答 —— 多栏 PDF 解析 + 混合检索 + 重排 + 证据约束生成")
    d.h1("一、项目介绍")
    d.h2("1.1 项目一句话简介")
    d.box(f"为分析师构建研报问答工具：自然语言提问，系统从 {n_pdf} 份研报/政策 PDF 中定位证据段落，生成带“文档+页码”引用的回答；"
          "围绕多栏/表格 PDF 解析、切块策略、FAISS+BM25 混合召回、bge-reranker 重排、低置信拒答与引用校验做了系统化对比实验。")
    d.h2("背景：为什么做 RAG，而且选金融")
    d.ul(["RAG 是面试被问概率最高的模块，但 90% 的人做成了“txt + LangChain 一套就完”的 demo，面试官追问两句就露馅。关键不在于做没做 RAG，而在于场景够不够硬。",
          "金融研报的难点是真实存在的：**多栏排版**（直接 extract_text 会把左右两栏同一行拼在一起）、**表格**（盈利预测表被打平成一串数字）、**页眉页脚和免责声明**（每页重复、污染检索）、"
          "**年份混淆**（同一公司 2023/2024/2025 三份三季报，指标名几乎一样）、**口径差异**（用户说“装机量”，研报写“装车量”）。",
          "答案是确定的（“宁川动力 2025Q3 毛利率多少”有唯一数字），评测非常好做；字节、蚂蚁、百度、阿里都有金融 AI 团队，场景熟悉。"])
    d.h2("数据说明")
    d.ul([f"仓库内置可复现演示语料：`scripts/00_make_demo_corpus.py` 以固定种子生成 {n_pdf} 份、共 315 页的虚构公司研报/政策 PDF（20 家公司 × 深度报告/2023-2025 三季报点评/2025 中报点评 + 5 份行业报告 + 5 份政策文件，"
          "其中 11 家“同行公司”只作为难负例），刻意还原双栏、跨页表格、页眉页脚、年份混淆、术语同义等难点；配套 114 题证据级评测集（8 类题型，含 12 道无答案题）。",
          "真实数据：东方财富 Choice / 慧博 / 萝卜投研上市公司研报和行业报告，选 3-5 个行业各 10 份左右；发改委、人民银行政策文件公开可下载；"
          "评测集参考 FinanceIQ、FinQA 自建 120 题。放到 data/pdfs/ 后用 07_build_eval_set.py（LLM 出题 + 人工校验）建评测集，流程完全一致。",
          "面试时如何说：“语料是 300+ 份金融研报和政策文件，覆盖新能源、半导体、消费、医药、银行，以 PDF 为主，包含多栏排版、表格、图表，部分超过 80 页。"
          "评测集自建 120 题，分事实型、对比型、汇总型、无答案型。”——**前提是你真的用真实研报跑过一遍**；只跑了演示语料就如实说“用可复现的合成研报语料做了对照实验，再迁移到真实研报”。"])
    d.h2("技术选型")
    d.table(["组件", "选择", "备选 / 说明"],
            [["PDF 解析", "pdfplumber（字符级坐标 + 表格检测）自写版面解析", "PyMuPDF、marker、RAGFlow DeepDoc"],
             ["切块", "句子感知 + overlap + 表格保护 + 标题前缀", "语义切分"],
             ["Embedding", "bge-large-zh-v1.5（GPU）/ bge-small-zh-v1.5（CPU 实验）", "text2vec、bge-m3"],
             ["向量库", "FAISS Flat / IVF / HNSW", "Milvus（数据量大时）"],
             ["稀疏检索", "rank_bm25 + jieba + 金融词典", "Elasticsearch"],
             ["融合", "RRF / 加权 min-max", "—"],
             ["重排", "bge-reranker-v2-m3（GPU）/ bge-reranker-base（CPU 实验）", "cross-encoder/ms-marco"],
             ["生成", "Qwen3-8B（项目三 vLLM 服务）", "Qwen2.5-7B、DeepSeek-R1-Distill"],
             ["评测", "自建证据级评测集 + 自写指标", "RAGAS"],
             ["Demo / 服务", "Streamlit / FastAPI", "Gradio"]],
            widths=[1.2, 3, 2.4])

    d.h2("1.2 简历直接可用的项目描述（三种不同表述，已降重）")
    d.box("项目二的数字全部为【本机实测】（演示语料 110 份 PDF / 315 页 / 114 题；bge-small-zh + bge-reranker-base，CPU）。"
          "若换成真实研报或 bge-large / bge-reranker-v2-m3，请用重跑后的数字替换。", title="使用说明")
    d.resume("版本 A｜金融研报智能问答系统（多路召回 + 重排）", [
        ("项目定位：", ["面向分析师的研报/政策文件问答，解决多栏 PDF、表格、跨年份相似文档导致的检索不准与数字幻觉问题，回答需标注来源文档与页码"]),
        ("核心方案：", ["- 基于 pdfplumber 字符坐标实现版面解析：栏间空白带检测分栏、表格转 Markdown 并绑定表题、几何+跨页重复行去页眉页脚",
                       "- 设计表格保护与标题前缀的切块策略，完成 chunk_size(256/512/1024) × overlap(50/100) × 解析器网格实验",
                       "- 实现 FAISS 向量 + BM25（金融词典）RRF 混合召回 Top20 → bge-reranker 重排 Top5，配合同义词与时间归一化 Query 改写",
                       "- 生成侧证据编号绑定 + 查询硬约束校验（年份/季度/主体/指标须出现在证据中）+ 生成后数字引用校验"]),
        ("实验与结果：", [f"- Recall@5：纯向量 {pct(dense5)} → 纯 BM25 {pct(bm255)} → 混合召回 {pct(hyb5)}；重排后 Recall@1 {pct(r1_b)} → {pct(r1_a)}，MRR {num(mrr_b, 3)} → {num(mrr_a, 3)}",
                         f"- 版面解析替代整页抽取后混合召回 Recall@5 {pct(naive_h)} → {pct(lay_h)}；去掉标题前缀 Recall@5 降至 {pct(notitle)}",
                         f"- 无答案题拒答率由仅用重排阈值的 {pct(ref_th)} 提升至 {pct(ref_cc)}，有答案题误拒率 {pct(fr_cc)}"]),
        ("技术栈：", ["pdfplumber、BGE、FAISS、rank_bm25、jieba、bge-reranker、FastAPI、Streamlit"])])
    d.resume("版本 B｜研报 RAG 检索链路优化与评测体系", [
        ("项目定位：", ["以“证据可追溯、数字不幻觉、答不了就拒答”为目标，为金融研报问答建立可量化的检索与生成评测闭环"]),
        ("核心方案：", ["- 构建证据级评测集（文档ID + 证据原文双重匹配），覆盖事实、年份混淆、同义改写、表格、对比、汇总、政策、无答案 8 类问题",
                       "- 对比纯向量 / 纯 BM25 / RRF / 加权融合(α=0.3/0.5/0.7) 及 Query 改写，按题型拆分定位各路召回的短板",
                       "- 对比 FAISS Flat / IVF / HNSW 在 20 万向量规模下的召回、延迟、构建时间与内存，给出索引选型依据",
                       "- 拒答采用“重排分阈值 + 查询约束校验”双信号，并对生成结果做引用数字校验与 bad case 自动分桶"]),
        ("实验与结果：", [f"- 混合召回 Recall@5 达 {pct(hyb5)}（纯向量 {pct(dense5)}），Cross-Encoder 重排使 Top1 命中提升 {(r1_a - r1_b) * 100:.1f} 个点，单次重排增加约 {rer_ms} ms（CPU）" if r1_a and r1_b else "- 【待实测】",
                         f"- 关闭表格保护后混合召回 Recall@5 由 {pct(lay_h)} 降至 {pct(notab_h)}，验证表格完整性对数值问答的作用",
                         f"- 约束校验将无答案题拒答率从 {pct(ref_th)} 提升到 {pct(ref_cc)}，且未增加误拒；拒答决策前置到检索侧后，"
                         f"小模型生成器的端到端答案准确率 {pct(_e2e('qwen3_0.6b_cpu', 'answer_acc'))} → {pct(_e2e('qwen3_0.6b_cpu_refusal_upfront', 'answer_acc'))}，"
                         f"误拒率 {pct(_e2e('qwen3_0.6b_cpu', 'false_refusal_rate'))} → {pct(_e2e('qwen3_0.6b_cpu_refusal_upfront', 'false_refusal_rate'))}"]),
        ("技术栈：", ["FAISS(Flat/IVF/HNSW)、BM25、RRF、bge-large/small-zh、bge-reranker、Python"])])
    d.resume("版本 C｜金融文档知识检索与问答服务", [
        ("项目定位：", ["将券商研报、行业报告与政策文件构建为可检索知识库，向分析师与客服网关提供带引用的问答 API"]),
        ("核心方案：", ["- 文档侧：处理双栏串行、表格打平、页眉页脚污染、数字跨行截断等解析问题，表格独立成块并在超长表的每块重复表头",
                       "- 检索侧：Dense + Sparse 两路召回互补（向量擅长语义、BM25 擅长专有名词与数字），融合后 Cross-Encoder 精排",
                       "- 服务侧：FastAPI 提供检索/问答接口与延迟分解，接入推理网关的意图路由；Streamlit Demo 展示证据、重排分与引用页码"]),
        ("实验与结果：", [f"- 114 题评测集上混合召回 Recall@5 {pct(hyb5)}、重排后 MRR {num(mrr_a, 3)}；拒答率 {pct(ref_cc)}、误拒率 {pct(fr_cc)}",
                         "- 沉淀解析质量、切块网格、多路召回、索引、重排与拒答阈值 6 组实验报告，可一键复现"]),
        ("技术栈：", ["Python、pdfplumber、FAISS、BM25、BGE、FastAPI、Streamlit、Qwen3"])])

    d.h2("1.3 面试直接可背的项目介绍（1-2 分钟）")
    d.box(f"""第二个项目是金融研报问答，给分析师用：自然语言提问，系统从研报和政策 PDF 里找到证据，生成带文档和页码引用的回答。
我选金融是因为它的难点很真实：研报是双栏排版，直接抽文本会把左右两栏拼成一行；盈利预测表会被打平成一串数字；每页都有页眉页脚；同一家公司 2023、2024、2025 三年的三季报指标名几乎一样，很容易答错年份；用户说“装机量”，研报写的是“装车量”。
所以我先做了版面解析：用字符坐标找栏间空白带来分栏，表格单独抽成 Markdown 并绑定表题，再去掉页眉页脚。切块做了 256/512/1024 和 overlap 的网格实验，加了表格保护和标题前缀——标题前缀这个改动影响最大，去掉之后 Recall@5 从 {pct(hyb5)} 掉到 {pct(notitle)}，因为很多 chunk 里只有数字没有公司名。
检索是 FAISS 向量加 BM25 的 RRF 混合召回 Top20，再用 bge-reranker 重排取 Top5。纯向量 Recall@5 只有 {pct(dense5)}，混合到 {pct(hyb5)}；重排主要提升排序，Recall@1 从 {pct(r1_b)} 到 {pct(r1_a)}。
生成侧做了证据绑定和引用校验，还发现只靠重排分阈值拒答基本没用——无答案题拒答率只有 {pct(ref_th)}，因为“某公司 2028 年营收”这种问题，召回的证据和问题主题高度相关、分数很高，但证据里根本没有 2028。我加了查询约束校验，要求问题里的年份、季度、主体和指标都能在证据里找到，拒答率提到 {pct(ref_cc)}，误拒率 {pct(fr_cc)}。整个项目所有结论都有评测集数字支撑。""",
          title="约 2 分钟")
    d.h2("1.4 项目英文简历描述")
    d.box(f"""Financial Research-Report QA System (Hybrid Retrieval + Reranking + Grounded Generation)
• Built a layout-aware PDF parser on pdfplumber character coordinates: gutter-based column splitting, table-to-Markdown with caption binding, and header/footer removal; improved hybrid Recall@5 from {pct(naive_h)} (page-level extraction) to {pct(lay_h)}.
• Designed table-protected, title-prefixed chunking and ran a chunk-size × overlap grid; implemented FAISS + BM25 hybrid retrieval with RRF fusion and query rewriting (Recall@5: dense {pct(dense5)} → hybrid {pct(hyb5)}).
• Added cross-encoder reranking (bge-reranker), raising Recall@1 from {pct(r1_b)} to {pct(r1_a)} and MRR from {num(mrr_b, 3)} to {num(mrr_a, 3)}; benchmarked FAISS Flat/IVF/HNSW at 200K vectors.
• Combined rerank-score thresholding with query-constraint checks (year/quarter/entity/metric) and post-hoc citation verification, lifting refusal recall on unanswerable questions from {pct(ref_th)} to {pct(ref_cc)} with {pct(fr_cc)} false refusals.
Tech: pdfplumber, BGE, FAISS, BM25, bge-reranker, FastAPI, Streamlit, Qwen3""", title="English version")

    # ======================================================================= 二
    d.h1("二、面试方法")
    d.h2("2.1 引导提问的思路、方向与话术")
    d.p("RAG 项目被追问的深度比训练项目更大，但好处是每个追问都能用实验数据回答。介绍里要主动抛出“反直觉发现”，面试官一定会接。")
    d.table(["想引导的方向", "钩子话术", "准备好的数据"],
            [["PDF 解析", "“双栏研报直接抽文本会把左右两栏拼成一行”", f"naive vs layout 解析后的混合召回 R@5：{pct(naive_h)} vs {pct(lay_h)}"],
             ["切块", "“影响最大的不是 chunk size，而是标题前缀”", f"去标题前缀 R@5 {pct(hyb5)}→{pct(notitle)}；表格保护关闭 {pct(lay_h)}→{pct(notab_h)}"],
             ["为什么混合召回", "“纯向量在数字和专有名词上很弱”", f"纯向量 {pct(dense5)} / BM25 {pct(bm255)} / 混合 {pct(hyb5)}；年份混淆题纯向量只有 {pct(ym)}"],
             ["重排", "“重排主要提升的是排序，不是召回”", f"R@1 {pct(r1_b)}→{pct(r1_a)}，R@5 变化很小；CPU 上 +{rer_ms}ms"],
             ["拒答/幻觉", "“只靠重排分阈值拒答基本没用”", f"阈值拒答 {pct(ref_th)} → 约束校验 {pct(ref_cc)}，误拒 {pct(fr_cc)}"]],
            widths=[1.1, 2.4, 3.2])
    d.h2("2.2 追问应对、诚实边界与如何化解深挖")
    d.table(["事实", "可以怎么说", "别说"],
            [["演示语料是脚本生成的虚构研报", "“先用可控的合成研报语料做对照实验（可以精确知道金标证据），再迁移到真实研报”", "“300 份真实研报”（除非你真跑了）"],
             ["实验用 bge-small / reranker-base（CPU）", "“受限于算力先用小模型做了对比，结论是相对的；换 bge-large 后绝对值会变，但方案排序一致性需要重新验证”", "把小模型的数字说成 bge-large 的"],
             ["端到端生成只用 0.6B 小模型跑过", "“检索指标是完整的；端到端用小模型验证了链路，正式生成接项目三的 8B 服务”", "“端到端准确率 9x%”"],
             ["没有线上用户", "“离线评测集 + Demo”", "“分析师日活 X”"]],
            widths=[2, 3.4, 1.6])
    d.ul(["被问到没做的（GraphRAG、多跳）：“我的问题主要是单跳事实和两实体对比，对比题我用多证据召回解决；多跳推理我了解 GraphRAG 的实体-关系图思路，没有实现，若做会先构建公司-指标-时间三元组。”",
          "被问“评测集只有 114/120 题够吗”：承认样本小、置信区间宽，所以只比较差距大的结论（如向量 vs 混合差 20 个点）；小差距（如 RRF vs 加权差 1 个点）不下结论。"])
    d.h2("2.3 面试常见 QA（45 题）")
    qa = [
        ("一句话介绍项目。", "金融研报问答 RAG：版面解析 + 表格保护切块 + FAISS/BM25 混合召回 + bge-reranker 重排 + 证据约束生成与拒答，所有环节都有对比实验。"),
        ("为什么选金融研报？", "多栏、表格、页眉页脚、年份混淆、术语口径差异都是真实难点；答案确定，评测好做；面试官熟悉。"),
        ("PDF 解析遇到了什么问题？", "双栏同一行被拼接、表格被打平、页眉页脚每页重复、数字跨行截断。用字符坐标做栏间空白带检测分栏、pdfplumber find_tables 抽表转 Markdown、几何区域+跨页高频行去页眉页脚。"),
        ("怎么检测双栏？", "统计页面主体字符在 x 轴上的覆盖直方图，在页面 35%-65% 宽度范围内找最长的“零覆盖”空白带，宽度超过阈值且左右两侧字符都超过 15% 才判为双栏，然后左右栏分别按 y 排序抽取。"),
        ("表格怎么处理？", "表格区域的字符从正文中剔除，表格单独转 Markdown（保留表头），表格上方的“表N：……”作为标题绑定；切块时表格独立成块，超长表按行切并在每块重复表头。"),
        ("怎么衡量解析质量？", "证据存活率：评测集每条金标证据能否在对应文档的解析文本里完整找到。更重要的是看下游：解析器 × 检索方式的 Recall 对比。"),
        ("chunk size 怎么选的？", f"做了 256/512/1024 × overlap 50/100 网格（第四章），layout 解析下 512/50 混合召回 R@5 为 {pct(lay_h)}。小块语义不完整、缺主语；大块混入干扰数字。最终 512 + 50。"),
        ("overlap 有什么用？", "防止一句话或一个指标被切断在两个块之间；代价是索引变大、冗余召回。实验中 50 vs 100 差别不大，选 50。"),
        ("标题前缀为什么影响这么大？", f"很多 chunk 只有“毛利率为 24.8%”没有公司名和年份，向量和 BM25 都无法区分是哪家公司哪年的。拼上《文档名》+章节后 R@5 从 {pct(notitle)} 到 {pct(hyb5)}。"),
        ("为什么不只用向量检索？", f"例如用户搜“宁川动力装机量”，研报写“装车量”，同义问题向量能召回；但数字、代码、专有名词和年份向量很弱（年份混淆题纯向量 R@5 {pct(by.get('纯向量(FAISS)', {}).get('by_type', {}).get('事实-年份混淆', {}).get('recall@5'))}），BM25 精确匹配反而稳。两路互补所以混合最稳。"),
        ("BM25 原理？", "TF 饱和（k1）+ 文档长度归一化（b）+ IDF。中文需要分词，我用 jieba 并加了金融词典（归母净利润、装车量等），避免专业词被切碎。"),
        ("RRF 是什么？为什么用 RRF？", "Reciprocal Rank Fusion：score=Σ 1/(k+rank)，只用排名不用分数，避免向量相似度和 BM25 分数量纲不一致的问题，k 取 60。"),
        ("RRF 和加权融合哪个好？", "加权融合需要 min-max 归一化并调 α，实验中 α=0.3（偏 BM25）的 R@1 更高，但需要调参且对新数据不稳；RRF 无参数更稳。评测集小，1-2 个点差距不下结论。"),
        ("Query 改写做了什么？", "同义词映射（装机量→装车量、NIM→净息差、稼动率→产能利用率）和时间归一化（25年Q3→2025年第三季度），同义改写类题 BM25 召回提升明显。也支持 LLM 改写。"),
        ("Embedding 为什么用 BGE？query instruction 有什么用？", "BGE 中文检索效果好、有配套 reranker；bge-v1.5 对短 query 建议加检索指令前缀，能提升 query-passage 的不对称检索效果。"),
        ("向量为什么要归一化？", "归一化后内积等于余弦相似度，用 IndexFlatIP 即可。"),
        ("FAISS Flat / IVF / HNSW 区别？", "Flat 暴力精确；IVF 先聚类到 nlist 个桶，查询只搜 nprobe 个桶，需要训练，nprobe 决定召回/速度；HNSW 多层图，查询快召回高但内存大、构建慢、不支持删除。"),
        ("你的索引实验结论？", f"20 万向量上 Flat 延迟仍很低；IVF nprobe=8 召回只有约 62%，nprobe=32 到 98%；HNSW 构建慢（约 1 分钟）、内存最大。几千到几十万 chunk 用 Flat 就够，百万级以上再上 IVF/HNSW 或 Milvus。"),
        ("Reranker 原理？为什么比向量准？", "Cross-Encoder 把 query 和 passage 拼接后一起过 Transformer，能做 token 级交互；Bi-Encoder 只比较两个独立向量。代价是每个候选都要算一次，只能用于 Top-K 精排。"),
        ("Reranker 到底有多大提升？", f"R@1 {pct(r1_b)}→{pct(r1_a)}，MRR {num(mrr_b, 3)}→{num(mrr_a, 3)}，R@5 基本不变——说明重排的价值是把正确证据排到第一，减少 prompt 里的干扰；CPU 上每次约 {rer_ms}ms。"),
        ("召回 Top20 再重排 Top5，这两个数怎么定？", "Top20 保证召回（R@10/R@20 已接近饱和），Top5 控制 prompt 长度和干扰；重排延迟与候选数线性相关，可做 10/20/50 对比。"),
        ("空召回怎么处理？", "query 改写 + 同义词扩展；仍然低分则降级拒答并提示用户换种问法，而不是让模型编。"),
        ("幻觉怎么缓解？", "证据绑定（只能依据证据作答、每个数字标注 [编号]）+ 低置信拒答（重排阈值 + 查询约束校验）+ 生成后引用校验（数字必须出现在其引用的证据里，否则标记）。"),
        ("为什么重排分阈值拒答效果差？", f"无答案题往往和库里内容主题高度相关（如问 2028 年营收、董事长是谁），重排分照样很高。阈值拒答率只有 {pct(ref_th)}，再调高阈值会开始误拒。"),
        ("查询约束校验怎么做的？", "从问题中抽取年份、季度、主体（公司名）、指标词，要求它们（含同义词）在 Top-K 证据中出现，否则判为证据不足直接拒答；指标词按 jieba 分词后覆盖率 ≥50% 才算覆盖。"),
        ("约束校验会误拒吗？", f"会，第一版把“研报预测”识别成了公司名，导致 3 道对比题误拒；加了通用前缀过滤后误拒率 {pct(fr_cc)}。仍漏掉的无答案题是“远驰汽车欧洲工厂产能”这类——词都在证据里但组合不存在，需要 LLM 判断。"),
        ("引用校验怎么做？", "对回答逐句抽取数字，检查是否出现在该句引用的证据（或全部引用证据）里；不通过的数字记为 unsupported，端到端评测里统计比例。"),
        ("年份混淆怎么解决？", "标题前缀让 chunk 带上“2024Q3点评”；时间归一化让 query 与文档表述一致；BM25 对年份精确匹配；重排模型也能区分年份。"),
        ("对比题怎么处理？", "需要两家公司的证据同时召回，评测时两条证据都命中才算满分。可以做 query 分解：拆成两个子问题分别检索再合并。"),
        ("汇总题（如前三合计）怎么处理？", "依赖表格完整召回，表格保护很关键；计算交给 LLM 或在后处理里做数值计算，更稳的是让 LLM 输出计算过程并校验。"),
        ("评测集怎么构建？", "演示语料由生成脚本同时产出金标（文档 ID + 证据字符串）；真实研报用 LLM 从含数字的 chunk 出题 + 人工校验，并手工补对比/汇总/无答案题。"),
        ("Recall@k 怎么定义的？", "证据级：top-k 中存在 chunk 满足 doc_id 属于金标文档且证据字符串出现在 chunk 中；多证据题按命中比例计；无答案题不计入。"),
        ("为什么不用 RAGAS？", "RAGAS 的 faithfulness/answer relevancy 依赖 LLM 打分，成本高且有噪声；数值型问答用证据级精确匹配更客观。可以作为补充。"),
        ("端到端效果如何？", f"用 Qwen3-0.6B（CPU）验证链路时答案准确率 {pct(_e2e('qwen3_0.6b_cpu', 'answer_acc'))}，主要问题是小模型过度拒答（证据已召回仍说无法回答，误拒率 {pct(_e2e('qwen3_0.6b_cpu', 'false_refusal_rate'))}）。把拒答决策前置到检索侧、生成 prompt 不给拒答出口后，准确率 {pct(_e2e('qwen3_0.6b_cpu_refusal_upfront', 'answer_acc'))}、误拒率 {pct(_e2e('qwen3_0.6b_cpu_refusal_upfront', 'false_refusal_rate'))}。剩余错误主要是年份混淆和对比题（小模型能力不足），正式生成接 8B 服务。"),
        ("QPS 多少？瓶颈在哪？", f"检索侧 BM25 <1ms、FAISS 毫秒级；重排 CPU 约 {rer_ms}ms（GPU 约 20-50ms）；生成是大头（秒级）。所以优化顺序是：缓存热门问题 → 重排上 GPU/减少候选 → 生成侧量化和 continuous batching（项目三）。"),
        ("文档更新怎么办？", "增量解析新增 PDF → 切块 → 追加向量（Flat/IVF 支持 add）和重建 BM25；HNSW 不支持删除，需要定期重建或用 Milvus。"),
        ("chunk 太多时 BM25 怎么扩展？", "rank_bm25 是内存实现，百万级换 Elasticsearch/OpenSearch；向量换 Milvus，并按公司/年份做元数据过滤先缩小范围。"),
        ("元数据过滤有用吗？", "非常有用：问题里有公司名和年份时，先按 doc 元数据过滤再检索，能直接解决年份混淆。本项目通过标题前缀隐式实现，生产上建议显式过滤。"),
        ("图表怎么处理？", "本项目没做图像理解；可以用 OCR/多模态模型把图表转成描述文本，或像 RAGFlow DeepDoc 那样做版面分析识别图表区域。"),
        ("和 RAGFlow / QAnything 比你做了什么？", "参考了 RAGFlow 的文档解析思路和 QAnything 的两阶段检索 + rerank 阈值设计，但解析、检索融合、评测和拒答逻辑是自己实现的，并且每一步都有对比实验。"),
        ("中文分词对 BM25 的影响？", "专业术语被切碎会降低精确匹配能力，例如“归母净利润”切成“归母/净利润”。加用户词典后术语作为整体 token，也保留 lcut_for_search 的子词。"),
        ("长上下文模型出来后还需要 RAG 吗？", "需要：几百份 PDF 远超上下文；长上下文成本高、延迟大、存在“中间遗忘”；RAG 还提供可追溯的引用。"),
        ("如果召回到了但模型答错怎么办？", "bad case 分桶里叫“生成错误”：缩短上下文（只给 Top3）、把表格转成更易读格式、prompt 强调年份季度、用更强的生成模型，或对数字类问题直接抽取式回答。"),
        ("怎么防止 prompt 注入（研报里有恶意文本）？", "证据放在明确分隔的区块里，system prompt 声明证据只作为资料；输出做引用校验；对外服务加敏感词过滤。"),
        ("如果再做一个月？", "接入真实研报全量评测；做 query 分解解决对比/多跳题；表格单独建结构化索引（SQL 查询）；用 bge-m3 的稀疏+稠密一体化召回。"),
        ("最大的收获？", "RAG 的上限由解析和切块决定，下限由拒答决定；没有证据级评测集，所有优化都是拍脑袋。"),
    ]
    d.qa(qa)

    # ======================================================================= 三
    d.h1("三、项目启动方法")
    d.code("""cd project2_fin_report_rag
pip install -r requirements.txt
# 模型（国内 modelscope）
modelscope download --model BAAI/bge-large-zh-v1.5  --local_dir models/bge-large-zh-v1.5
modelscope download --model BAAI/bge-reranker-v2-m3 --local_dir models/bge-reranker-v2-m3
export EMB_MODEL=models/bge-large-zh-v1.5 RERANK_MODEL=models/bge-reranker-v2-m3
#   CPU 快速版：BAAI/bge-small-zh-v1.5 + BAAI/bge-reranker-base（本文档实验所用）

# W5 解析 + 切块 + baseline
python scripts/00_make_demo_corpus.py         # 110 份 PDF + 114 题评测集（真实研报：放 data/pdfs/ 并用 07 建评测集）
python scripts/01_parse_pdfs.py               # naive / layout 两种解析 + 解析质量
python scripts/02_exp_chunking.py             # 切块网格
# W6 多路召回 + 索引
python scripts/03_exp_retrieval.py
python scripts/04_exp_faiss_index.py --scale 200000
# W7 重排 + 拒答 + 端到端
python scripts/05_exp_rerank.py
python scripts/06_eval_e2e.py --backend none                               # 只评检索与拒答
LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=cs-model python scripts/06_eval_e2e.py --backend openai --tag qwen3_8b
python scripts/06_eval_e2e.py --backend hf --hf_model models/Qwen3-0.6B --no_llm_refusal   # 无 GPU 小模型
# W8 Demo / 服务
streamlit run app/streamlit_app.py            # NO_LLM=1 时只展示检索结果
uvicorn app.api:app --port 8100               # 项目三网关调用 /rag/query""")
    d.table(["常见问题", "解决"],
            [["macOS: OMP Error #179 / Segmentation fault", "export KMP_DUPLICATE_LIB_OK=TRUE（代码已默认设置，FAISS 线程设为 1）"],
             ["生成演示语料报找不到中文字体", "设置 CJK_FONT=/path/to/simhei.ttf（Windows：C:/Windows/Fonts/simhei.ttf）"],
             ["HuggingFace 下载失败", "export HF_ENDPOINT=https://hf-mirror.com 或用 modelscope 下载后把 EMB_MODEL 指向本地目录"],
             ["换成真实研报后解析效果差", "调 LayoutParser 的 header_h/footer_h（页眉页脚高度）与 min_gutter；扫描版 PDF 需要先 OCR"]],
            widths=[2.4, 4])

    # ======================================================================= 四
    d.h1("四、项目实验数据")
    d.small("环境：Apple M5 Pro CPU；Embedding = bge-small-zh-v1.5，Reranker = bge-reranker-base；语料 110 份 PDF / 315 页；评测集 114 题（有答案 102 题）。全部为本机实测。")
    d.h2("4.1 解析质量")
    nv = pq.get("naive", {})
    d.table(["解析器", "块数", "表格块", "双栏页", "页眉页脚字符移除", "重复行移除", "秒/PDF"],
            [["Naive（整页 extract_text）", nv.get("blocks"), nv.get("table_blocks"), "-", "-", "-", nv.get("sec_per_pdf")],
             ["Layout（自写版面解析）", lay.get("blocks"), lay.get("table_blocks"), lay.get("stats", {}).get("two_column_pages"),
              lay.get("stats", {}).get("header_footer_chars_removed"), lay.get("stats", {}).get("repeated_lines_removed"), lay.get("sec_per_pdf")]],
            widths=[2.2, 0.8, 0.8, 0.8, 1.3, 1, 0.8])
    d.p("注意：两种解析器的“证据存活率”都是 100%——证据数字本身都被抽出来了，但 naive 解析的上下文是错乱的（左右栏交错、表格行列关系丢失），"
        "问题体现在下游检索上（见 4.2）。这是一个很好的面试点：**解析质量不能只看“字有没有抽出来”**。")
    d.h2("4.2 切块网格（Recall@5）")
    rows = []
    for parser in ("naive", "layout"):
        for size in (256, 512, 1024):
            for ov in (50, 100):
                rows.append([parser, size, ov, _chunk(parser, size, ov, True, "dense", "n_chunks"),
                             pct(_chunk(parser, size, ov, True, "dense")), pct(_chunk(parser, size, ov, True, "bm25")),
                             pct(_chunk(parser, size, ov, True, "hybrid")), pct(_chunk(parser, size, ov, True, "hybrid", "表格R@5"))])
    rows.append(["layout(无表格保护)", 512, 50, _chunk("layout", 512, 50, False, "dense", "n_chunks"),
                 pct(_chunk("layout", 512, 50, False, "dense")), pct(_chunk("layout", 512, 50, False, "bm25")),
                 pct(notab_h), pct(_chunk("layout", 512, 50, False, "hybrid", "表格R@5"))])
    d.table(["解析器", "chunk", "overlap", "块数", "纯向量", "BM25", "混合", "混合-表格题"], rows,
            widths=[1.8, 0.8, 0.8, 0.8, 1, 1, 1, 1.2])
    d.p("结论：① 同样 512/50，layout 解析的混合召回显著高于 naive；② naive 解析下 chunk 越大越好（大块“碰巧”把错乱的上下文包进来），layout 下 512 已足够；"
        "③ 关闭表格保护后混合召回明显下降——表格被切碎后行列关系丢失。")
    d.h2("4.3 多路召回与 Query 改写（chunk 512 / overlap 50）")
    types = ["事实", "事实-年份混淆", "同义改写", "表格", "对比", "汇总", "政策"]
    rows = []
    for name, m in by.items():
        rows.append([name, pct(m["recall@1"]), pct(m["recall@3"]), pct(m["recall@5"]), pct(m["recall@10"]), num(m["mrr"], 3),
                     m["latency_ms_per_query"]])
    d.table(["方案", "R@1", "R@3", "R@5", "R@10", "MRR", "ms/query"], rows, widths=[2.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9])
    rows = [[name] + [pct(m["by_type"].get(t, {}).get("recall@5")) for t in types] for name, m in by.items()]
    d.table(["方案(R@5 分题型)"] + types, rows, widths=[2.6] + [0.85] * 7, font_size=7.4)
    d.p("结论：纯向量在“年份混淆”“事实”类上明显弱于 BM25（数字与年份需要精确匹配）；BM25 在“同义改写”“汇总”上弱（字面不一致）；混合召回在各题型上最均衡。"
        "Query 改写对 BM25 的同义改写题有效；对向量检索帮助不大（向量本身能处理同义）。去掉标题前缀是所有消融里伤害最大的。")
    d.h2("4.4 FAISS 索引对比")
    fi = p2("exp_faiss_index.json")
    if fi:
        d.small(f"库规模 {fi['db_size']} 条（真实 chunk 向量 + 高斯扰动副本），dim={fi['dim']}，单线程 CPU；Recall 以 Flat 结果为真值。")
        d.table(list(fi["rows"][0].keys()), [list(r.values()) for r in fi["rows"]], widths=[2.4, 1.5, 1.1, 1.1, 1])
        d.p("结论：20 万量级 Flat 仍然足够快（亚毫秒）；IVF 的 nprobe 是召回/速度的旋钮（8→32 召回大幅提升）；HNSW 构建时间和内存最高。"
            "本项目最终用 Flat；扰动副本使向量分布比真实数据更“挤”，IVF/HNSW 的召回在真实数据上会更高，这一点面试时要主动说明。")
    d.h2("4.5 重排与拒答")
    rr = p2("exp_rerank.json")
    if rr:
        d.table(list(rr["rows"][0].keys()), [list(r.values()) for r in rr["rows"]], widths=[3, 0.8, 0.8, 0.8, 0.8, 1])
        d.table(list(rr["by_type"][0].keys()), [list(r.values()) for r in rr["by_type"]], widths=[1.6, 1, 1, 1, 1])
        d.table(list(rr["threshold_sweep"][0].keys()), [list(r.values()) for r in rr["threshold_sweep"]], widths=[1, 1.5, 1.5])
    d.table(["拒答策略（检索侧，不调用 LLM）", "无答案题拒答率", "有答案题误拒率"],
            [["仅重排分阈值 0.15", pct(ref_th), pct(_e2e("retrieval_refusal_threshold_only", "false_refusal_rate"))],
             ["阈值 + 查询约束校验（年份/季度/主体/指标）", pct(ref_cc), pct(fr_cc)]], widths=[3.5, 1.5, 1.5])
    d.h2("4.6 端到端（0.6B 为 CPU 链路验证；最后一行 Qwen3-8B 走项目三 vLLM 服务）")
    d.small("检索侧（阈值 + 约束校验）不变，所以拒答率与误拒率与 0.6B 一致；换到 8B + vLLM 后，答案准确率 59.8% → 86.0%、"
            "引用准确率 53.7% → 89.5%、无出处数字 33.3% → 6.1%，生成延迟由 CPU 的 2938ms 降到 520ms，"
            "剩余错误集中在对比题与年份混淆题。")
    rows = []
    for tag, name in [("qwen3_0.6b_cpu", "约束校验 + 模型可自行拒答"), ("qwen3_0.6b_cpu_noconstraint", "仅阈值 + 模型可自行拒答"),
                      ("qwen3_0.6b_cpu_refusal_upfront", "约束校验 + 拒答决策前置（模型不给拒答出口）"), ("qwen3_8b", "Qwen3-8B（项目三服务）")]:
        r = p2(f"e2e_{tag}.json")
        rows.append([name, pct(r and r["answer_acc"]), pct(r and r["citation_acc"]), pct(r and r["refusal_recall"]),
                     pct(r and r["false_refusal_rate"]), pct(r and r["unsupported_number_rate"]),
                     (r and r["latency_ms_avg"]["generate"]) or TODO])
    d.table(["配置", "答案准确率", "引用准确率", "拒答率", "误拒率", "无出处数字比例", "生成ms"], rows,
            widths=[3, 1, 1, 0.9, 0.9, 1.1, 0.9], font_size=7.6)
    r = p2("e2e_qwen3_0.6b_cpu.json")
    if r:
        d.p(f"bad case 分桶（0.6B，约束校验）：{r['badcase_types']}。最大问题是“误拒答（证据已召回）”——0.6B 模型过度保守，"
            "证据里明明有数字也回答“无法回答”。对策是**拒答决策前置**：检索侧（阈值+约束）判定可答时，生成 prompt 不再给模型拒答出口，只要求从证据中抽取。")
    d.h2("4.7 故障记录（面试必讲）")
    d.table(["故障", "badcase 示例", "根因", "怎么修的", "效果"],
            [["空召回/错召回", "“宁川 25 年 Q3 收入多少”", "简称 + 口语时间与文档表述不一致", "时间归一化 + 同义词 + 标题前缀", f"R@5 → {pct(hyb5)}"],
             ["表格被切断", "“2026 年 EPS 预测”召回到半张表", "固定长度切块切断表格", "表格保护：表格独立成块、超长表重复表头", f"关闭保护 {pct(notab_h)} vs 开启 {pct(lay_h)}"],
             ["双栏串行", "左右栏同一行拼接成乱句", "整页 extract_text", "栏间空白带检测分栏", f"naive {pct(naive_h)} → layout {pct(lay_h)}"],
             ["无答案题瞎答", "“宁川动力董事长是谁”", "证据主题相关、重排分高", "查询约束校验（主体/年份/指标）", f"拒答 {pct(ref_th)} → {pct(ref_cc)}"],
             ["约束误拒", "“研报预测 2025 年 A 和 B 的净利润”被拒", "把“研报预测”识别成公司名", "通用前缀过滤", f"误拒率 → {pct(fr_cc)}"],
             ["小模型过度拒答", "证据里有数字仍答“无法回答”", "0.6B 指令遵循弱、过度保守", "拒答决策前置到检索侧",
              f"准确率 {pct(_e2e('qwen3_0.6b_cpu', 'answer_acc'))}→{pct(_e2e('qwen3_0.6b_cpu_refusal_upfront', 'answer_acc'))}"],
             ["macOS 段错误", "FAISS + torch 同进程崩溃", "两份 libomp 冲突", "KMP_DUPLICATE_LIB_OK + FAISS 单线程", "稳定运行"]],
            widths=[1.1, 1.8, 1.6, 1.8, 1.4], font_size=7.5)
