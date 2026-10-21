# 项目二：金融研报问答 —— 多栏 PDF 解析 + 混合检索 + 重排 + 证据约束生成的 RAG 系统

> 业务背景：给分析师做研报问答工具。自然语言提问，系统从几百份研报/政策 PDF 中找到相关段落，生成回答并**标注来源文档与页码**。
> 难点：研报是多栏排版 + 大量表格 + 页眉页脚 + 跨年份相似文档，“直接当 txt 切块”效果很差。

## 目录结构

```
project2_fin_report_rag/
├── src/finrag/
│   ├── pdf_parser.py     # NaiveParser(基线) vs LayoutParser：表格检测→Markdown、栏间空白带检测分栏、页眉页脚去除、段落合并
│   ├── chunker.py        # 句子感知切块 + overlap + 表格保护（表格独立成块、超长表按行切并重复表头）+ 标题前缀
│   ├── bm25.py           # jieba + 金融词典 + rank_bm25
│   ├── dense.py          # BGE embedding（带 query instruction、缓存）+ FAISS Flat/IVF/HNSW
│   ├── retriever.py      # 混合检索：RRF / 加权融合
│   ├── query_rewrite.py  # 同义词（装机量→装车量、NIM→净息差…）+ 时间归一化（25年Q3→2025年第三季度）+ 可选 LLM 改写
│   ├── reranker.py       # bge-reranker-v2-m3 / base Cross-Encoder
│   ├── generator.py      # 证据绑定 prompt、查询硬约束校验（年份/季度/主体/指标）、生成后引用校验
│   ├── pipeline.py       # 改写→混合召回Top20→重排Top5→拒答判定→生成→校验
│   └── evaluation.py     # Recall@k / MRR（证据级金标）、答案准确率、引用准确率、拒答率/误拒率
├── scripts/
│   ├── 00_make_demo_corpus.py  # 生成 110 份 / 315 页可复现演示语料 + 114 题评测集（虚构公司，无版权问题）
│   ├── 01_parse_pdfs.py        # 两种解析器 + 解析质量（证据存活率）
│   ├── 02_exp_chunking.py      # 解析器 × chunk_size(256/512/1024) × overlap(50/100) × 表格保护
│   ├── 03_exp_retrieval.py     # 纯向量 / 纯BM25 / 混合RRF / 加权α / Query改写 / 标题前缀 消融
│   ├── 04_exp_faiss_index.py   # Flat / IVF / HNSW：召回、延迟、构建时间、内存（20 万向量）
│   ├── 05_exp_rerank.py        # 重排前后 + 拒答阈值扫描
│   ├── 06_eval_e2e.py          # 端到端：答案准确率、引用准确率、拒答、延迟分解、bad case 分桶
│   └── 07_build_eval_set.py    # 真实研报：LLM 出题 + 人工校验 → 评测集
├── app/streamlit_app.py  # Demo：问题→证据→重排分→答案+引用页码
├── app/api.py            # FastAPI 检索服务（项目三网关调用）
└── configs/default.yaml
```

## 快速开始

```bash
pip install -r requirements.txt
# 模型（国内用 modelscope）
modelscope download --model BAAI/bge-large-zh-v1.5 --local_dir models/bge-large-zh-v1.5
modelscope download --model BAAI/bge-reranker-v2-m3 --local_dir models/bge-reranker-v2-m3
export EMB_MODEL=models/bge-large-zh-v1.5 RERANK_MODEL=models/bge-reranker-v2-m3
# CPU 演示可用小模型：BAAI/bge-small-zh-v1.5 + BAAI/bge-reranker-base

python scripts/00_make_demo_corpus.py          # 演示语料 + 评测集（真实研报放 data/pdfs/ 并用 07 建评测集）
python scripts/01_parse_pdfs.py
python scripts/02_exp_chunking.py
python scripts/03_exp_retrieval.py
python scripts/04_exp_faiss_index.py --scale 200000
python scripts/05_exp_rerank.py
# 端到端（生成模型用项目三部署的 vLLM 服务）
LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=cs-model python scripts/06_eval_e2e.py --backend openai
# Demo / 服务
streamlit run app/streamlit_app.py
uvicorn app.api:app --port 8100
```

macOS 上若出现 `OMP: Error #179` 段错误：`export KMP_DUPLICATE_LIB_OK=TRUE`（代码已默认设置并把 FAISS 线程数设为 1）。

## 评测口径

* 检索金标是“证据级”的：文档 ID + 必须出现的证据字符串（如 `24.8`），chunk 同时满足才算命中；对比题需两条证据都命中。
* 题型：事实 / 年份混淆（同公司 2023/2024/2025 三份季报）/ 同义改写 / 表格 / 对比 / 汇总 / 政策 / 无答案。
* 实验结果见 `results/*.md`。
