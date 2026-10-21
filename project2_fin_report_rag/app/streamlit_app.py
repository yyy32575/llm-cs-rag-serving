"""Streamlit demo: 输入问题 -> 展示召回证据 -> 重排分数 -> 生成答案 + 引用来源页码。

  EMB_MODEL=BAAI/bge-large-zh-v1.5 LLM_BASE_URL=http://localhost:8000/v1 streamlit run app/streamlit_app.py
"""
import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from finrag.pipeline import build_default_pipeline  # noqa: E402

st.set_page_config(page_title="金融研报智能问答", layout="wide")
st.title("金融研报智能问答（RAG Demo）")
st.caption("混合召回(FAISS+BM25) → bge-reranker 重排 → 证据约束生成 → 引用校验；演示语料为虚构数据")


@st.cache_resource
def get_pipe():
    return build_default_pipeline(ROOT, with_generator=os.environ.get("NO_LLM") != "1")


pipe = get_pipe()
examples = ["宁川动力2025年三季度的动力电池装机量是多少？", "2025年第三季度，黔台酒业和蒙泉乳业谁的单季毛利率更高？",
            "新能源行业2025年上半年市场份额前三的企业合计份额是多少？", "集成电路设计企业研发费用加计扣除比例是多少？",
            "宁川动力的董事长是谁？"]
q = st.selectbox("示例问题", examples)
q = st.text_input("或输入你的问题", q)
with st.sidebar:
    st.header("检索配置")
    pipe.cfg["mode"] = st.radio("召回方式", ["hybrid", "dense", "bm25"], index=0)
    pipe.cfg["rerank"] = st.checkbox("启用重排", True)
    pipe.cfg["final_k"] = st.slider("进入 Prompt 的证据数", 1, 10, 5)
    pipe.cfg["refuse_threshold"] = st.slider("低置信度拒答阈值", 0.0, 0.8, 0.15, 0.05)

if st.button("提问", type="primary") and q:
    with st.spinner("检索与生成中..."):
        r = pipe.answer(q)
    st.subheader("回答")
    (st.warning if r["refused"] else st.success)(r["answer"])
    if r["unsupported_numbers"]:
        st.error(f"引用校验未通过：以下数字未在引用证据中找到 {r['unsupported_numbers']}")
    st.caption(f"延迟(ms)：召回 {r['latency_ms']['recall']} | 重排 {r['latency_ms']['rerank']} | "
               f"生成 {r['latency_ms']['generate']} | Top1 重排分 {r['top_score']}")
    st.subheader("证据来源")
    for i, s in enumerate(r["sources"], 1):
        with st.expander(f"[{i}] 《{s['doc_id']}》 第 {s['page']} 页"):
            st.write(s["text"])
