"""RAG HTTP service, called by the project-3 gateway when the intent router decides “研报问答”.

  uvicorn app.api:app --host 0.0.0.0 --port 8100
  curl -X POST localhost:8100/rag/query -H 'Content-Type: application/json' -d '{"query":"宁川动力2025年三季度毛利率"}'
"""
import os
import sys
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from finrag.generator import check_constraints  # noqa: E402
from finrag.pipeline import build_default_pipeline  # noqa: E402

app = FastAPI(title="FinReport RAG")
_pipe = None


def pipe():
    global _pipe
    if _pipe is None:
        _pipe = build_default_pipeline(ROOT, with_generator=os.environ.get("NO_LLM") != "1")
    return _pipe


class Query(BaseModel):
    query: str
    final_k: Optional[int] = None
    retrieve_only: bool = False


@app.on_event("startup")
def _warmup():
    pipe()


@app.get("/healthz")
def healthz():
    return {"status": "ok", "chunks": len(pipe().chunks)}


@app.post("/rag/query")
def rag_query(body: Query):
    p = pipe()
    if body.final_k:
        p.cfg["final_k"] = body.final_k
    if body.retrieve_only:
        r = p.retrieve(body.query)
        top = [p.chunks[i] for i in r["idxs"][:p.cfg["final_k"]]]
        missing = check_constraints(body.query, top) if top else ["empty"]
        return {"sources": [{"doc_id": c.doc_id, "page": c.page, "text": c.text} for c in top],
                "rerank_scores": r["rerank_scores"][:p.cfg["final_k"]], "missing_constraints": missing,
                "should_refuse": bool(missing)}
    return p.answer(body.query)
