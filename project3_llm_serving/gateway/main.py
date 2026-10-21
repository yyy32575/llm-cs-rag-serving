"""FastAPI gateway: 限流 -> 意图路由 -> 缓存 -> (RAG 检索) -> LLM（超时/熔断/降级）-> 日志与监控。

  uvicorn gateway.main:app --host 0.0.0.0 --port 9000 --workers 1
  curl -X POST localhost:9000/chat -H 'Content-Type: application/json' -d '{"user_id":"u1","query":"双11能叠加优惠券吗"}'

Endpoints
  POST /chat                    业务入口（客服 / 研报自动路由）
  POST /v1/chat/completions     OpenAI 兼容透传（带限流+监控，支持 stream），便于压测/接入现有 SDK
  GET  /stats                   滚动 60s：QPS、平均/P50/P95/P99 延迟、错误率、缓存命中率、熔断状态
  GET  /metrics                 Prometheus
  GET  /healthz
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from .cache import ResponseCache
from .config import ROOT, load_settings
from .degrade import FALLBACK_TEXT, CircuitBreaker
from .llm_client import LLMClient, LLMError
from .metrics import CONTENT_TYPE_LATEST, RollingStats, prometheus_payload
from .ratelimit import RateLimiter
from .router import route as route_intent

CS_SYSTEM = ("你是某电商平台的智能客服“小智”。请严格依据平台售后与物流规则回答用户问题：\n"
             "1. 先共情/致歉（如适用），再给出明确的政策依据，最后告诉用户下一步具体操作路径；\n"
             "2. 不确定或涉及赔偿认定、威胁投诉升级的问题，礼貌说明并转人工客服；\n"
             "3. 不要编造政策和商品参数，不要重复啰嗦，回答控制在150字以内。")
RAG_SYSTEM = ("你是严谨的金融研究助理。只能依据给定的【证据】回答问题，不得使用证据之外的知识。"
              "回答中出现的每个数字后面都要用 [编号] 标注来源证据；如果证据不足以回答，"
              "请直接回答“资料中未找到相关信息，无法回答”。回答简洁，不超过120字。")

cfg = load_settings()
app = FastAPI(title="LLM Serving Gateway")
os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
req_log = logging.getLogger("gateway.requests")
req_log.setLevel(logging.INFO)
_fh = logging.FileHandler(os.path.join(ROOT, "logs", "requests.jsonl"), encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(message)s"))
req_log.addHandler(_fh)
req_log.propagate = False

S = {}  # singletons


class ChatReq(BaseModel):
    query: str
    user_id: str = ""
    context: Optional[str] = None  # 订单背景信息（类目/签收天数/订单状态）——有个人上下文不走缓存
    max_tokens: Optional[int] = None


@app.on_event("startup")
async def startup():
    S["cache"] = ResponseCache(cfg.get("redis_url"), cfg["cache_ttl_s"])
    S["cache_backend"] = await S["cache"].init()
    rl = cfg["rate_limit"]
    S["limiter"] = RateLimiter(rl["global_qps"], rl["global_burst"], rl["per_user_qps"], rl["per_user_burst"])
    S["breaker"] = CircuitBreaker(**cfg["circuit_breaker"])
    S["stats"] = RollingStats(60)
    t = cfg["request_timeout_s"]
    S["cs"] = LLMClient(cfg["cs_model_url"], cfg["cs_model_name"], t)
    S["rag_llm"] = LLMClient(cfg["rag_model_url"], cfg["rag_model_name"], t)
    S["fallback"] = LLMClient(cfg["fallback_model_url"], cfg["fallback_model_name"], t)
    S["http"] = httpx.AsyncClient(timeout=3.0)
    for item in cfg.get("hot_questions") or []:  # 热点问题预热
        await S["cache"].set("cs", item["q"], {"answer": item["a"], "route": "cs", "source": "hot"},
                             cfg["hot_cache_ttl_s"])


@app.on_event("shutdown")
async def shutdown():
    for k in ("cs", "rag_llm", "fallback"):
        await S[k].close()
    await S["http"].aclose()


def _log(rec: dict):
    req_log.info(json.dumps(rec, ensure_ascii=False))


async def _rag_messages(query: str):
    r = await S["http"].post(cfg["rag_service_url"].rstrip("/") + "/rag/query",
                             json={"query": query, "retrieve_only": True})
    r.raise_for_status()
    j = r.json()
    srcs = j.get("sources", [])
    scores = [s for s in (j.get("rerank_scores") or []) if s is not None]
    ctx = "\n".join(f"[{i}]《{s['doc_id']}》第{s['page']}页：{s['text']}" for i, s in enumerate(srcs, 1))
    msgs = [{"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": f"【证据】\n{ctx}\n\n【问题】{query}\n【回答】"}]
    best = max(scores) if scores else None
    if j.get("should_refuse"):
        best = -1.0  # 约束校验未通过（年份/主体/指标不在证据中）-> 直接拒答
    return msgs, srcs, best


async def _call_with_degrade(primary: LLMClient, messages, max_tokens):
    """主模型 -> (失败/熔断) 降级小模型(更短 max_tokens) -> 兜底话术。返回 (result, served_by)。"""
    br: CircuitBreaker = S["breaker"]
    t_budget = cfg["request_timeout_s"]
    if br.use_primary():
        t0 = time.perf_counter()
        try:
            res = await primary.chat(messages, max_tokens, timeout_s=t_budget)
            br.record(True, (time.perf_counter() - t0) * 1000)
            return res, "primary"
        except LLMError as e:
            br.record(False, (time.perf_counter() - t0) * 1000)
            err = str(e)
    else:
        err = "circuit_open"
    try:
        res = await S["fallback"].chat(messages, max(64, max_tokens // 2), timeout_s=t_budget / 2)
        res["degrade_reason"] = err
        return res, "fallback"
    except LLMError as e2:
        return {"text": FALLBACK_TEXT, "prompt_tokens": 0, "completion_tokens": 0,
                "degrade_reason": f"{err};fallback:{e2}"}, "canned"


@app.post("/chat")
async def chat(req: ChatReq):
    rid, t0 = uuid.uuid4().hex[:12], time.perf_counter()
    if not S["limiter"].allow_user(req.user_id):
        S["stats"].add(0, False, "ratelimited", False, 0, 0)
        return JSONResponse({"error": "rate_limited_user", "request_id": rid}, status_code=429)
    route, conf = route_intent(req.query)
    cacheable = not req.context
    if cacheable:
        hit = await S["cache"].get(route, req.query)
        if hit:
            ms = (time.perf_counter() - t0) * 1000
            S["stats"].add(ms, True, route, True, 0, 0)
            _log({"rid": rid, "route": route, "cache": "hit", "latency_ms": round(ms, 2), "status": "ok"})
            return {**hit, "request_id": rid, "cache_hit": True, "latency_ms": round(ms, 2)}
    if not S["limiter"].allow_global():  # 缓存未命中、需要占用 GPU 的请求才消耗全局令牌
        S["stats"].add(0, False, "ratelimited", False, 0, 0)
        return JSONResponse({"error": "rate_limited", "request_id": rid}, status_code=429)

    sources, top_score = [], None
    if route == "rag":
        try:
            messages, sources, top_score = await _rag_messages(req.query)
        except Exception as e:  # RAG 服务不可用：不瞎答，直接说明
            ms = (time.perf_counter() - t0) * 1000
            S["stats"].add(ms, False, route, False, 0, 0)
            _log({"rid": rid, "route": route, "status": "rag_unavailable", "err": str(e)[:100]})
            return JSONResponse({"answer": "研报检索服务暂时不可用，请稍后再试。", "route": route, "request_id": rid},
                                status_code=503)
        primary = S["rag_llm"]
        if top_score is not None and top_score < 0.15:
            res, served = {"text": "资料中未找到相关信息，无法回答。", "prompt_tokens": 0, "completion_tokens": 0}, "refuse"
        else:
            res, served = await _call_with_degrade(primary, messages, req.max_tokens or cfg["max_tokens"])
    else:
        user = f"【背景信息】{req.context}\n【用户问题】{req.query}" if req.context else req.query
        messages = [{"role": "system", "content": CS_SYSTEM}, {"role": "user", "content": user}]
        res, served = await _call_with_degrade(S["cs"], messages, req.max_tokens or cfg["max_tokens"])

    ms = (time.perf_counter() - t0) * 1000
    ok = served in ("primary", "fallback", "refuse")
    S["stats"].add(ms, ok, route, False, res.get("prompt_tokens", 0), res.get("completion_tokens", 0))
    out = {"answer": res["text"], "route": route, "route_conf": round(conf, 2), "served_by": served,
           "request_id": rid, "cache_hit": False, "latency_ms": round(ms, 1),
           "sources": [{"doc_id": s["doc_id"], "page": s["page"]} for s in sources]}
    if res.get("degrade_reason"):
        out["degrade_reason"] = res["degrade_reason"]
    _log({"rid": rid, "route": route, "served_by": served, "cache": "miss", "latency_ms": round(ms, 1),
          "input_tokens": res.get("prompt_tokens", 0), "output_tokens": res.get("completion_tokens", 0),
          "status": "ok" if ok else "degraded", "degrade_reason": res.get("degrade_reason")})
    if served == "primary" and cacheable:
        await S["cache"].set(route, req.query, {"answer": res["text"], "route": route, "source": "llm",
                                                "sources": out["sources"]})
    return out


@app.post("/v1/chat/completions")
async def openai_passthrough(request: Request):
    body = await request.json()
    user = request.headers.get("x-user-id", "")
    if not S["limiter"].allow(user):
        S["stats"].add(0, False, "ratelimited", False, 0, 0)
        return JSONResponse({"error": {"message": "rate_limited", "type": "rate_limit"}}, status_code=429)
    client: LLMClient = S["cs"] if S["breaker"].use_primary() else S["fallback"]
    t0 = time.perf_counter()
    if body.get("stream"):
        async def gen():
            ok = True
            try:
                async with client.stream(body) as r:
                    async for chunk in r.aiter_raw():
                        yield chunk
            except Exception:
                ok = False
                yield b'data: {"error": "upstream"}\n\n'
            ms = (time.perf_counter() - t0) * 1000
            S["breaker"].record(ok, ms)
            S["stats"].add(ms, ok, "passthrough", False, 0, 0)
        return StreamingResponse(gen(), media_type="text/event-stream")
    try:
        r = await client.http.post(f"{client.base_url}/chat/completions", json={**body, "model": client.model},
                                   timeout=cfg["request_timeout_s"])
        ok = r.status_code == 200
        ms = (time.perf_counter() - t0) * 1000
        S["breaker"].record(ok, ms)
        u = (r.json().get("usage") or {}) if ok else {}
        S["stats"].add(ms, ok, "passthrough", False, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        return Response(r.content, status_code=r.status_code, media_type="application/json")
    except httpx.HTTPError as e:
        ms = (time.perf_counter() - t0) * 1000
        S["breaker"].record(False, ms)
        S["stats"].add(ms, False, "passthrough", False, 0, 0)
        return JSONResponse({"error": {"message": f"upstream {type(e).__name__}"}}, status_code=504)


@app.get("/stats")
async def stats():
    return {**S["stats"].snapshot(), "cache_backend": S["cache_backend"],
            "cache_hit_rate_total": round(S["cache"].hit_rate, 4), "rate_limited_total": S["limiter"].rejected,
            "circuit_breaker": S["breaker"].snapshot()}


@app.get("/metrics")
async def metrics():
    return Response(prometheus_payload(), media_type=CONTENT_TYPE_LATEST)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "cache": S.get("cache_backend")}


if __name__ == "__main__":
    import uvicorn

    asyncio.run(uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=9000)).serve())
