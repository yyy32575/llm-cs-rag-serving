"""Mock OpenAI-compatible LLM server —— 没有 GPU 时验证网关功能（缓存/限流/超时/熔断/降级）和压测脚本本身。

注意：这里的延迟是“参数化模拟”（prefill 线性于 prompt 长度；decode 每步耗时随并发增大，近似 continuous batching
的访存瓶颈），**不代表任何真实 GPU 的性能**。真实性能数据必须对 vLLM 实测（scripts/start_vllm.sh + bench/load_test.py）。

  python mock/mock_llm_server.py --port 8000 --step_ms 25 --max_seqs 64
  curl -X POST localhost:8000/admin/fault -d '{"mode":"slow","duration_s":30}'   # 故障注入：slow / error / oom / none
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="mock-llm")
ARGS = None
STATE = {"active": 0, "fault": "none", "fault_until": 0.0, "served": 0}
SEM: asyncio.Semaphore = None

ANSWERS = ["亲，非常理解您的心情，根据平台规则，您可以在【我的订单】-【申请售后】中提交申请，我们会在审核通过后第一时间为您处理。",
           "您好，双11期间平台券和店铺券可以叠加使用，但同类型的优惠券不能叠加，下单时系统会自动为您选择最优组合。",
           "根据证据[1]，公司2025年第三季度单季毛利率为24.8%[1]，环比提升1.2个百分点。"]


def _fault():
    if STATE["fault"] != "none" and time.time() > STATE["fault_until"]:
        STATE["fault"] = "none"
    return STATE["fault"]


def _ntok(text: str) -> int:
    return max(1, int(len(text) * 0.75))


async def _generate(prompt_tokens: int, max_tokens: int):
    """Yields tokens with simulated timing."""
    fault = _fault()
    if fault == "error":
        raise RuntimeError("injected upstream error")
    if fault == "oom" and STATE["active"] > ARGS.max_seqs // 4:
        raise MemoryError("CUDA out of memory (injected)")
    await asyncio.sleep(ARGS.prefill_base_ms / 1000 + prompt_tokens * ARGS.prefill_ms_per_tok / 1000)
    n_out = min(max_tokens, max(8, int(random.gauss(ARGS.mean_out, ARGS.mean_out * 0.3))))
    text = random.choice(ANSWERS)
    for i in range(n_out):
        slow = 6.0 if _fault() == "slow" else 1.0
        step = ARGS.step_ms * (1 + ARGS.batch_penalty * max(0, STATE["active"] - 1)) * slow
        await asyncio.sleep(step / 1000)
        yield text[i % len(text)]


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    prompt = "".join(m.get("content", "") for m in body.get("messages", []))
    pt, mt = _ntok(prompt), int(body.get("max_tokens") or 256)
    model = body.get("model", "mock")
    rid = "chatcmpl-" + uuid.uuid4().hex[:10]

    async def run_collect():
        async with SEM:
            STATE["active"] += 1
            try:
                return "".join([t async for t in _generate(pt, mt)])
            finally:
                STATE["active"] -= 1
                STATE["served"] += 1

    if body.get("stream"):
        async def sse():
            async with SEM:
                STATE["active"] += 1
                try:
                    async for t in _generate(pt, mt):
                        d = {"id": rid, "object": "chat.completion.chunk", "model": model,
                             "choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]}
                        yield f"data: {json.dumps(d, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                finally:
                    STATE["active"] -= 1
                    STATE["served"] += 1
        return StreamingResponse(sse(), media_type="text/event-stream")
    try:
        text = await run_collect()
    except (RuntimeError, MemoryError) as e:
        return JSONResponse({"error": {"message": str(e)}}, status_code=500)
    return {"id": rid, "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": pt, "completion_tokens": len(text), "total_tokens": pt + len(text)}}


@app.post("/admin/fault")
async def fault(request: Request):
    b = await request.json()
    STATE["fault"], STATE["fault_until"] = b.get("mode", "none"), time.time() + float(b.get("duration_s", 30))
    return STATE


@app.get("/health")
@app.get("/v1/models")
async def health():
    return {"object": "list", "data": [{"id": ARGS.model_name}], "state": STATE}


@app.on_event("startup")
async def _init():
    global SEM
    SEM = asyncio.Semaphore(ARGS.max_seqs)  # 必须在服务的事件循环内创建


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model_name", default="cs-model")
    ap.add_argument("--step_ms", type=float, default=25.0, help="单序列 decode 每 token 耗时")
    ap.add_argument("--batch_penalty", type=float, default=0.03, help="每多一个并发序列，单步耗时增加比例")
    ap.add_argument("--prefill_base_ms", type=float, default=15.0)
    ap.add_argument("--prefill_ms_per_tok", type=float, default=0.08)
    ap.add_argument("--mean_out", type=int, default=90)
    ap.add_argument("--max_seqs", type=int, default=64, help="类似 vLLM --max-num-seqs，超出排队")
    ARGS = ap.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=ARGS.port, log_level="warning")


if __name__ == "__main__":
    main()
