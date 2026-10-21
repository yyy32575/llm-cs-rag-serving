"""Async load tester (并发 1/4/8/16/32 阶梯压测) for vLLM / any OpenAI-compatible server / the gateway.

  # 直接压 vLLM（流式，测 TTFT）
  python bench/load_test.py --target openai --url http://localhost:8000/v1 --model cs-model --stream --tag vllm_awq
  # 压网关业务接口（含缓存/路由/降级）
  python bench/load_test.py --target gateway --url http://localhost:9000 --tag gateway --hot_ratio 0.3

Prompts: 默认混合项目一客服测试集 + 项目二研报问题，保持真实的输入长度分布；--hot_ratio 控制热点问题占比（测缓存收益）。
Outputs: results/loadtest_<tag>.json / .md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import subprocess
import threading
import time

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P1 = os.path.join(ROOT, "..", "project1_ecom_cs_sft_dpo", "data", "test", "test.jsonl")
P2 = os.path.join(ROOT, "..", "project2_fin_report_rag", "data", "eval", "qa.jsonl")
HOT = ["双11能叠加优惠券吗", "什么时候发货", "退款多久到账"]
SYSTEM = "你是某电商平台的智能客服“小智”。请严格依据平台售后与物流规则回答用户问题。"


def load_prompts(path=None, rag_ratio=0.2):
    rows = []
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]
    if os.path.exists(P1):
        with open(P1, encoding="utf-8") as f:
            for x in f:
                r = json.loads(x)
                rows.append({"query": r["instruction"], "context": r.get("input") or None})
    if os.path.exists(P2):
        with open(P2, encoding="utf-8") as f:
            rag = [{"query": json.loads(x)["query"], "context": None} for x in f if x.strip()]
        rows += random.Random(0).sample(rag, min(len(rag), int(len(rows) * rag_ratio) or len(rag)))
    return rows or [{"query": q, "context": None} for q in HOT]


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))], 1)


class GPUMonitor(threading.Thread):
    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval, self.peak_mb, self.util, self.stop_flag = interval, 0, [], False

    def run(self):
        while not self.stop_flag:
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                               "--format=csv,noheader,nounits"], text=True, timeout=2)
                mem, util = out.strip().splitlines()[0].split(",")
                self.peak_mb = max(self.peak_mb, int(mem))
                self.util.append(int(util))
            except Exception:
                return
            time.sleep(self.interval)


async def one_request(client, args, item):
    t0 = time.perf_counter()
    ttft, out_tok, status = None, 0, 200
    try:
        if args.target == "gateway":
            r = await client.post(f"{args.url}/chat", json={"query": item["query"], "context": item.get("context"),
                                                            "user_id": f"u{random.randint(0, 10 ** 6)}"})
            status = r.status_code
            if status == 200:
                out_tok = len(r.json().get("answer", ""))
        else:
            user = (f"【背景信息】{item['context']}\n【用户问题】{item['query']}" if item.get("context") else item["query"])
            body = {"model": args.model, "max_tokens": args.max_tokens, "temperature": 0.3,
                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                    "chat_template_kwargs": {"enable_thinking": False}}
            if args.stream:
                body["stream"] = True
                body["stream_options"] = {"include_usage": True}
                async with client.stream("POST", f"{args.url}/chat/completions", json=body) as r:
                    status = r.status_code
                    async for line in r.aiter_lines():
                        if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                            continue
                        d = json.loads(line[6:])
                        if d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                            if ttft is None:
                                ttft = (time.perf_counter() - t0) * 1000
                            out_tok += 1
                        if d.get("usage"):
                            out_tok = d["usage"].get("completion_tokens", out_tok)
                        if "error" in d:
                            status = 500
            else:
                r = await client.post(f"{args.url}/chat/completions", json=body)
                status = r.status_code
                if status == 200:
                    out_tok = (r.json().get("usage") or {}).get("completion_tokens", 0)
    except Exception:
        status = -1
    return {"latency_ms": (time.perf_counter() - t0) * 1000, "ttft_ms": ttft, "out_tokens": out_tok, "status": status}


async def run_level(args, prompts, conc):
    q = asyncio.Queue()
    rng = random.Random(conc)
    for _ in range(args.requests_per_level):
        q.put_nowait({"query": rng.choice(HOT), "context": None} if rng.random() < args.hot_ratio else rng.choice(prompts))
    res = []
    limits = httpx.Limits(max_connections=conc * 2, max_keepalive_connections=conc * 2)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        async def worker():
            while not q.empty():
                item = q.get_nowait()
                res.append(await one_request(client, args, item))

        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(conc)])
        el = time.perf_counter() - t0
    ok = [r for r in res if r["status"] == 200]
    lat = [r["latency_ms"] for r in ok]
    ttft = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    return {"concurrency": conc, "requests": len(res), "ok": len(ok), "QPS": round(len(ok) / el, 2),
            "P50_ms": pct(lat, .5), "P95_ms": pct(lat, .95), "P99_ms": pct(lat, .99),
            "avg_ms": round(statistics.mean(lat), 1) if lat else 0,
            "TTFT_P50_ms": pct(ttft, .5) if ttft else None, "TTFT_P95_ms": pct(ttft, .95) if ttft else None,
            "out_tok_per_s": round(sum(r["out_tokens"] for r in ok) / el, 1),
            "error_rate": round(1 - len(ok) / max(1, len(res)), 4),
            "http_429": sum(r["status"] == 429 for r in res), "elapsed_s": round(el, 1)}


async def main_async(args):
    prompts = load_prompts(args.prompts)
    rows = []
    for c in map(int, args.concurrency.split(",")):
        mon = GPUMonitor()
        mon.start()
        r = await run_level(args, prompts, c)
        mon.stop_flag = True
        r["gpu_peak_mem_MB"] = mon.peak_mb or None
        r["gpu_util_avg"] = round(sum(mon.util) / len(mon.util), 1) if mon.util else None
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False))
        await asyncio.sleep(args.cooldown)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    meta = {"tag": args.tag, "target": args.target, "url": args.url, "model": args.model, "stream": args.stream,
            "max_tokens": args.max_tokens, "hot_ratio": args.hot_ratio, "rows": rows}
    with open(os.path.join(ROOT, "results", f"loadtest_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    cols = ["concurrency", "QPS", "P50_ms", "P95_ms", "P99_ms", "TTFT_P50_ms", "out_tok_per_s", "error_rate", "http_429",
            "gpu_peak_mem_MB"]
    md = [f"# 压测报告 {args.tag}（target={args.target}, stream={args.stream}, max_tokens={args.max_tokens}）", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r.get(c)) for c in cols) + " |" for r in rows]
    with open(os.path.join(ROOT, "results", f"loadtest_{args.tag}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["openai", "gateway"], default="openai")
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="cs-model")
    ap.add_argument("--concurrency", default="1,4,8,16,32")
    ap.add_argument("--requests_per_level", type=int, default=128)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--hot_ratio", type=float, default=0.0)
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--cooldown", type=float, default=2)
    ap.add_argument("--tag", default="run")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
