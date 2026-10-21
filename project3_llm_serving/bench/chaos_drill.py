"""W11: 故障演练 (chaos drill) —— 对网关注入故障，记录 现象 / 排查信号 / 恢复时间，生成故障记录表。

场景（与文档故障表一一对应）：
  1. upstream_error : 上游 vLLM 报错（模拟 OOM 进程被 kill / 5xx）       -> 熔断打开 -> 降级小模型
  2. slow_decode    : 解码变慢（模拟 KV Cache 被长请求占满、抢占/重算）   -> 超时 + P95 超 SLO -> 降级
  3. burst_traffic  : 双11 流量突增 3 倍                                   -> 令牌桶 429 保护，正常请求 P95 可控
  4. hot_query      : 热门问题集中访问                                     -> 缓存命中率
需要：mock_llm_server(:8000 主, :8001 降级) 或真实 vLLM；网关 :9000

  python bench/chaos_drill.py --gateway http://localhost:9000 --primary_admin http://localhost:8000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERIES = ["这个拆封了还能退吗", "快递到哪了", "能开发票吗", "尺码不合适能换吗", "保修多久", "定金能退吗", "我要投诉快递员",
           "退货运费谁出", "买完降价了能补差价吗", "地址填错了怎么办"]


async def fire(client, gw, n, conc, hot=False, ctx=True):
    sem = asyncio.Semaphore(conc)
    res = []

    async def one(i):
        async with sem:
            q = "双11能叠加优惠券吗" if hot else random.choice(QUERIES)
            body = {"query": q, "user_id": f"u{random.randint(0, 10 ** 6)}"}
            if ctx and not hot:
                body["context"] = f"订单号尾号{i}"  # 带个人上下文，不走缓存，确保真正打到模型
            t0 = time.perf_counter()
            try:
                r = await client.post(f"{gw}/chat", json=body)
                j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                res.append({"t": time.time(), "status": r.status_code, "ms": (time.perf_counter() - t0) * 1000,
                            "served_by": j.get("served_by"), "cache": j.get("cache_hit")})
            except Exception:
                res.append({"t": time.time(), "status": -1, "ms": (time.perf_counter() - t0) * 1000,
                            "served_by": None, "cache": None})

    await asyncio.gather(*[one(i) for i in range(n)])
    return res


def summarize(res):
    ok = [r for r in res if r["status"] == 200]
    lat = sorted(r["ms"] for r in ok)
    p95 = round(lat[int(0.95 * (len(lat) - 1))], 1) if lat else None
    by = {}
    for r in res:
        k = r["served_by"] or (f"http_{r['status']}")
        by[k] = by.get(k, 0) + 1
    return {"n": len(res), "success_rate": round(len(ok) / max(1, len(res)), 4), "p95_ms": p95,
            "served_by": by, "http_429": sum(r["status"] == 429 for r in res),
            "cache_hit_rate": round(sum(1 for r in ok if r["cache"]) / max(1, len(ok)), 4)}


async def main_async(a):
    out = {}
    async with httpx.AsyncClient(timeout=30) as c:
        await c.post(f"{a.primary_admin}/admin/fault", json={"mode": "none"})
        out["0_baseline"] = summarize(await fire(c, a.gateway, 120, 8))
        print("baseline", out["0_baseline"])

        # 1) upstream error
        await c.post(f"{a.primary_admin}/admin/fault", json={"mode": "error", "duration_s": a.fault_s})
        t0 = time.time()
        r1 = await fire(c, a.gateway, 160, 8)
        st = (await c.get(f"{a.gateway}/stats")).json()
        out["1_upstream_error"] = {**summarize(r1), "breaker": st["circuit_breaker"]}
        print("upstream_error", out["1_upstream_error"])
        await asyncio.sleep(max(0, a.fault_s - (time.time() - t0)) + 1)
        # recovery: wait for breaker half-open -> closed
        t_rec = time.time()
        while time.time() - t_rec < 90:
            await fire(c, a.gateway, 8, 4)
            if (await c.get(f"{a.gateway}/stats")).json()["circuit_breaker"]["state"] == "closed":
                break
            await asyncio.sleep(2)
        out["1_upstream_error"]["recovery_s_after_fault_cleared"] = round(time.time() - t_rec, 1)

        # 2) slow decode
        await c.post(f"{a.primary_admin}/admin/fault", json={"mode": "slow", "duration_s": a.fault_s})
        r2 = await fire(c, a.gateway, 120, 8)
        st = (await c.get(f"{a.gateway}/stats")).json()
        out["2_slow_decode"] = {**summarize(r2), "breaker": st["circuit_breaker"]}
        print("slow_decode", out["2_slow_decode"])
        await c.post(f"{a.primary_admin}/admin/fault", json={"mode": "none"})
        await asyncio.sleep(a.fault_s / 2)

        # 3) burst: 3x traffic
        out["3_burst_3x"] = summarize(await fire(c, a.gateway, 600, 96))
        print("burst", out["3_burst_3x"])
        await asyncio.sleep(5)
        # 4) hot query
        out["4_hot_query"] = summarize(await fire(c, a.gateway, 200, 16, hot=True))
        print("hot", out["4_hot_query"])
        out["final_stats"] = (await c.get(f"{a.gateway}/stats")).json()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", f"chaos_{a.tag}.json"), "w"), ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://localhost:9000")
    ap.add_argument("--primary_admin", default="http://localhost:8000")
    ap.add_argument("--fault_s", type=float, default=20)
    ap.add_argument("--tag", default="mock")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
