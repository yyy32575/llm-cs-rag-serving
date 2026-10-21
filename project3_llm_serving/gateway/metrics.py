"""Metrics: Prometheus (if installed) + in-process rolling stats (每分钟 QPS / 平均延迟 / 错误率 / P95)."""
from __future__ import annotations

import time
from collections import deque

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    REQS = Counter("llmgw_requests_total", "requests", ["route", "status", "cache"])
    LAT = Histogram("llmgw_latency_seconds", "end-to-end latency", ["route"],
                    buckets=(0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 5, 8, 13))
    TOK = Counter("llmgw_tokens_total", "tokens", ["kind"])
    CB = Gauge("llmgw_circuit_open", "1 if circuit breaker open")
    PROM = True
except ImportError:  # pragma: no cover
    PROM = False
    CONTENT_TYPE_LATEST = "text/plain"


class RollingStats:
    def __init__(self, window_s: int = 60):
        self.window_s = window_s
        self.ev = deque()  # (ts, latency_ms, ok, route, cache_hit, in_tok, out_tok)

    def add(self, latency_ms, ok, route, cache_hit, in_tok, out_tok):
        now = time.time()
        self.ev.append((now, latency_ms, ok, route, cache_hit, in_tok, out_tok))
        while self.ev and self.ev[0][0] < now - self.window_s:
            self.ev.popleft()
        if PROM:
            REQS.labels(route, "ok" if ok else "error", "hit" if cache_hit else "miss").inc()
            LAT.labels(route).observe(latency_ms / 1000)
            TOK.labels("input").inc(in_tok)
            TOK.labels("output").inc(out_tok)

    def snapshot(self):
        now = time.time()
        ev = [e for e in self.ev if e[0] >= now - self.window_s]
        n = len(ev)
        lats = sorted(e[1] for e in ev)

        def pct(p):
            return round(lats[int(p * (len(lats) - 1))], 1) if lats else 0

        return {"window_s": self.window_s, "requests": n, "qps": round(n / self.window_s, 2),
                "avg_latency_ms": round(sum(lats) / n, 1) if n else 0, "p50_ms": pct(0.5), "p95_ms": pct(0.95),
                "p99_ms": pct(0.99), "error_rate": round(sum(1 for e in ev if not e[2]) / n, 4) if n else 0,
                "cache_hit_rate": round(sum(1 for e in ev if e[4]) / n, 4) if n else 0,
                "routes": {r: sum(1 for e in ev if e[3] == r) for r in {e[3] for e in ev}}}


def prometheus_payload():
    if not PROM:
        return b"prometheus_client not installed"
    return generate_latest()
