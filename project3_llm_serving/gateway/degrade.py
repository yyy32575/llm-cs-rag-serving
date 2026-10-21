"""Circuit breaker + degradation policy.

状态机：CLOSED（正常）-> OPEN（错误率或 P95 超 SLO，走降级）-> HALF_OPEN（放少量请求探测主模型）-> CLOSED
降级动作：① 切到小模型（量化的 4B）② 缩短 max_tokens ③ 主备都不可用时返回兜底话术 + 转人工
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, Tuple

FALLBACK_TEXT = "亲，当前咨询人数较多，已为您记录问题并转接人工客服，请稍候~"


class CircuitBreaker:
    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, window_s=30, min_requests=20, error_rate_threshold=0.3, p95_slo_ms=2000, open_seconds=20):
        self.window_s, self.min_requests = window_s, min_requests
        self.err_th, self.slo, self.open_s = error_rate_threshold, p95_slo_ms, open_seconds
        self.events: Deque[Tuple[float, bool, float]] = deque()
        self.state, self.opened_at, self.reason = self.CLOSED, 0.0, ""
        self.trips = 0

    def _trim(self):
        now = time.time()
        while self.events and self.events[0][0] < now - self.window_s:
            self.events.popleft()

    def record(self, ok: bool, latency_ms: float):
        self.events.append((time.time(), ok, latency_ms))
        self._trim()
        if self.state == self.HALF_OPEN:
            if ok and latency_ms < self.slo:
                self.state, self.reason = self.CLOSED, ""
                self.events.clear()
            else:
                self._open("half-open probe failed")
            return
        if self.state == self.CLOSED and len(self.events) >= self.min_requests:
            errs = sum(1 for _, o, _ in self.events if not o)
            lats = sorted(l for _, o, l in self.events if o)
            p95 = lats[int(0.95 * (len(lats) - 1))] if lats else 0
            if errs / len(self.events) > self.err_th:
                self._open(f"error_rate={errs / len(self.events):.2f}")
            elif p95 > self.slo:
                self._open(f"p95={p95:.0f}ms>{self.slo}ms")

    def _open(self, reason):
        self.state, self.opened_at, self.reason = self.OPEN, time.time(), reason
        self.trips += 1

    def use_primary(self) -> bool:
        if self.state == self.OPEN and time.time() - self.opened_at > self.open_s:
            self.state = self.HALF_OPEN
        return self.state in (self.CLOSED, self.HALF_OPEN)

    def snapshot(self):
        return {"state": self.state, "reason": self.reason, "trips": self.trips, "window_events": len(self.events)}
