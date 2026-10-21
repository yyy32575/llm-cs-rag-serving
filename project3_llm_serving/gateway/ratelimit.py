"""Token-bucket rate limiting (global + per user). 超过阈值直接 429，保护 GPU 不被打爆。

单机用进程内令牌桶；多副本部署时可换成 Redis + Lua 原子脚本（见 REDIS_LUA）。
"""
from __future__ import annotations

import time
from typing import Dict


class TokenBucket:
    def __init__(self, rate: float, burst: float):
        self.rate, self.capacity = float(rate), float(burst)
        self.tokens, self.ts = float(burst), time.monotonic()

    def allow(self, cost: float = 1.0) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.ts) * self.rate)
        self.ts = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


class RateLimiter:
    def __init__(self, global_qps: float, global_burst: float, per_user_qps: float, per_user_burst: float):
        self.g = TokenBucket(global_qps, global_burst)
        self.per_user_qps, self.per_user_burst = per_user_qps, per_user_burst
        self.users: Dict[str, TokenBucket] = {}
        self.rejected = 0

    def allow_user(self, user: str = "") -> bool:
        """单用户防刷（先于缓存检查）。"""
        if not user:
            return True
        b = self.users.get(user)
        if b is None:
            if len(self.users) > 100000:
                self.users.clear()
            b = self.users[user] = TokenBucket(self.per_user_qps, self.per_user_burst)
        if not b.allow():
            self.rejected += 1
            return False
        return True

    def allow_global(self) -> bool:
        """全局令牌桶：只对“要打到 GPU 的请求”计数，缓存命中不消耗令牌。"""
        if not self.g.allow():
            self.rejected += 1
            return False
        return True

    def allow(self, user: str = "") -> bool:
        return self.allow_user(user) and self.allow_global()


REDIS_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1]); local cap = tonumber(ARGV[2]); local now = tonumber(ARGV[3])
local v = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(v[1]) or cap; local ts = tonumber(v[2]) or now
tokens = math.min(cap, tokens + (now - ts) * rate)
local ok = 0
if tokens >= 1 then tokens = tokens - 1; ok = 1 end
redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, 60)
return ok
"""
