"""Response cache: Redis (shared by gateway replicas) with an in-process LRU fallback.

Key = 归一化 query（去语气词/标点/空白，小写）+ 路由；只缓存“与用户个人订单无关”的通用问答（FAQ 类），
带个人上下文（订单状态、签收天数）的请求不走缓存，避免答非所问。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import OrderedDict
from typing import Optional

FILLER = r"^(你好|您好|在吗|请问|亲|客服|问下|问一下)[，,？?\s]*|[啊呢吧呀哦嘛]+$"
PUNCT = r"[\s,，。.!！?？~、；;：:“”\"'（）()]+"


def normalize(q: str) -> str:
    q = q.strip()
    for _ in range(3):
        q = re.sub(FILLER, "", q)
    return re.sub(PUNCT, "", q).lower()


def make_key(route: str, q: str) -> str:
    return "llmgw:" + route + ":" + hashlib.md5(normalize(q).encode("utf-8")).hexdigest()


class LocalLRU:
    def __init__(self, capacity: int = 10000):
        self.d: "OrderedDict[str, tuple]" = OrderedDict()
        self.cap = capacity

    async def get(self, k):
        v = self.d.get(k)
        if not v:
            return None
        exp, val = v
        if exp < time.time():
            self.d.pop(k, None)
            return None
        self.d.move_to_end(k)
        return val

    async def set(self, k, val, ttl):
        self.d[k] = (time.time() + ttl, val)
        self.d.move_to_end(k)
        while len(self.d) > self.cap:
            self.d.popitem(last=False)

    async def ping(self):
        return True


class ResponseCache:
    def __init__(self, redis_url: Optional[str], ttl: int = 3600):
        self.ttl = ttl
        self.backend_name = "local-lru"
        self.backend = LocalLRU()
        self.hits = self.misses = 0
        if redis_url:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(redis_url, decode_responses=True, socket_connect_timeout=0.5)
            except Exception:
                self._redis = None
        else:
            self._redis = None

    async def init(self):
        if self._redis is not None:
            try:
                await self._redis.ping()
                self.backend, self.backend_name = self._redis, "redis"
            except Exception:
                self.backend_name = "local-lru(redis unavailable)"
        return self.backend_name

    async def get(self, route: str, q: str) -> Optional[dict]:
        v = await self.backend.get(make_key(route, q))
        if v is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(v) if isinstance(v, str) else v

    async def set(self, route: str, q: str, val: dict, ttl: Optional[int] = None):
        payload = json.dumps(val, ensure_ascii=False)
        k = make_key(route, q)
        if isinstance(self.backend, LocalLRU):
            await self.backend.set(k, payload, ttl or self.ttl)
        else:
            await self.backend.set(k, payload, ex=ttl or self.ttl)

    @property
    def hit_rate(self) -> float:
        tot = self.hits + self.misses
        return self.hits / tot if tot else 0.0
