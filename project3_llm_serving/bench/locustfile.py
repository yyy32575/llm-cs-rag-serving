"""Locust alternative (Web UI 可视化压测)：
  locust -f bench/locustfile.py --host http://localhost:9000 -u 32 -r 4
"""
import json
import os
import random

from locust import HttpUser, between, task

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P1 = os.path.join(ROOT, "..", "project1_ecom_cs_sft_dpo", "data", "test", "test.jsonl")
QUERIES = [json.loads(x)["instruction"] for x in open(P1, encoding="utf-8")] if os.path.exists(P1) else ["什么时候发货"]


class ChatUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(7)
    def cs(self):
        self.client.post("/chat", json={"query": random.choice(QUERIES), "user_id": f"u{random.randint(0, 9999)}"})

    @task(3)
    def hot(self):
        self.client.post("/chat", json={"query": "双11能叠加优惠券吗", "user_id": f"u{random.randint(0, 9999)}"})
