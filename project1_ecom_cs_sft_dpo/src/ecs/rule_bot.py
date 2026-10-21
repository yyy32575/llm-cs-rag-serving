"""Rule-based FAQ bot = the *online baseline* (现有规则引擎只能解决约 40%).

Mimics a typical keyword-matching customer-service engine:
  * matches the user query against keyword rules -> returns ONE fixed FAQ answer;
  * ignores background info (类目 / 签收天数 / 订单状态), so context-dependent questions are often wrong;
  * no match -> "转人工客服".
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .sop import INTENT_BY_NAME, POLICY

# (keywords(all must hit any-of-group), intent, default slots)
# NOTE: keywords are written ONLY from historical high-frequency phrasings (= train templates),
# never from held-out test phrasings -> this reproduces “规则写死、泛化不到新问法”的线上现状。
RULES: List[Tuple[List[List[str]], str, Dict]] = [
    ([["12315", "消协", "赔偿", "领导", "假一赔十", "起诉", "曝光"]], "complain_escalate", {}),
    ([["运费", "邮费"]], "return_fee", {"insurance": "否"}),
    ([["尺码", "码数", "换大", "换小"]], "exchange_size", {"category": "服装", "days": "3"}),
    ([["拆", "开封", "打开", "开箱"]], "return_opened", {"category": "服装", "days": "3"}),
    ([["无理由", "不想要", "不喜欢", "买错"]], "return_no_reason", {"category": "服装", "days": "3"}),
    ([["退款"], ["到账", "多久", "几天", "没到"]], "refund_time", {}),
    ([["快递员", "驿站", "不送上门"]], "complain_courier", {}),
    ([["压坏", "破", "碎", "烂", "裂"]], "complain_damage", {}),
    ([["质量", "瑕疵", "坏的"]], "return_quality", {"days": "3"}),
    ([["定金"]], "deposit_refund", {}),
    ([["价保", "差价", "降价"]], "price_protect", {"days": "3"}),
    ([["券", "叠加", "满减"]], "coupon_stack", {}),
    ([["发票"]], "invoice", {}),
    ([["保修", "维修", "怎么修", "质保"]], "warranty", {}),
    ([["地址"]], "change_address", {"status": "未发货", "courier": "中通"}),
    ([["指定", "顺丰", "京东"]], "designate_courier", {}),
    ([["还没到", "太慢", "不动", "还没收到", "还没送到"]], "delivery_delay", {}),
    ([["物流", "到哪"]], "track", {"status": "运输中", "courier": "中通"}),
    ([["发货", "发吗"]], "ship_time", {}),
]

FALLBACK = "您好，您的问题我暂时无法解答，正在为您转人工客服，请稍候。"


class RuleBot:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.cache = {}
        for kws, intent, slots in RULES:
            self.cache[intent] = INTENT_BY_NAME[intent].build(slots, random.Random(0), POLICY)["answer"]

    def answer(self, ex: Dict) -> str:
        q = ex["instruction"]
        for kws, intent, _ in RULES:
            if all(any(k in q for k in group) for group in kws):
                return self.cache[intent]
        return FALLBACK

    def generate(self, examples, **_):
        return [[self.answer(e)] for e in examples]
