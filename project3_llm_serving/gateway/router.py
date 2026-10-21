"""Intent routing: 研报/财务/政策问题 -> RAG 链路；其余 -> 客服模型。

规则优先（零延迟、可解释），规则未命中再用轻量打分；生产上可替换为一个小分类模型（如 bge-small + LR）。
"""
from __future__ import annotations

import re
from typing import Tuple

RESEARCH_KW = ["研报", "财报", "营收", "营业收入", "净利润", "毛利率", "净息差", "EPS", "PE", "市盈率", "季度", "季报",
               "年报", "同比", "环比", "市场份额", "装车量", "装机量", "出货量", "产能利用率", "政策", "补贴", "资本充足率",
               "行业报告", "盈利预测", "估值", "目标价"]
CS_KW = ["退货", "退款", "换货", "快递", "物流", "发货", "地址", "优惠券", "发票", "保修", "维修", "运费", "订单", "尺码",
         "价保", "差价", "定金", "投诉", "客服", "拆封", "签收"]
COMPANY_RE = re.compile(r"(动力|汽车|光能|微电子|装备|酒业|乳业|医药|银行|锂电|新能|半导体|集成|制药|生物)")


def route(query: str) -> Tuple[str, float]:
    """Return (route, confidence). route in {"rag", "cs"}"""
    q = query.upper()
    r = sum(k.upper() in q for k in RESEARCH_KW) + (1 if COMPANY_RE.search(query) else 0)
    c = sum(k in query for k in CS_KW)
    if r == c == 0:
        return "cs", 0.5
    if r > c:
        return "rag", r / (r + c)
    return "cs", c / (r + c)
