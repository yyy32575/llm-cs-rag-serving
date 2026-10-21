"""Business SOP knowledge base for the e-commerce customer-service scenario.

All policy numbers live here so that:
  * data construction (SFT / DPO) uses one single source of truth;
  * the cleaning step can detect answers that contradict the *current* SOP
    (e.g. answers written against an outdated policy version);
  * evaluation can check key points / forbidden statements automatically.

Each intent provides
  - ``train_templates`` / ``test_templates``: disjoint user-query paraphrases
    (test templates are held out to avoid template leakage);
  - ``build(slots, rng, policy)`` -> dict(answer, keypoints, forbidden, transfer)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List

# --------------------------------------------------------------------------
# Policy (current version) and an outdated version used to inject conflicts
# --------------------------------------------------------------------------
POLICY_VERSION = "2026-09"
POLICY: Dict[str, int] = {
    "no_reason_days": 7,          # 7 天无理由退货
    "quality_exchange_days": 15,  # 质量问题 15 天内换货
    "warranty_months": 12,        # 整机保修 12 个月
    "price_protect_days": 15,     # 价保 15 天
    "ship_hours": 48,             # 付款后 48 小时内发货
    "refund_days": 3,             # 退款审核通过后 1-3 个工作日原路退回
    "fresh_report_hours": 48,     # 生鲜签收 48 小时内反馈
    "invoice_days": 7,            # 电子发票 7 个工作日内开具
}
OUTDATED_POLICY: Dict[str, int] = {
    "no_reason_days": 15,
    "quality_exchange_days": 30,
    "warranty_months": 6,
    "price_protect_days": 30,
    "ship_hours": 72,
    "refund_days": 7,
    "fresh_report_hours": 24,
    "invoice_days": 15,
}

CATEGORIES = {
    # name: (supports_no_reason_return, opened_returnable, note)
    "食品": (True, False, "食品类商品属于特殊商品，拆封后不支持无理由退货"),
    "生鲜": (False, False, "生鲜类商品不支持7天无理由退货"),
    "数码": (True, False, "数码产品激活或拆封后影响二次销售，不支持无理由退货"),
    "服装": (True, True, "服装类商品在吊牌完好、未洗涤的情况下支持无理由退换"),
    "家电": (True, False, "大家电拆封安装后不支持无理由退货"),
    "美妆": (True, False, "美妆个护类商品拆封后不支持无理由退货"),
    "图书": (True, True, "图书在无污损、无缺页的情况下支持无理由退货"),
    "定制商品": (False, False, "定制类商品属于个性化商品，不支持无理由退货"),
}

PRODUCTS = {
    "X1 Pro 手机": {"类目": "数码", "电池": "5000mAh", "快充": "支持100W有线快充", "重量": "约205g",
                   "屏幕": "6.7英寸 OLED 120Hz", "防水": "IP68 防尘防水", "存储": "12GB+256GB / 16GB+512GB"},
    "AirBuds 3 耳机": {"类目": "数码", "电池": "单次续航约7小时，配合充电盒约30小时", "快充": "充电10分钟可用2小时",
                     "重量": "单耳约4.8g", "防水": "IPX4 防汗防水", "降噪": "支持主动降噪，最高降噪深度48dB"},
    "智能空气炸锅 S6": {"类目": "家电", "容量": "6L", "功率": "1500W", "重量": "约5.2kg",
                     "材质": "内胆为不粘涂层，可拆卸清洗", "功能": "支持APP远程控制和12种预设菜单"},
    "轻跑 7 跑鞋": {"类目": "服装", "尺码": "39-46码，建议按平时尺码购买", "重量": "单只约228g（42码）",
                 "材质": "鞋面为工程网布，中底为超临界发泡材料", "适用": "适合日常慢跑和路跑训练"},
    "极地羽绒服": {"类目": "服装", "填充": "90%白鸭绒，充绒量约200g", "尺码": "S-3XL，偏大半码",
               "材质": "面料为防泼水尼龙", "适用": "适合零下15度左右的环境"},
    "每日坚果礼盒": {"类目": "食品", "规格": "30袋/盒，每袋25g", "保质期": "180天", "配料": "腰果、扁桃仁、榛子、蔓越莓干等",
                 "储存": "请置于阴凉干燥处，开封后尽快食用"},
    "云南高山蓝莓": {"类目": "生鲜", "规格": "125g×4盒，单果18mm+", "保质期": "冷藏条件下约7天",
                 "储存": "收货后请尽快冷藏，食用前清洗"},
    "水光精华 30ml": {"类目": "美妆", "容量": "30ml", "保质期": "未开封36个月，开封后建议12个月内用完",
                   "适用": "适合干性及混合性肌肤", "成分": "主要成分为透明质酸钠和烟酰胺"},
}

ORDER_STATUS = ["未发货", "已发货", "运输中", "派送中", "已签收", "物流异常"]
COURIERS = ["顺丰", "京东物流", "中通", "圆通", "韵达"]

CATEGORY_ITEMS = {
    "食品": ["每日坚果礼盒", "手撕牛肉干", "进口黑巧克力", "低脂燕麦片", "螺蛳粉5袋装"],
    "生鲜": ["云南高山蓝莓", "智利车厘子", "阳澄湖大闸蟹", "新鲜三文鱼"],
    "数码": ["X1 Pro 手机", "AirBuds 3 耳机", "智能手表 W5", "机械键盘 K8", "平板电脑 Pad 11"],
    "服装": ["轻跑 7 跑鞋", "极地羽绒服", "纯棉T恤", "直筒牛仔裤", "加绒卫衣"],
    "家电": ["智能空气炸锅 S6", "扫地机器人 R3", "破壁料理机", "电热水壶", "空气净化器 P2"],
    "美妆": ["水光精华 30ml", "防晒霜 SPF50", "保湿面霜", "口红礼盒"],
    "图书": ["Python编程从入门到实践", "深度学习入门", "三体全集"],
    "定制商品": ["刻字保温杯", "定制照片书", "定制情侣对戒"],
}
ALL_ITEMS = [(c, x) for c, xs in CATEGORY_ITEMS.items() for x in xs]

OPENERS = ["亲，", "您好，", "亲亲，", "您好呀，", ""]
CLOSERS = ["", "还有其他问题随时找我哦~", "如有其他疑问可以随时联系我们。", "祝您购物愉快！"]
USER_PREFIX = ["", "", "你好，", "在吗？", "亲，", "客服，", "请问", "问下，", "急！"]
USER_SUFFIX = ["", "？", "?", "啊", "呢", "。。", "，谢谢", "？？", "！"]

TRANSFER_HINTS = ["转人工", "人工客服", "专员"]


@dataclass
class Intent:
    name: str
    domain: str  # 退换货 / 物流 / 商品参数 / 投诉 / 售后 / 优惠活动
    train_templates: List[str]
    test_templates: List[str]
    slot_fn: Callable[[random.Random], Dict[str, str]]
    build: Callable[[Dict[str, str], random.Random, Dict[str, int]], Dict]
    weight: float = 1.0
    extra: Dict = field(default_factory=dict)


def _pick(rng: random.Random, xs):
    return xs[rng.randrange(len(xs))]


def _wrap(rng: random.Random, body: str) -> str:
    return f"{_pick(rng, OPENERS)}{body}{_pick(rng, CLOSERS)}".strip()


def _res(answer, keypoints, forbidden=None, transfer=False):
    return {"answer": answer, "keypoints": keypoints, "forbidden": forbidden or [], "transfer": transfer}


# --------------------------------------------------------------------------
# 退换货
# --------------------------------------------------------------------------
def _days(rng, hi=25):
    # 70% 在政策窗口内，30% 超窗口，逼模型学会“看背景信息再回答”
    return str(rng.randint(1, 7) if rng.random() < 0.7 else rng.randint(8, hi))


def _slot_category(rng):
    return {"category": _pick(rng, list(CATEGORIES.keys())), "days": _days(rng)}


def _beyond_no_reason(s, rng, p):
    body = (f"查询到您的商品已签收{s['days']}天，已超过{p['no_reason_days']}天无理由退货期限，暂时无法申请无理由退货。"
            f"如果商品存在质量问题，可以在【申请售后】中提交照片凭证，我们核实后为您处理维修或换新。")
    return _res(_wrap(rng, body), [["超过"], [f"{p['no_reason_days']}天"], ["质量问题"]],
                forbidden=["可以申请无理由退货", "支持无理由退货"])


def _build_return_opened(s, rng, p):
    cat = s["category"]
    no_reason, opened_ok, note = CATEGORIES[cat]
    if opened_ok and no_reason and int(s["days"]) > p["no_reason_days"]:
        return _beyond_no_reason(s, rng, p)
    if opened_ok and no_reason:
        body = (f"{note}。您的商品属于{cat}类，签收后{p['no_reason_days']}天内、商品不影响二次销售的情况下可以申请无理由退货，"
                f"请在【我的订单】-【申请售后】中提交，退货运费由买家承担，若购买了运费险可自动理赔。")
        return _res(_wrap(rng, body), [[f"{p['no_reason_days']}天"], ["申请售后", "我的订单"], ["运费"]],
                    forbidden=["不支持退货"])
    if not no_reason:
        body = (f"很抱歉，{note}。如果商品存在质量问题（如变质、破损、错发），请在签收后{p['fresh_report_hours']}小时内"
                f"拍照上传至【申请售后】，我们核实后会为您办理退款或补发。")
        return _res(_wrap(rng, body), [["不支持"], ["质量问题"], ["拍照", "照片", "凭证"]],
                    forbidden=["可以无理由退货", "支持无理由退货"])
    body = (f"很抱歉，{note}。如果是商品本身存在质量问题，您可以在签收后{p['quality_exchange_days']}天内提供照片凭证申请售后，"
            f"我们会为您优先处理退换。")
    return _res(_wrap(rng, body), [["不支持"], ["质量问题"], [f"{p['quality_exchange_days']}天"]],
                forbidden=["可以无理由退货", "支持无理由退货"])


def _build_return_no_reason(s, rng, p):
    cat = s["category"]
    no_reason, _, note = CATEGORIES[cat]
    if no_reason and int(s["days"]) > p["no_reason_days"]:
        return _beyond_no_reason(s, rng, p)
    if no_reason:
        body = (f"{cat}类商品支持{p['no_reason_days']}天无理由退货，请确保商品及包装完好、不影响二次销售，"
                f"在【我的订单】-【申请售后】选择“七天无理由”提交即可。退款会在仓库验收后{p['refund_days']}个工作日内原路退回。")
        return _res(_wrap(rng, body), [[f"{p['no_reason_days']}天"], ["申请售后"], ["原路退回", "退款"]])
    body = f"很抱歉，{note}。若商品存在质量问题，可以提供照片在【申请售后】中提交，我们核实后为您处理。"
    return _res(_wrap(rng, body), [["不支持"], ["质量问题"]], forbidden=["支持7天无理由", "支持七天无理由"])


def _build_return_fee(s, rng, p):
    ins = s.get("insurance", "否")
    tail = ("查询到您下单时购买了运费险，无理由退货的运费会在退货签收后自动理赔到您的账户。" if ins == "是"
            else "查询到您的订单未购买运费险，无理由退货的运费需要您自行承担。")
    body = ("退货运费需要分情况：如果是商品质量问题、错发漏发，运费由商家承担，您可先垫付后在售后单中上传运费凭证申请补偿；"
            "如果是无理由退货，运费由买家承担。" + tail)
    return _res(_wrap(rng, body), [["质量问题"], ["商家承担"], ["买家承担"], ["运费险"]],
                forbidden=["全部由商家承担", "都是免费的"])


def _build_exchange_size(s, rng, p):
    if int(s["days"]) > p["no_reason_days"]:
        body = (f"查询到您已签收{s['days']}天，已超过{p['no_reason_days']}天无理由退换期限，暂时无法直接换码。"
                "如果商品有开线、脱胶等质量问题，可以在【申请售后】中上传照片，我们核实后为您处理。")
        return _res(_wrap(rng, body), [["超过"], [f"{p['no_reason_days']}天"], ["质量问题"]], forbidden=["可以换"])
    body = (f"可以的~服装鞋类在吊牌完好、未穿着洗涤的情况下，签收后{p['no_reason_days']}天内支持换码。"
            "请在【申请售后】中选择“换货”并填写需要的尺码，我们收到退回商品后会第一时间为您发出新尺码。")
    return _res(_wrap(rng, body), [["吊牌"], [f"{p['no_reason_days']}天"], ["换货"]])


def _build_return_quality(s, rng, p):
    if int(s["days"]) > p["quality_exchange_days"]:
        body = (f"非常抱歉给您带来不便！您的商品已签收{s['days']}天，已超过{p['quality_exchange_days']}天的质量问题退换期，"
                f"数码家电类商品可在{p['warranty_months']}个月保修期内申请免费维修，其他类目请在【申请售后】上传照片，"
                "我们会为您核实处理。")
        return _res(_wrap(rng, body), [["抱歉"], ["超过"], ["维修"], ["照片"]], forbidden=["可以直接退货"])
    body = (f"非常抱歉给您带来不好的体验！收到商品有质量问题的话，请在签收后{p['quality_exchange_days']}天内，"
            "在【申请售后】中选择“质量问题”，上传商品问题照片或视频，审核通过后可以为您办理退货退款或换新，"
            "退回运费由商家承担。")
    return _res(_wrap(rng, body), [["抱歉"], [f"{p['quality_exchange_days']}天"], ["照片", "视频"], ["商家承担"]])


def _build_refund_time(s, rng, p):
    body = (f"退款在商家确认收货并审核通过后，会在{p['refund_days']}个工作日内原路退回您的支付账户，"
            "银行卡可能会有一定的入账延迟。您可以在【我的订单】-【退款详情】中查看实时进度。")
    return _res(_wrap(rng, body), [[f"{p['refund_days']}个工作日"], ["原路退回"], ["退款详情", "进度"]])


# --------------------------------------------------------------------------
# 物流
# --------------------------------------------------------------------------
def _slot_status(rng):
    return {"status": _pick(rng, ORDER_STATUS), "courier": _pick(rng, COURIERS)}


def _build_ship_time(s, rng, p):
    body = (f"一般情况下商品会在付款后{p['ship_hours']}小时内发货，大促期间可能会有延迟，"
            "发货后您可以在【我的订单】中查看物流单号和实时进度。")
    return _res(_wrap(rng, body), [[f"{p['ship_hours']}小时"], ["物流", "单号"]])


def _build_track(s, rng, p):
    st, c = s["status"], s["courier"]
    if st == "未发货":
        body = f"查询到您的订单目前还未发货，仓库会在付款后{p['ship_hours']}小时内发出，发货后会短信通知您物流单号。"
        kp = [["未发货"], [f"{p['ship_hours']}小时"]]
    elif st == "物流异常":
        body = (f"查询到您的包裹在{c}运输途中出现异常，我已经为您登记催查，"
                "预计24小时内会有专员联系您反馈结果，如包裹丢失我们会为您补发或全额退款。")
        kp = [["异常"], ["催查", "登记"], ["补发", "退款"]]
    elif st == "已签收":
        body = ("系统显示您的包裹已签收，如果您本人没有收到，请先确认是否放在驿站、快递柜或由家人代收，"
                f"如仍未找到请联系{c}快递员核实，我们也可以帮您发起丢件核查。")
        kp = [["已签收"], ["驿站", "快递柜", "代收"], ["核查", "核实"]]
    else:
        body = f"您的包裹目前由{c}承运，状态为“{st}”，正在正常配送中，您可以在【我的订单】-【查看物流】中查看实时位置，请耐心等待。"
        kp = [[st], ["查看物流", "实时"]]
    return _res(_wrap(rng, body), kp)


def _build_change_address(s, rng, p):
    st = s["status"]
    if st == "未发货":
        body = "订单尚未发货的情况下可以修改地址，请在【我的订单】-【修改地址】中直接修改，或把新地址发给我，我帮您备注给仓库。"
        return _res(_wrap(rng, body), [["未发货"], ["修改地址"]], forbidden=["无法修改"])
    body = (f"您的订单已经发出（当前状态：{st}），系统无法直接修改地址。建议您联系承运快递{s['courier']}申请改派，"
            "或在派送时与快递员沟通，部分快递改派可能产生额外费用。")
    return _res(_wrap(rng, body), [["已经发出", "已发货", "无法直接修改"], ["改派"], [s["courier"]]],
                forbidden=["可以直接修改"])


def _build_courier(s, rng, p):
    body = ("目前仓库会根据您的收货地址自动匹配最快的快递，暂不支持指定快递公司。"
            "如果对快递有特殊需求，可以在下单时备注，我们会尽量协调，但不能保证一定满足。")
    return _res(_wrap(rng, body), [["不支持指定"], ["备注"]], forbidden=["可以指定"])


def _build_delay(s, rng, p):
    body = ("非常抱歉让您久等了！我这边已经帮您催促快递加急派送，大促期间物流压力较大可能有1-2天的延迟。"
            "如果超过预计时间3天仍未送达，可以申请延迟发货赔付。")
    return _res(_wrap(rng, body), [["抱歉"], ["催", "加急"], ["赔付", "补偿"]])


# --------------------------------------------------------------------------
# 商品参数
# --------------------------------------------------------------------------
def _slot_product(rng):
    name = _pick(rng, list(PRODUCTS.keys()))
    attrs = [k for k in PRODUCTS[name] if k != "类目"]
    return {"product": name, "attr": _pick(rng, attrs)}


def _build_spec(s, rng, p):
    name, attr = s["product"], s["attr"]
    val = PRODUCTS[name][attr]
    body = f"{name}的{attr}：{val}。"
    if PRODUCTS[name]["类目"] in ("数码", "家电"):
        body += f"整机享受{p['warranty_months']}个月官方保修。"
    key = val.split("，")[0].split("/")[0][:8]
    return _res(_wrap(rng, body), [[key]])


# --------------------------------------------------------------------------
# 投诉
# --------------------------------------------------------------------------
def _build_complain_courier(s, rng, p):
    body = ("非常抱歉给您带来了不愉快的体验！我已经记录您反馈的快递服务问题，并同步给承运快递公司进行处理，"
            "处理结果会在24小时内通过短信告知您，同时为您申请了一张5元无门槛优惠券作为补偿。")
    return _res(_wrap(rng, body), [["抱歉"], ["记录", "反馈"], ["24小时"], ["补偿", "优惠券"]])


def _build_complain_damage(s, rng, p):
    body = (f"非常抱歉！包裹破损给您添麻烦了。请在签收后{p['fresh_report_hours']}小时内拍摄外包装和商品照片，"
            "在【申请售后】中选择“商品破损”提交，审核通过后可以为您补发或者全额退款，无需寄回破损商品。")
    return _res(_wrap(rng, body), [["抱歉"], ["照片", "拍"], ["补发", "退款"]])


def _build_complain_escalate(s, rng, p):
    body = ("非常理解您的心情，也很抱歉没有让您满意。您反馈的情况涉及赔偿认定，需要专属售后专员为您核实处理，"
            "我现在为您转人工客服，专员会在10分钟内接入，请您稍候。")
    return _res(_wrap(rng, body), [["抱歉", "理解"], ["转人工", "专员"]], transfer=True)


# --------------------------------------------------------------------------
# 售后 / 发票
# --------------------------------------------------------------------------
def _build_warranty(s, rng, p):
    body = (f"数码和家电类商品享受{p['warranty_months']}个月全国联保，保修期内非人为损坏可以免费维修。"
            "您可以在【申请售后】中选择“维修”，按照指引寄修或预约上门，维修完成后会免费寄回给您。")
    return _res(_wrap(rng, body), [[f"{p['warranty_months']}个月"], ["免费维修"], ["寄修", "上门"]])


def _build_invoice(s, rng, p):
    body = (f"支持开具电子发票的~订单确认收货后，在【我的订单】-【申请开票】中填写抬头和税号，"
            f"电子发票会在{p['invoice_days']}个工作日内发送到您的邮箱。")
    return _res(_wrap(rng, body), [["电子发票"], ["抬头", "税号"], [f"{p['invoice_days']}个工作日"]])


# --------------------------------------------------------------------------
# 优惠活动
# --------------------------------------------------------------------------
def _build_coupon(s, rng, p):
    body = ("双11期间平台券和店铺券可以叠加使用，但同类型的优惠券（比如两张店铺满减券）不能叠加。"
            "下单时系统会自动为您选择最优的优惠组合，您也可以在结算页手动切换。")
    return _res(_wrap(rng, body), [["平台券"], ["店铺券"], ["叠加"], ["同类", "同类型"]],
                forbidden=["所有优惠券都可以叠加", "不能叠加任何"])


def _build_price_protect(s, rng, p):
    if int(s["days"]) > p["price_protect_days"]:
        body = (f"查询到您的商品已签收{s['days']}天，超过{p['price_protect_days']}天价保期，暂时无法申请补差价，"
                "后续大促前可以关注【价格保护】入口，及时申请哦。")
        return _res(_wrap(rng, body), [["超过"], [f"{p['price_protect_days']}天"]], forbidden=["可以补差价"])
    body = (f"支持价保哦~商品签收后{p['price_protect_days']}天内如果同款降价，可以在【我的订单】-【价格保护】中申请补差价，"
            "差价会退回到您的支付账户，注意秒杀、拼团等特殊活动价不参与价保。")
    return _res(_wrap(rng, body), [[f"{p['price_protect_days']}天"], ["价格保护", "价保"], ["秒杀", "特殊活动"]])


def _build_deposit(s, rng, p):
    body = ("预售定金在一般情况下不支持退还，如因商家原因无法发货则会全额退还定金。"
            "如果您未支付尾款，定金将不予退还，建议在尾款时间内及时付款。")
    return _res(_wrap(rng, body), [["定金"], ["不支持退", "不予退还"], ["商家原因"]], forbidden=["定金可以随时退"])


# --------------------------------------------------------------------------
# Intent registry
# --------------------------------------------------------------------------
INTENTS: List[Intent] = [
    Intent("return_opened", "退换货",
           ["这个能退吗？已经拆封了", "拆开包装了还能退货吗", "打开用了一下能退不", "东西拆封了可以退款吗", "已经开封了还可以退吗",
            "拆了包装不想要了能退吗", "包装拆了能不能退", "开箱了还支持退吗"],
           ["我把封口撕开了，这种情况退得了吗", "开封之后还给退么", "盒子已经打开过了还能申请退货吗"],
           _slot_category, _build_return_opened, weight=2.0),
    Intent("return_no_reason", "退换货",
           ["买了不想要了可以退吗", "支持七天无理由吗", "不喜欢能退吗", "可以无理由退货吗", "买错了想退",
            "我不想要了，怎么退", "能七天无理由退吗"],
           ["后悔下单了可以退掉吗", "没质量问题也能退吗", "单纯不想要了给退吗"],
           _slot_category, _build_return_no_reason, weight=1.5),
    Intent("return_fee", "退换货",
           ["退货运费谁出", "退货要我出运费吗", "退回去的邮费谁承担", "退货邮费怎么算", "寄回去运费谁付"],
           ["退回的快递费是我掏吗", "退的时候运费要自己垫吗"],
           lambda r: {"insurance": _pick(r, ["是", "否"])}, _build_return_fee),
    Intent("exchange_size", "退换货",
           ["尺码不合适能换吗", "衣服小了能换大一码吗", "鞋子大了可以换吗", "能换个尺码吗", "码数买错了怎么换"],
           ["穿着有点紧想换大号，可以吗", "尺寸不对能调换不"],
           lambda r: {"category": "服装", "days": _days(r)}, _build_exchange_size),
    Intent("return_quality", "退换货",
           ["收到的东西有质量问题怎么办", "东西坏的怎么处理", "收到就是坏的", "商品有瑕疵能退吗", "质量太差了要退货"],
           ["刚到手就发现有毛病，怎么弄", "货有问题，你们怎么处理"],
           lambda r: {"days": str(r.randint(1, 15) if r.random() < 0.75 else r.randint(16, 40))},
           _build_return_quality, weight=1.5),
    Intent("refund_time", "退换货",
           ["退款多久到账", "钱什么时候退回来", "退款怎么还没到", "退款几天到", "多久能收到退款"],
           ["申请退款了啥时候能到卡里", "退的钱要等多久"],
           lambda r: {}, _build_refund_time),
    Intent("ship_time", "物流",
           ["什么时候发货", "多久能发货", "今天能发吗", "下单了啥时候发", "几天发货"],
           ["付完款大概多久能发出", "我的货什么时候寄出"],
           lambda r: {}, _build_ship_time),
    Intent("track", "物流",
           ["快递到哪了", "帮我查下物流", "我的包裹到哪里了", "物流信息查一下", "货到哪了"],
           ["我的件现在什么位置了", "能帮我看看包裹进度吗"],
           _slot_status, _build_track, weight=2.0),
    Intent("change_address", "物流",
           ["能改地址吗", "地址填错了怎么办", "我想修改收货地址", "可以换个地址吗", "收货地址写错了"],
           ["送货地址能帮我改一下吗", "我搬家了，地址能变更吗"],
           _slot_status, _build_change_address, weight=1.5),
    Intent("designate_courier", "物流",
           ["能指定顺丰吗", "可以发京东快递吗", "能不能换个快递", "指定快递可以吗"],
           ["我只要顺丰，可以安排吗", "快递能自己选吗"],
           lambda r: {}, _build_courier),
    Intent("delivery_delay", "物流",
           ["怎么还没到", "快递太慢了", "好几天了还没收到", "物流一直不动", "为什么这么久还没送到"],
           ["等了快一周了货还没来", "物流卡住不更新了"],
           lambda r: {}, _build_delay),
    Intent("spec", "商品参数",
           ["{product}的{attr}是多少", "{product}{attr}怎么样", "想问下{product}的{attr}", "{product}的{attr}介绍一下",
            "{product}，{attr}呢"],
           ["请介绍下{product}在{attr}方面的情况", "{attr}这块{product}表现如何"],
           _slot_product, _build_spec, weight=3.0),
    Intent("complain_courier", "投诉",
           ["快递员态度太差了", "快递员不送上门", "快递没打电话就放驿站了", "我要投诉快递员", "快递员骂人",
            "快递员让我自己下楼拿", "快递被放门口丢了一次了"],
           ["送货的人很不礼貌，我要反馈", "快递小哥直接扔门口了"],
           lambda r: {}, _build_complain_courier, weight=1.2),
    Intent("complain_damage", "投诉",
           ["收到的包裹压坏了", "外包装破了东西也坏了", "快递盒子都烂了", "东西摔碎了", "收到的时候已经裂开了",
            "盒子被压变形了里面也坏了"],
           ["到手箱子瘪了里面也碎了", "包裹被压扁了怎么办"],
           lambda r: {}, _build_complain_damage, weight=1.2),
    Intent("complain_escalate", "投诉",
           ["我要投诉到12315", "你们必须三倍赔偿", "我要找你们领导", "不给说法我就去消协", "我要求假一赔十",
            "你们这是欺诈，我要起诉", "我要曝光你们店"],
           ["再不处理我就报警了", "叫你们经理出来"],
           lambda r: {}, _build_complain_escalate),
    Intent("warranty", "售后",
           ["保修多久", "坏了怎么修", "可以保修吗", "售后怎么维修", "保修期是多长", "用了两个月坏了能修吗",
            "开不了机了怎么办", "有质保吗"],
           ["坏了能免费修吗", "出问题了去哪修"],
           lambda r: {}, _build_warranty, weight=1.2),
    Intent("invoice", "售后",
           ["能开发票吗", "怎么开发票", "可以开专票吗", "发票怎么申请", "发票在哪里开", "公司报销要发票",
            "能开电子发票吗"],
           ["我需要报销，发票在哪弄", "能给我开个票吗"],
           lambda r: {}, _build_invoice, weight=1.2),
    Intent("coupon_stack", "优惠活动",
           ["双11能叠加优惠券吗", "优惠券可以一起用吗", "店铺券和平台券能叠加吗", "满减能和券一起用吗", "两张券能同时用吗"],
           ["大促的券能不能叠着用", "几张优惠券能合并使用吗"],
           lambda r: {}, _build_coupon, weight=1.5),
    Intent("price_protect", "优惠活动",
           ["买完降价了能补差价吗", "支持价保吗", "刚买就降价了怎么办", "价格保护怎么申请"],
           ["前脚买后脚降价，能退差价不", "降价了可以补偿吗"],
           lambda r: {"days": str(r.randint(1, 15) if r.random() < 0.7 else r.randint(16, 40))},
           _build_price_protect),
    Intent("deposit_refund", "优惠活动",
           ["定金能退吗", "预售定金可以退吗", "不想付尾款了定金退吗"],
           ["交了订金反悔了能拿回来吗", "预付款能取消吗"],
           lambda r: {}, _build_deposit),
]

INTENT_BY_NAME = {i.name: i for i in INTENTS}


def render_query(intent: Intent, slots: Dict[str, str], rng: random.Random, split: str = "train") -> str:
    templates = intent.train_templates if split == "train" else intent.test_templates
    t = _pick(rng, templates)
    q = t.format(**slots) if "{" in t else t
    return f"{_pick(rng, USER_PREFIX)}{q}{_pick(rng, USER_SUFFIX)}"


# Which item categories make sense as "purchased item" context for each intent
_ITEM_CATS = {
    "warranty": ["数码", "家电"],
    "spec": [],  # already has product
    "exchange_size": ["服装"],
    "deposit_refund": ["数码", "家电", "美妆", "服装"],
}


def attach_item(intent: Intent, slots: Dict[str, str], rng: random.Random, p: float = 0.85) -> Dict[str, str]:
    """Attach the purchased item to the background info (真实客服工作台都会带订单商品信息)."""
    if "product" in slots or rng.random() > p:
        return slots
    slots = dict(slots)
    if "category" in slots:
        slots["item"] = _pick(rng, CATEGORY_ITEMS[slots["category"]])
        return slots
    cats = _ITEM_CATS.get(intent.name)
    if cats == []:
        return slots
    pool = [x for x in ALL_ITEMS if not cats or x[0] in cats]
    cat, item = _pick(rng, pool)
    slots["item"], slots["item_cat"] = item, cat
    return slots


def render_input(intent: Intent, slots: Dict[str, str]) -> str:
    parts = []
    if "item" in slots:
        parts.append(f"商品：{slots['item']}（{slots.get('category', slots.get('item_cat', ''))}）")
    elif "category" in slots:
        parts.append(f"商品类目：{slots['category']}")
    if "product" in slots:
        parts.append(f"商品：{slots['product']}")
    if "days" in slots:
        parts.append(f"已签收天数：{slots['days']}天")
    if "insurance" in slots:
        parts.append(f"运费险：{slots['insurance']}")
    if "status" in slots:
        parts.append(f"订单状态：{slots['status']}；承运快递：{slots['courier']}")
    return "；".join(parts)


# Colloquial synonym substitution (used for query diversity; NOT applied to test templates' structure)
SYNONYMS = {
    "东西": ["宝贝", "货", "商品"], "快递": ["包裹", "快件"], "能": ["可以", "可不可以"], "怎么": ["咋", "如何"],
    "退": ["退掉", "退回"], "多久": ["多长时间", "几天"], "吗": ["不", "么"], "收到": ["到手", "签收"],
}


def augment_query(q: str, rng: random.Random, p: float = 0.35) -> str:
    for k, vs in SYNONYMS.items():
        if k in q and rng.random() < p:
            q = q.replace(k, _pick(rng, vs), 1)
    return q
