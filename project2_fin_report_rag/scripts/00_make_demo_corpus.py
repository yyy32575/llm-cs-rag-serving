"""Generate a reproducible *demo* corpus of financial research-report PDFs + a QA evaluation set.

Why: 真实研报有版权，不能随源码分发。这里用固定随机种子生成 36 份“虚构公司”的研报/政策 PDF，
刻意还原真实研报的解析难点，保证整条 RAG 链路和全部实验可以在任何机器上复现：
  * 双栏排版（pdfplumber 直接 extract_text 会把左右栏同一行拼在一起）
  * 跨栏/整页表格（盈利预测表、季度经营数据表、行业市占率表）
  * 页眉页脚 + 免责声明（每页重复，污染检索）
  * 同一公司 2024Q3 / 2025Q3 两份点评（年份混淆的难负例）
  * 专业术语同义（文档写“装车量”，用户问“装机量”）

真实数据：把研报 PDF 放到 data/pdfs_real/，参考 data/eval/qa_template.csv 自建 120 题评测集即可（流程完全一致）。

  python scripts/00_make_demo_corpus.py --out_dir data
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finrag.utils import find_cjk_font, write_json, write_jsonl  # noqa: E402

DISCLAIMER = "本报告为RAG技术演示用虚构数据，公司与数据均不对应任何真实主体，不构成投资建议。"
BROKERS = ["星辰证券研究所", "东岳证券研究所", "南山证券研究所"]

# name, sector, kpi_name(doc term), kpi_unit, kpi_synonym(user term), revenue_2023(亿元), extra_metric
COMPANIES = [
    ("宁川动力", "动力电池", "动力电池装车量", "GWh", "动力电池装机量", 3200, "海外收入占比"),
    ("远驰汽车", "整车", "新能源汽车销量", "万辆", "电动车销量", 4100, "海外销量占比"),
    ("隆曦光能", "光伏", "组件出货量", "GW", "组件发货量", 1150, "海外收入占比"),
    ("华芯微电子", "晶圆代工", "产能利用率", "%", "稼动率", 520, "12英寸晶圆收入占比"),
    ("拓锐装备", "半导体设备", "新签订单", "亿元", "新接订单", 180, "刻蚀设备收入占比"),
    ("黔台酒业", "白酒", "直销收入占比", "%", "直营渠道占比", 1350, "系列酒收入占比"),
    ("蒙泉乳业", "乳制品", "液态奶收入", "亿元", "液奶收入", 1100, "奶粉收入占比"),
    ("康衡医药", "创新药", "创新药收入占比", "%", "新药收入占比", 230, "研发投入占比"),
    ("江海银行", "银行", "净息差", "%", "NIM", 3300, "不良贷款率"),
]
PEERS = {  # extra companies (full reports as hard negatives + appear in sector tables)
    "新能源": ["中旭锂电", "国创新能", "瑞丰储能"],
    "半导体": ["申微半导体", "北辰集成"],
    "消费": ["川南酒业", "华北乳品"],
    "医药": ["瑞安生物", "海诺制药"],
    "银行": ["东海银行", "滨江农商行"],
}
PEER_TEMPLATE = {"中旭锂电": "宁川动力", "国创新能": "隆曦光能", "瑞丰储能": "宁川动力", "申微半导体": "华芯微电子",
                 "北辰集成": "拓锐装备", "川南酒业": "黔台酒业", "华北乳品": "蒙泉乳业", "瑞安生物": "康衡医药",
                 "海诺制药": "康衡医药", "东海银行": "江海银行", "滨江农商行": "江海银行"}
YEARS_Q = (2023, 2024, 2025)
SECTOR_OF = {"动力电池": "新能源", "整车": "新能源", "光伏": "新能源", "晶圆代工": "半导体", "半导体设备": "半导体",
             "白酒": "消费", "乳制品": "消费", "创新药": "医药", "银行": "银行"}

FILLER = {
    "新能源": ["行业层面，海外市场需求恢复叠加国内以旧换新政策持续发力，产业链排产环比改善。",
              "原材料价格企稳后，产业链利润分配趋于均衡，龙头企业凭借规模与技术优势维持较高盈利水平。",
              "储能需求高增长成为第二增长曲线，大储与户储订单在中东、欧洲等地区持续落地。",
              "快充、半固态等新技术加速产业化，头部企业研发投入维持高位，技术代差有望进一步拉大。"],
    "半导体": ["国产替代持续推进，下游客户验证导入节奏加快，设备与材料环节订单能见度提升。",
              "消费电子补库存周期接近尾声，汽车电子与工业控制需求相对稳健，AI 相关需求高速增长。",
              "成熟制程扩产仍在持续，行业资本开支维持高位，先进封装成为新的增长点。"],
    "消费": ["高端消费需求承压，但品牌集中度持续提升，龙头公司渠道库存保持健康水平。",
            "公司积极推进渠道数字化改革，直营与线上渠道占比提升带动吨价与毛利率上行。",
            "节庆旺季动销表现平稳，批价整体稳定，公司全年经营目标完成度较高。"],
    "医药": ["集采影响逐步出清，创新药放量成为公司业绩增长的核心驱动力。",
            "公司在研管线丰富，多款产品进入III期临床，海外授权（License-out）交易打开估值空间。",
            "医保谈判规则趋于温和，创新药准入速度加快，商业化能力成为竞争关键。"],
    "银行": ["存款挂牌利率下调缓解负债成本压力，净息差降幅有望收窄。",
            "零售贷款需求逐步修复，对公贷款投向制造业与基础设施领域保持较快增长。",
            "资产质量整体稳健，拨备覆盖率维持较高水平，风险抵补能力充足。"],
}
RISKS = ["宏观经济下行导致需求不及预期的风险；", "行业竞争加剧导致价格战的风险；", "原材料价格大幅波动的风险；",
         "政策变化不及预期的风险；", "海外贸易摩擦加剧的风险。"]

# 带随机数字的行业分析句式：让长文档里充满“看起来相关”的干扰数字（真实研报的常态）
ANALYSIS = {
    "新能源": ["据测算，{y}年全球储能电池需求有望达到{a}GWh，同比增长{p}%，其中大储占比约{q}%。",
              "{y}年国内新能源汽车渗透率预计提升至{q}%，带动上游材料需求增长约{p}%。",
              "截至{y}年末，行业有效产能约{a}GWh，产能利用率约{q}%，二三线企业开工率明显偏低。",
              "海外市场方面，{y}年欧洲新能源车销量同比增长{p}%，北美市场增速约{q}%。",
              "原材料端，碳酸锂价格自高点回落约{q}%，单位成本下降带动行业平均毛利率修复约{s}个百分点。"],
    "半导体": ["据测算，{y}年全球半导体销售额约{a}亿美元，同比增长{p}%，其中AI相关芯片占比约{q}%。",
              "{y}年国内晶圆厂资本开支约{a}亿元，设备国产化率提升至约{q}%。",
              "成熟制程晶圆价格较高点下降约{q}%，但先进封装产能仍供不应求，交期延长至{s}个月以上。",
              "{y}年全球前十大晶圆代工厂合计市场份额约{q}%，行业集中度进一步提升。"],
    "消费": ["{y}年社会消费品零售总额同比增长{p}%，其中线上渠道占比提升至约{q}%。",
            "行业库存周转天数约{a}天，较上年同期下降约{s}天，渠道去库存基本完成。",
            "{y}年行业CR5约{q}%，高端产品价格带保持稳定，大众价格带竞争加剧。",
            "消费者调研显示，约{q}%的受访者表示未来一年将维持或增加相关消费支出。"],
    "医药": ["{y}年国内创新药市场规模约{a}亿元，同比增长{p}%，占医药市场比重提升至约{q}%。",
            "{y}年医保谈判新增药品平均降价约{q}%，谈判成功率约{p}%。",
            "行业研发投入强度约{s}%，头部企业在研管线数量平均超过{a}个。",
            "海外授权交易首付款合计约{a}亿美元，同比增长约{p}%。"],
    "银行": ["{y}年上市银行平均净息差约{s}%，较上年下降约{q}个基点。",
            "{y}年末行业不良贷款率约{s}%，拨备覆盖率约{a}%，资产质量整体稳健。",
            "{y}年社会融资规模存量同比增长{p}%，其中对公贷款增速约{q}%。",
            "存款利率下调后，行业负债成本率预计下降约{q}个基点。"],
}


def analysis_para(sector, rng, n=3):
    out = []
    for _ in range(n):
        t = rng.choice(ANALYSIS[sector])
        out.append(t.format(y=rng.choice([2023, 2024, 2025, 2026]), a=r1(rng.uniform(50, 3000)),
                            p=r1(rng.uniform(-10, 60)), q=r1(rng.uniform(5, 85)), s=r1(rng.uniform(0.5, 9))))
    return "".join(out) + rng.choice(FILLER[sector])

POLICIES = [
    {"id": "policy_nev_tradein_2024", "title": "关于开展新能源汽车以旧换新工作的通知（2024年版，示例，已废止）",
     "issuer": "示例部委",
     "articles": [
         ("第一条", "对个人消费者报废符合条件的燃油乘用车并购买新能源乘用车的，给予一次性补贴1万元。"),
         ("第二条", "对个人消费者转让旧车并购买新能源乘用车的，给予一次性置换补贴0.7万元。"),
         ("第三条", "补贴资金由中央财政和地方财政按照5:5的比例共同承担。"),
         ("第四条", "本通知自发布之日起施行，有效期至2024年12月31日。"),
     ]},
    {"id": "policy_nev_tradein", "title": "关于加力支持新能源汽车以旧换新的通知（示例）", "issuer": "示例部委",
     "articles": [
         ("第一条", "对个人消费者报废符合条件的燃油乘用车并购买新能源乘用车的，给予一次性补贴2万元。"),
         ("第二条", "对个人消费者转让旧车并购买新能源乘用车的，给予一次性置换补贴1.5万元。"),
         ("第三条", "本通知所称新能源乘用车，是指纳入《减免车辆购置税的新能源汽车车型目录》的纯电动、插电式混合动力乘用车。"),
         ("第四条", "补贴资金由中央财政和地方财政按照6:4的比例共同承担。"),
         ("第五条", "本通知自发布之日起施行，有效期至2025年12月31日。"),
     ]},
    {"id": "policy_ic", "title": "促进集成电路产业高质量发展的若干政策（示例）", "issuer": "示例部委",
     "articles": [
         ("第一条", "线宽小于28纳米且经营期在15年以上的集成电路生产企业，第一年至第十年免征企业所得税。"),
         ("第二条", "集成电路设计企业研发费用按照实际发生额的120%在税前加计扣除。"),
         ("第三条", "对集成电路关键设备和材料首台套、首批次应用给予最高3000万元的保险补偿。"),
         ("第四条", "支持符合条件的集成电路企业在科创板上市融资，优先审核。"),
     ]},
    {"id": "policy_consume", "title": "提振消费专项行动方案（示例）", "issuer": "示例部委",
     "articles": [
         ("第一条", "到2027年，社会消费品零售总额年均增长目标为5%以上。"),
         ("第二条", "加大消费品以旧换新支持力度，2025年安排超长期特别国债资金3000亿元。"),
         ("第三条", "实施育儿补贴制度，对3周岁以下婴幼儿每孩每年发放补贴3600元。"),
         ("第四条", "推动带薪错峰休假，鼓励有条件的地区探索实施春秋假。"),
     ]},
    {"id": "policy_bank_capital", "title": "商业银行资本管理办法（示例）", "issuer": "示例监管机构",
     "articles": [
         ("第一条", "商业银行核心一级资本充足率不得低于7.5%，一级资本充足率不得低于8.5%。"),
         ("第二条", "商业银行资本充足率不得低于10.5%。"),
         ("第三条", "系统重要性银行应额外计提附加资本，附加资本要求为风险加权资产的0.25%至1.5%。"),
         ("第四条", "本办法自2024年1月1日起施行。"),
     ]},
]


# ----------------------------------------------------------------------------- facts
def r1(x):
    return round(x, 1)


def r2(x):
    return round(x, 2)


def build_facts(rng):
    facts = {}
    tmpl = {c[0]: c for c in COMPANIES}
    all_cos = list(COMPANIES) + [(p,) + tmpl[t][1:5] + (int(tmpl[t][5] * rng.uniform(0.2, 0.6)),) + tmpl[t][6:]
                                 for p, t in PEER_TEMPLATE.items()]
    for name, ind, kpi, unit, syn, rev23, extra in all_cos:
        g = [rng.uniform(0.05, 0.25) for _ in range(4)]
        rev = [rev23]
        for gi in g:
            rev.append(rev[-1] * (1 + gi))
        nm = rng.uniform(0.08, 0.30) if ind not in ("银行",) else rng.uniform(0.35, 0.42)
        npf = [x * nm * rng.uniform(0.9, 1.1) for x in rev]
        shares = rng.choice([24.4, 43.9, 12.6, 79.5, 18.2, 64.0, 31.8, 55.3, 120.0])
        price = npf[2] / shares * rng.uniform(15, 35)
        eps = [x / shares for x in npf]
        pe = [price / e for e in eps]
        gm = rng.uniform(0.18, 0.45) if ind != "白酒" else rng.uniform(0.88, 0.92)
        years = ["2023A", "2024A", "2025E", "2026E", "2027E"]
        quarters = {}
        for y in YEARS_Q:
            for q in (1, 2, 3):
                base = rev[max(0, y - 2023)] / 4 * rng.uniform(0.85, 1.15) * (0.9 if y == 2023 else 1)
                if unit == "%":
                    kv = r1(rng.uniform(70, 95) if kpi == "产能利用率" else rng.uniform(25, 65)) \
                        if ind not in ("银行",) else r2(rng.uniform(1.4, 1.9))
                else:
                    kv = r1(base * rng.uniform(0.02, 0.4))
                quarters[f"{y}Q{q}"] = {"revenue": r2(base), "net_profit": r2(base * nm * rng.uniform(0.85, 1.15)),
                                        "gross_margin": r1(100 * gm * rng.uniform(0.92, 1.08)), "kpi": kv}
        for y in YEARS_Q:
            q3 = quarters[f"{y}Q3"]
            q3["revenue_yoy"] = r1(rng.uniform(-8, 35))
            q3["np_yoy"] = r1(rng.uniform(-15, 45))
            q3["ytd_revenue"] = r2(sum(quarters[f"{y}Q{q}"]["revenue"] for q in (1, 2, 3)))
            q3["share"] = r1(rng.uniform(5, 40))
        h1 = {"revenue": r2(quarters["2025Q1"]["revenue"] + quarters["2025Q2"]["revenue"]),
              "gross_margin": r1(100 * gm * rng.uniform(0.92, 1.08)), "np_yoy": r1(rng.uniform(-15, 45))}
        facts[name] = {
            "industry": ind, "sector": SECTOR_OF[ind], "kpi": kpi, "kpi_unit": unit, "kpi_syn": syn, "extra": extra,
            "extra_value": r1(rng.uniform(8, 45)) if extra != "不良贷款率" else r2(rng.uniform(0.9, 1.5)),
            "years": years, "revenue": [r2(x) for x in rev], "net_profit": [r2(x) for x in npf],
            "eps": [r2(x) for x in eps], "pe": [r1(x) for x in pe], "gross_margin_year": r1(gm * 100),
            "target_price": r2(price * rng.uniform(1.1, 1.35)), "rating": rng.choice(["买入", "增持", "买入"]),
            "quarters": quarters, "broker": rng.choice(BROKERS), "h1_2025": h1, "is_peer": name in PEER_TEMPLATE,
        }
    sector_tables = {}
    for sec, peers in PEERS.items():
        members = [n for n, f in facts.items() if f["sector"] == sec]
        shares = sorted([r1(rng.uniform(4, 30)) for _ in members], reverse=True)
        rng.shuffle(members)
        rows = [{"company": m, "share": s, "growth": r1(rng.uniform(-10, 40))} for m, s in zip(members, shares)]
        sector_tables[sec] = rows
    return facts, sector_tables


# ----------------------------------------------------------------------------- PDF rendering
class Renderer:
    def __init__(self, font_path):
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        pdfmetrics.registerFont(TTFont("CJK", font_path))
        self.H = ParagraphStyle("H", fontName="CJK", fontSize=15, leading=22, spaceAfter=8, wordWrap="CJK")
        self.H2 = ParagraphStyle("H2", fontName="CJK", fontSize=11.5, leading=17, spaceBefore=6, spaceAfter=4,
                                 wordWrap="CJK")
        self.P = ParagraphStyle("P", fontName="CJK", fontSize=9.5, leading=15, wordWrap="CJK", firstLineIndent=19)
        self.S = ParagraphStyle("S", fontName="CJK", fontSize=8.5, leading=12, wordWrap="CJK")
        self.T = ParagraphStyle("T", fontName="CJK", fontSize=8.5, leading=11, wordWrap="CJK")

    def doc(self, path, header, twocol_pages=True):
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate

        W, H = A4
        m, gap = 50, 18

        def deco(c, d):
            c.saveState()
            c.setFont("CJK", 8)
            c.drawString(m, H - 32, header)
            c.line(m, H - 36, W - m, H - 36)
            c.drawString(m, 30, "请务必阅读正文之后的免责声明部分  " + DISCLAIMER[:22])
            c.drawRightString(W - m, 30, f"第 {d.page} 页")
            c.restoreState()

        one = Frame(m, 50, W - 2 * m, H - 100, id="one")
        cw = (W - 2 * m - gap) / 2
        left = Frame(m, 50, cw, H - 100, id="l")
        right = Frame(m + cw + gap, 50, cw, H - 100, id="r")
        d = BaseDocTemplate(path, pagesize=A4, title=os.path.basename(path), author="demo")
        d.addPageTemplates([PageTemplate("one", [one], onPage=deco),
                            PageTemplate("two", [left, right], onPage=deco)])
        return d, W - 2 * m

    def table(self, title, header, rows, width):
        from reportlab.lib import colors
        from reportlab.platypus import Paragraph, Table, TableStyle

        data = [[Paragraph(str(x), self.T) for x in header]] + [[Paragraph(str(x), self.T) for x in r] for r in rows]
        t = Table(data, colWidths=[width / len(header)] * len(header), repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        return [Paragraph(title, self.S), t, Paragraph("资料来源：公司公告，示例研究所测算", self.S)]


def para_block(R, texts):
    from reportlab.platypus import Paragraph

    return [Paragraph(t, R.P) for t in texts]


def gen_deep_report(R, name, f, rng, path):
    from reportlab.platypus import FrameBreak, NextPageTemplate, PageBreak, Paragraph, Spacer

    ind, kpi, unit = f["industry"], f["kpi"], f["kpi_unit"]
    q = f["quarters"]["2025Q3"]
    d, width = R.doc(path, f"{f['broker']} | 公司深度报告 | {name}")
    fill = FILLER[f["sector"]]
    s = [Paragraph(f"{name}：{ind}龙头地位稳固，盈利能力持续提升（首次覆盖，{f['rating']}）", R.H),
         Paragraph(f"发布日期：2025年10月28日    评级：{f['rating']}    目标价：{f['target_price']}元", R.S),
         Spacer(1, 6), Paragraph("投资要点", R.H2)]
    s += para_block(R, [
        f"公司是国内{ind}行业的领先企业。2024年公司实现营业收入{f['revenue'][1]}亿元，归母净利润{f['net_profit'][1]}亿元，"
        f"全年综合毛利率为{f['gross_margin_year']}%。",
        f"我们预计公司2025-2027年归母净利润分别为{f['net_profit'][2]}亿元、{f['net_profit'][3]}亿元、{f['net_profit'][4]}亿元，"
        f"对应EPS分别为{f['eps'][2]}元、{f['eps'][3]}元、{f['eps'][4]}元。给予目标价{f['target_price']}元，评级“{f['rating']}”。",
        rng.choice(fill)])
    s += [Spacer(1, 8)] + R.table("表1：盈利预测与估值",
                                  ["指标"] + f["years"],
                                  [["营业收入(亿元)"] + f["revenue"], ["归母净利润(亿元)"] + f["net_profit"],
                                   ["EPS(元)"] + f["eps"], ["PE(倍)"] + f["pe"]], width)
    sec = f["sector"]
    s += [NextPageTemplate("two"), PageBreak(), Paragraph("一、公司概况与竞争格局", R.H2)]
    s += para_block(R, [analysis_para(sec, rng), analysis_para(sec, rng),
                        f"公司持续推进全球化布局，2025年上半年{f['extra']}达到{f['extra_value']}%，较上年同期明显提升。"
                        + (f"截至2025年三季度末，公司{f['extra']}为{f['extra_value']}%。" if f["extra"] == "不良贷款率" else ""),
                        analysis_para(sec, rng)])
    s += [FrameBreak(), Paragraph("二、核心经营指标跟踪", R.H2)]
    kpi_txt = (f"2025年第三季度，公司{kpi}为{q['kpi']}{unit}" if unit != "%" else
               f"2025年第三季度，公司{kpi}为{q['kpi']}%")
    s += para_block(R, [analysis_para(sec, rng, 2), kpi_txt + f"，国内市场份额约{q['share']}%，继续保持行业前列。",
                        analysis_para(sec, rng),
                        "从费用端看，公司持续优化销售与管理费用结构，期间费用率稳中有降，规模效应逐步显现。",
                        analysis_para(sec, rng)])
    s += [PageBreak(), Paragraph("三、行业空间测算", R.H2)]
    s += para_block(R, [analysis_para(sec, rng, 4) for _ in range(4)])
    s += [FrameBreak(), Paragraph("四、盈利预测与估值", R.H2)]
    s += para_block(R, [analysis_para(sec, rng),
                        f"基于上述分析，我们预计公司2025-2027年营业收入分别为{f['revenue'][2]}亿元、{f['revenue'][3]}亿元、"
                        f"{f['revenue'][4]}亿元。", analysis_para(sec, rng), analysis_para(sec, rng)])
    s += [NextPageTemplate("one"), PageBreak()]
    qs = [f"2025Q{i}" for i in (1, 2, 3)]
    s += R.table("表2：2025年前三季度单季经营数据", ["指标"] + qs,
                 [["营业收入(亿元)"] + [f["quarters"][k]["revenue"] for k in qs],
                  ["归母净利润(亿元)"] + [f["quarters"][k]["net_profit"] for k in qs],
                  ["毛利率(%)"] + [f["quarters"][k]["gross_margin"] for k in qs],
                  [f"{kpi}({unit})"] + [f["quarters"][k]["kpi"] for k in qs]], width)
    s += [Spacer(1, 10), Paragraph("风险提示", R.H2)] + para_block(R, rng.sample(RISKS, 3))
    s += [Spacer(1, 10), Paragraph("免责声明：" + DISCLAIMER + "本报告仅供示例，市场有风险，投资需谨慎。" * 3, R.S)]
    d.build(s)


def gen_quarterly(R, name, f, year, rng, path):
    from reportlab.platypus import FrameBreak, NextPageTemplate, PageBreak, Paragraph, Spacer

    q = f["quarters"][f"{year}Q3"]
    prev = f["quarters"][f"{year}Q2"]
    kpi, unit = f["kpi"], f["kpi_unit"]
    d, width = R.doc(path, f"{f['broker']} | 季报点评 | {name} {year}年三季报")
    s = [Paragraph(f"{name}{year}年三季报点评：单季业绩{'超预期' if q['np_yoy'] > 10 else '符合预期'}", R.H),
         Paragraph(f"发布日期：{year}年10月{rng.randint(20, 31)}日    评级：{f['rating']}", R.S), Spacer(1, 6),
         NextPageTemplate("two"), PageBreak(), Paragraph("事件", R.H2)]
    sec = f["sector"]
    s += para_block(R, [f"公司发布{year}年三季度报告。{year}年前三季度公司累计实现营业收入{q['ytd_revenue']}亿元。"
                        f"其中第三季度单季实现营业收入{q['revenue']}亿元，同比{'增长' if q['revenue_yoy'] >= 0 else '下降'}"
                        f"{abs(q['revenue_yoy'])}%；实现归母净利润{q['net_profit']}亿元，同比"
                        f"{'增长' if q['np_yoy'] >= 0 else '下降'}{abs(q['np_yoy'])}%。",
                        analysis_para(sec, rng, 3), analysis_para(sec, rng, 2)])
    s += [FrameBreak(), Paragraph("点评", R.H2)]
    gm_delta = r1(q["gross_margin"] - prev["gross_margin"])
    s += para_block(R, [
        analysis_para(sec, rng, 2) + f"盈利能力方面，{year}年第三季度公司单季毛利率为{q['gross_margin']}%，环比"
        f"{'提升' if gm_delta >= 0 else '下降'}{abs(gm_delta)}个百分点，主要受产品结构优化及原材料成本变化影响。",
        analysis_para(sec, rng, 2),
        (f"经营数据方面，{year}年三季度公司{kpi}达到{q['kpi']}{unit}，" if unit != "%" else
         f"经营数据方面，{year}年三季度公司{kpi}为{q['kpi']}%，") + "环比继续改善。" + analysis_para(sec, rng, 2),
        analysis_para(sec, rng, 3), "我们维持此前盈利预测，维持评级。"])
    s += [NextPageTemplate("one"), PageBreak()]
    qs = [f"{year}Q{i}" for i in (1, 2, 3)]
    s += R.table(f"表1：{year}年分季度核心财务数据", ["指标"] + qs,
                 [["营业收入(亿元)"] + [f["quarters"][k]["revenue"] for k in qs],
                  ["归母净利润(亿元)"] + [f["quarters"][k]["net_profit"] for k in qs],
                  ["毛利率(%)"] + [f["quarters"][k]["gross_margin"] for k in qs]], width)
    s += [Spacer(1, 10), Paragraph("风险提示", R.H2)] + para_block(R, rng.sample(RISKS, 2))
    s += [Paragraph("免责声明：" + DISCLAIMER, R.S)]
    d.build(s)


def gen_sector(R, sec, rows, facts, rng, path):
    from reportlab.platypus import FrameBreak, NextPageTemplate, PageBreak, Paragraph, Spacer

    d, width = R.doc(path, f"{rng.choice(BROKERS)} | 行业深度报告 | {sec}行业")
    top3 = sorted(rows, key=lambda r: -r["share"])[:3]
    s = [Paragraph(f"{sec}行业2025年中期策略：格局优化，龙头集中度提升", R.H),
         Paragraph("发布日期：2025年7月15日    行业评级：推荐", R.S), Spacer(1, 6), Paragraph("核心观点", R.H2)]
    s += para_block(R, [f"2025年上半年，{sec}行业集中度继续提升，排名前三的企业分别为{top3[0]['company']}、{top3[1]['company']}"
                        f"和{top3[2]['company']}，三者合计市场份额达到{r1(sum(x['share'] for x in top3))}%。",
                        rng.choice(FILLER[sec])])
    s += [Spacer(1, 6)] + R.table(f"表1：2025年上半年{sec}行业主要企业市场份额", ["企业", "市场份额(%)", "同比增速(%)"],
                                  [[r["company"], r["share"], r["growth"]] for r in rows], width)
    s += [NextPageTemplate("two"), PageBreak(), Paragraph("一、行业景气度回顾", R.H2)]
    s += para_block(R, [analysis_para(sec, rng, 3) for _ in range(4)])
    s += [FrameBreak(), Paragraph("二、重点公司梳理", R.H2)]
    for n, f in facts.items():
        if f["sector"] == sec:
            s += para_block(R, [f"{n}：公司2024年实现营业收入{f['revenue'][1]}亿元，我们预计2025年归母净利润为{f['net_profit'][2]}亿元，"
                                f"对应PE为{f['pe'][2]}倍。" + rng.choice(FILLER[sec])])
    s += [Paragraph("风险提示", R.H2)] + para_block(R, rng.sample(RISKS, 3))
    d.build(s)


def gen_h1(R, name, f, rng, path):
    """2025 半年报点评：与三季报点评指标名高度相似，是 Q3 问题的强干扰文档。"""
    from reportlab.platypus import FrameBreak, NextPageTemplate, PageBreak, Paragraph, Spacer

    h = f["h1_2025"]
    q2 = f["quarters"]["2025Q2"]
    d, width = R.doc(path, f"{f['broker']} | 中报点评 | {name} 2025年半年报")
    s = [Paragraph(f"{name}2025年中报点评：上半年经营稳健", R.H),
         Paragraph(f"发布日期：2025年8月{rng.randint(20, 31)}日    评级：{f['rating']}", R.S), Spacer(1, 6),
         NextPageTemplate("two"), PageBreak(), Paragraph("事件", R.H2)]
    s += para_block(R, [f"公司发布2025年半年度报告，上半年实现营业收入{h['revenue']}亿元，归母净利润同比"
                        f"{'增长' if h['np_yoy'] >= 0 else '下降'}{abs(h['np_yoy'])}%。",
                        analysis_para(f["sector"], rng, 3)])
    s += [FrameBreak(), Paragraph("点评", R.H2)]
    s += para_block(R, [f"上半年公司综合毛利率为{h['gross_margin']}%，其中第二季度单季毛利率为{q2['gross_margin']}%，"
                        f"第二季度{f['kpi']}为{q2['kpi']}{'' if f['kpi_unit'] == '%' else f['kpi_unit']}"
                        f"{'%' if f['kpi_unit'] == '%' else ''}。",
                        rng.choice(FILLER[f["sector"]]), rng.choice(FILLER[f["sector"]])])
    s += [Paragraph("风险提示", R.H2)] + para_block(R, rng.sample(RISKS, 2))
    d.build(s)


def gen_policy(R, p, path):
    from reportlab.platypus import Paragraph, Spacer

    d, width = R.doc(path, f"{p['issuer']} | 政策文件")
    s = [Paragraph(p["title"], R.H), Paragraph(f"发文机关：{p['issuer']}    （示例文件，非真实政策）", R.S), Spacer(1, 8)]
    s += para_block(R, ["为贯彻落实相关决策部署，进一步促进产业高质量发展，现就有关事项通知如下。"])
    for a, t in p["articles"]:
        s += para_block(R, [f"{a} {t}"])
    s += para_block(R, ["各地区、各有关部门要结合实际，认真抓好贯彻落实。"])
    d.build(s)


# ----------------------------------------------------------------------------- QA
def build_qa(facts, sector_tables, rng):
    """金标证据允许多个文档（同一数字可能同时出现在季报点评和深度报告的季度表中）。"""
    qa = []

    def add(qtype, q, ans, evid, **kw):
        ev2 = [{"doc_ids": e["doc_id"] if isinstance(e["doc_id"], list) else [e["doc_id"]], "must": e["must"]}
               for e in evid]
        qa.append({"qid": f"q{len(qa):03d}", "type": qtype, "query": q, "answer": str(ans), "evidence": ev2, **kw})

    main = [c[0] for c in COMPANIES]
    for n in main:
        f = facts[n]
        sn = n if rng.random() < 0.6 else n[:2]  # 用户常用简称：宁川 / 黔台 / 江海
        q25, q24 = f["quarters"]["2025Q3"], f["quarters"]["2024Q3"]
        d25, d24, deep = f"{n}_2025Q3点评", f"{n}_2024Q3点评", f"{n}_深度报告"
        unit = f["kpi_unit"]
        kv = f"{q25['kpi']}{'' if unit == '%' else unit}" + ("%" if unit == "%" else "")
        add("事实", rng.choice([f"{sn}2025年第三季度营业收入是多少？", f"{sn}25年Q3单季营收多少亿元？",
                              f"{sn}今年三季度收入多少"]),
            f"{q25['revenue']}亿元", [{"doc_id": [d25, deep], "must": str(q25['revenue'])}])
        add("事实", rng.choice([f"{sn}2025年三季度单季毛利率是多少？", f"{sn}2025Q3毛利率多少"]),
            f"{q25['gross_margin']}%", [{"doc_id": [d25, deep], "must": str(q25['gross_margin'])}])
        add("事实-年份混淆", f"{sn}2024年第三季度归母净利润同比增速是多少？", f"{q24['np_yoy']}%",
            [{"doc_id": d24, "must": str(abs(q24['np_yoy']))}])
        add("事实", f"{sn}2025年前三季度累计营业收入是多少？", f"{q25['ytd_revenue']}亿元",
            [{"doc_id": d25, "must": str(q25['ytd_revenue'])}])
        add("同义改写", f"{sn}2025年三季度的{f['kpi_syn']}是多少？", kv,
            [{"doc_id": [d25, deep], "must": str(q25['kpi'])}])
        add("表格", rng.choice([f"根据盈利预测表，{sn}2026年EPS预测值是多少？", f"{sn}2026E的每股收益预测是多少元？"]),
            f"{f['eps'][3]}元", [{"doc_id": deep, "must": str(f['eps'][3])}])
        add("表格", f"{sn}2027年预测PE是多少倍？", f"{f['pe'][4]}倍", [{"doc_id": deep, "must": str(f['pe'][4])}])
        add("事实", f"{sn}2025年上半年{f['extra']}是多少？" if f["extra"] != "不良贷款率" else f"{sn}三季度末不良贷款率是多少？",
            f"{f['extra_value']}%", [{"doc_id": deep, "must": str(f['extra_value'])}])
    names = main
    pairs = [("宁川动力", "隆曦光能"), ("远驰汽车", "宁川动力"), ("华芯微电子", "拓锐装备"), ("黔台酒业", "蒙泉乳业"),
             ("康衡医药", "黔台酒业"), ("江海银行", "远驰汽车"), ("隆曦光能", "华芯微电子"), ("拓锐装备", "康衡医药")]
    for a, b in pairs:
        qa_, qb_ = facts[a]["quarters"]["2025Q3"], facts[b]["quarters"]["2025Q3"]
        win = a if qa_["gross_margin"] > qb_["gross_margin"] else b
        add("对比", f"2025年第三季度，{a}和{b}谁的单季毛利率更高？", f"{win}（{a}{qa_['gross_margin']}%，{b}{qb_['gross_margin']}%）",
            [{"doc_id": [f"{a}_2025Q3点评", f"{a}_深度报告"], "must": str(qa_['gross_margin'])},
             {"doc_id": [f"{b}_2025Q3点评", f"{b}_深度报告"], "must": str(qb_['gross_margin'])}])
    for a, b in [("宁川动力", "远驰汽车"), ("黔台酒业", "蒙泉乳业"), ("华芯微电子", "拓锐装备")]:
        ra, rb = facts[a]["net_profit"][2], facts[b]["net_profit"][2]
        add("对比", f"研报预测2025年{a}和{b}的归母净利润分别是多少？", f"{a}{ra}亿元，{b}{rb}亿元",
            [{"doc_id": f"{a}_深度报告", "must": str(ra)}, {"doc_id": f"{b}_深度报告", "must": str(rb)}])
    for sec, rows in sector_tables.items():
        top3 = sorted(rows, key=lambda r: -r["share"])[:3]
        add("汇总", f"{sec}行业2025年上半年市场份额前三的企业合计份额是多少？", f"{r1(sum(x['share'] for x in top3))}%",
            [{"doc_id": f"{sec}行业报告", "must": str(r1(sum(x['share'] for x in top3)))}])
        pick = rng.choice(rows)
        add("表格", f"{sec}行业报告中，{pick['company']}的市场份额是多少？", f"{pick['share']}%",
            [{"doc_id": f"{sec}行业报告", "must": str(pick['share'])}])
    pol_q = {
        "policy_nev_tradein": [("2025年加力支持以旧换新通知中，报废燃油车购买新能源乘用车可以获得多少补贴？", "2万元", "2万元"),
                               ("2025年新能源汽车以旧换新通知的有效期到什么时候？", "2025年12月31日", "2025年12月31日"),
                               ("2025年以旧换新补贴资金中央和地方怎么分担？", "按6:4比例共同承担", "6:4")],
        "policy_ic": [("集成电路设计企业研发费用加计扣除比例是多少？", "120%", "120%"),
                      ("28纳米以下集成电路生产企业可以免征几年企业所得税？", "第一年至第十年免征", "第一年至第十年")],
        "policy_consume": [("提振消费方案中育儿补贴标准是多少？", "每孩每年3600元", "3600元"),
                           ("2025年以旧换新安排了多少超长期特别国债资金？", "3000亿元", "3000亿元")],
        "policy_bank_capital": [("商业银行核心一级资本充足率最低要求是多少？", "7.5%", "7.5%"),
                                ("系统重要性银行附加资本要求范围是多少？", "0.25%至1.5%", "0.25%")],
    }
    for pid, qs in pol_q.items():
        for q, a, must in qs:
            add("政策", q, a, [{"doc_id": pid, "must": must}])
    unans = [f"{rng.choice(names)}2028年的营业收入预测是多少？", "星河电池2025年三季度毛利率是多少？",
             "宁川动力的董事长是谁？", "黔台酒业2025年第四季度营收是多少？", "江海银行2025年的分红率是多少？",
             "华芯微电子5纳米工艺良率是多少？", "远驰汽车2025年在欧洲的工厂产能是多少？", "康衡医药的市值是多少？",
             "提振消费方案中汽车贷款利率是多少？", "半导体行业2030年市场规模预测是多少？", "蒙泉乳业的员工人数是多少？",
             "隆曦光能2025年钙钛矿组件转换效率是多少？"]
    for q in unans:
        add("无答案", q, "无法回答", [])
    return qa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    facts, sector_tables = build_facts(rng)
    R = Renderer(find_cjk_font())
    pdf_dir = os.path.join(args.out_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    docs = []
    for n, f in facts.items():
        for doc_id, fn in [(f"{n}_深度报告", lambda p: gen_deep_report(R, n, f, rng, p)),
                           (f"{n}_2025Q3点评", lambda p: gen_quarterly(R, n, f, 2025, rng, p)),
                           (f"{n}_2024Q3点评", lambda p: gen_quarterly(R, n, f, 2024, rng, p)),
                           (f"{n}_2023Q3点评", lambda p: gen_quarterly(R, n, f, 2023, rng, p)),
                           (f"{n}_2025H1点评", lambda p: gen_h1(R, n, f, rng, p))]:
            p = os.path.join(pdf_dir, doc_id + ".pdf")
            fn(p)
            docs.append({"doc_id": doc_id, "file": os.path.basename(p), "company": n, "type": doc_id.split("_")[1]})
    for sec, rows in sector_tables.items():
        p = os.path.join(pdf_dir, f"{sec}行业报告.pdf")
        gen_sector(R, sec, rows, facts, rng, p)
        docs.append({"doc_id": f"{sec}行业报告", "file": os.path.basename(p), "company": "", "type": "行业报告"})
    for pol in POLICIES:
        p = os.path.join(pdf_dir, pol["id"] + ".pdf")
        gen_policy(R, pol, p)
        docs.append({"doc_id": pol["id"], "file": os.path.basename(p), "company": "", "type": "政策"})
    qa = build_qa(facts, sector_tables, rng)
    write_json(os.path.join(args.out_dir, "facts.json"), {"facts": facts, "sector_tables": sector_tables})
    write_jsonl(os.path.join(args.out_dir, "docs.jsonl"), docs)
    write_jsonl(os.path.join(args.out_dir, "eval", "qa.jsonl"), qa)
    from collections import Counter

    print(f"PDFs: {len(docs)} -> {pdf_dir}")
    print(f"QA: {len(qa)}", dict(Counter(x['type'] for x in qa)))


if __name__ == "__main__":
    main()
