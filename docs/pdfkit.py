"""PDF 排版框架（reportlab）：标题/段落/列表/表格/代码/提示框/QA，中文字体自动查找。"""
from __future__ import annotations

import os
import re
from typing import List

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table,
                                TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

BODY_FONTS = [os.environ.get("CJK_FONT", ""), "/Library/Fonts/Arial Unicode.ttf",
              "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "C:/Windows/Fonts/msyh.ttc",
              "C:/Windows/Fonts/simhei.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
HEAD_FONTS = [os.environ.get("CJK_FONT_BOLD", ""), "/System/Library/Fonts/STHeiti Medium.ttc",
              "C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/simhei.ttf"]


def _reg(name, cands):
    for c in cands:
        if c and os.path.exists(c):
            try:
                pdfmetrics.registerFont(TTFont(name, c, subfontIndex=0) if c.endswith(".ttc") else TTFont(name, c))
                return True
            except Exception:
                continue
    return False


assert _reg("Body", BODY_FONTS), "未找到中文字体，请设置 CJK_FONT"
if not _reg("Head", HEAD_FONTS):
    pdfmetrics.registerFont(TTFont("Head", [c for c in BODY_FONTS if c and os.path.exists(c)][0]))

ACCENT = colors.HexColor("#C0602B")
DARK = colors.HexColor("#1F2A44")
GREY = colors.HexColor("#666666")
LIGHT = colors.HexColor("#F6EFE7")
TBL_HEAD = colors.HexColor("#EADBC8")

ST = {
    "title": ParagraphStyle("title", fontName="Head", fontSize=26, leading=36, alignment=TA_CENTER, textColor=DARK),
    "subtitle": ParagraphStyle("subtitle", fontName="Body", fontSize=13, leading=20, alignment=TA_CENTER, textColor=GREY),
    "h0": ParagraphStyle("h0", fontName="Head", fontSize=20, leading=28, textColor=ACCENT, spaceBefore=4, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="Head", fontSize=15.5, leading=22, textColor=DARK, spaceBefore=12, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Head", fontSize=12.5, leading=18, textColor=ACCENT, spaceBefore=9, spaceAfter=4),
    "h3": ParagraphStyle("h3", fontName="Head", fontSize=11, leading=16, textColor=DARK, spaceBefore=6, spaceAfter=2),
    "p": ParagraphStyle("p", fontName="Body", fontSize=9.6, leading=15.5, wordWrap="CJK", spaceAfter=3),
    "li": ParagraphStyle("li", fontName="Body", fontSize=9.6, leading=15, wordWrap="CJK", leftIndent=14,
                         bulletIndent=4, spaceAfter=1.5),
    "small": ParagraphStyle("small", fontName="Body", fontSize=8.3, leading=12, wordWrap="CJK", textColor=GREY),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.1, leading=11, wordWrap="CJK"),
    "cellh": ParagraphStyle("cellh", fontName="Head", fontSize=8.3, leading=11, wordWrap="CJK"),
    "code": ParagraphStyle("code", fontName="Body", fontSize=7.9, leading=11, backColor=colors.HexColor("#F4F4F4"),
                           borderPadding=5, leftIndent=4, rightIndent=4, spaceBefore=3, spaceAfter=6),
    "box": ParagraphStyle("box", fontName="Body", fontSize=9.4, leading=15, wordWrap="CJK"),
    "q": ParagraphStyle("q", fontName="Head", fontSize=9.8, leading=15, wordWrap="CJK", textColor=DARK, spaceBefore=5),
    "a": ParagraphStyle("a", fontName="Body", fontSize=9.4, leading=14.8, wordWrap="CJK", leftIndent=12, spaceAfter=2),
    "toc1": ParagraphStyle("toc1", fontName="Head", fontSize=11, leading=17, leftIndent=6),
    "toc2": ParagraphStyle("toc2", fontName="Body", fontSize=9.5, leading=14, leftIndent=22),
    "toc3": ParagraphStyle("toc3", fontName="Body", fontSize=8.8, leading=13, leftIndent=38, textColor=GREY),
}


def esc(s: str) -> str:
    """Escape & keep **bold** / `code` light markup."""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r'<font name="Head" color="#1F2A44">\1</font>', s)
    s = re.sub(r"`(.+?)`", r'<font color="#B03A2E">\1</font>', s)
    return s


class Doc:
    def __init__(self):
        self.flow: List = []
        self._n = 0

    # ---------------- headings (registered in TOC)
    def _heading(self, text, style, level):
        self._n += 1
        key = f"h{self._n}"
        p = Paragraph(f'<a name="{key}"/>{esc(text)}', ST[style])
        p._toc = (level, text, key)
        self.flow.append(p)

    def part(self, text):
        self.flow.append(PageBreak())
        self._heading(text, "h0", 0)

    def h1(self, t):
        self._heading(t, "h1", 1)

    def h2(self, t):
        self._heading(t, "h2", 2)

    def h3(self, t):
        self.flow.append(Paragraph(esc(t), ST["h3"]))

    # ---------------- body
    def p(self, t):
        self.flow.append(Paragraph(esc(t), ST["p"]))

    def small(self, t):
        self.flow.append(Paragraph(esc(t), ST["small"]))

    def ul(self, items):
        for it in items:
            self.flow.append(Paragraph(esc(it), ST["li"], bulletText="•"))

    def ol(self, items):
        for i, it in enumerate(items, 1):
            self.flow.append(Paragraph(esc(it), ST["li"], bulletText=f"{i}."))

    def code(self, text):
        self.flow.append(Preformatted(text.strip("\n"), ST["code"]))

    def box(self, text, color="#F6EFE7", border="#C0602B", title=None):
        inner = []
        if title:
            inner.append(Paragraph(f'<font name="Head" color="{border}">{esc(title)}</font>', ST["box"]))
        for line in text.strip().split("\n"):
            if line.strip():
                inner.append(Paragraph(esc(line.strip()), ST["box"]))
        t = Table([[inner]], colWidths=[170 * mm])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(color)),
                               ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(border)),
                               ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                               ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        self.flow += [Spacer(1, 3), t, Spacer(1, 5)]

    def resume(self, title, sections):
        """简历块：sections = [(小标题, [行...]), ...]，排成与参考简历一致的格式。"""
        inner = [Paragraph(f'<font name="Head" size="10.5" color="#1F2A44">{esc(title)}</font>', ST["box"])]
        for head, lines in sections:
            if head:
                inner.append(Paragraph(f'<font name="Head">{esc(head)}</font>', ST["box"]))
            for ln in lines:
                inner.append(Paragraph(esc(ln), ST["box"]))
        t = Table([[inner]], colWidths=[170 * mm])
        t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#3B5BA5")),
                               ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                               ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 7),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
        self.flow += [Spacer(1, 3), t, Spacer(1, 6)]

    def table(self, header, rows, widths=None, font_size=None):
        hs = ST["cellh"]
        cs = ST["cell"] if not font_size else ParagraphStyle("c2", parent=ST["cell"], fontSize=font_size,
                                                              leading=font_size + 3)
        data = [[Paragraph(esc(str(h)), hs) for h in header]]
        data += [[Paragraph(esc("" if c is None else str(c)), cs) for c in r] for r in rows]
        total = 170 * mm
        if widths:
            s = sum(widths)
            cw = [total * w / s for w in widths]
        else:
            cw = [total / len(header)] * len(header)
        t = Table(data, colWidths=cw, repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
                               ("BACKGROUND", (0, 0), (-1, 0), TBL_HEAD),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FBF8F4")]),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        self.flow += [Spacer(1, 2), t, Spacer(1, 6)]

    def qa(self, items, start=1):
        for i, (q, a) in enumerate(items, start):
            self.flow.append(KeepTogether([Paragraph(f"Q{i}. {esc(q)}", ST["q"]), Paragraph(esc(a), ST["a"])]))
        return start + len(items)

    def space(self, h=6):
        self.flow.append(Spacer(1, h))

    def pagebreak(self):
        self.flow.append(PageBreak())


class _Template(SimpleDocTemplate):
    def afterFlowable(self, f):
        if hasattr(f, "_toc"):
            level, text, key = f._toc
            self.notify("TOCEntry", (level, text, self.page, key))
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)


def build(doc: Doc, path: str, cover: List, header_text: str):
    def on_page(c, d):
        c.saveState()
        c.setFont("Body", 7.8)
        c.setFillColor(GREY)
        c.drawString(20 * mm, 287 * mm, header_text)
        c.setStrokeColor(colors.HexColor("#DDDDDD"))
        c.line(20 * mm, 285.5 * mm, 190 * mm, 285.5 * mm)
        c.drawRightString(190 * mm, 10 * mm, f"{d.page}")
        c.restoreState()

    toc = TableOfContents()
    toc.levelStyles = [ST["toc1"], ST["toc2"], ST["toc3"]]
    toc.dotsMinLevel = 0
    story = list(cover) + [PageBreak(), Paragraph("目  录", ST["h0"]), toc] + doc.flow
    t = _Template(path, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm,
                  bottomMargin=16 * mm, title="大模型算法岗三项目：项目文档", author="LLM Projects")
    t.multiBuild(story, onFirstPage=lambda c, d: None, onLaterPages=on_page)
