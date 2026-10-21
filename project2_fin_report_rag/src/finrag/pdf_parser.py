"""PDF parsing for research reports.

Two parsers so the difference can be *measured* (see scripts/01_parse_pdfs.py):
  * NaiveParser  : pdfplumber page.extract_text() on the full page —— 等价于“直接把 PDF 当 txt split”。
                   双栏会把左右两栏同一行拼在一起，表格被打平成一串数字，页眉页脚原样保留。
  * LayoutParser : 1) pdfplumber 检测表格 -> 转 markdown（保留表头，表格标题“表N：...”绑定到表格）
                   2) 表格区域外的字符做“栏间空白带(gutter)检测”，双栏页按左右栏分别抽取
                   3) 去页眉页脚：几何区域 + 跨页高频重复行双重判定
                   4) 中文行合并：同段落内换行直接拼接，按句末标点/标题模式断段
Optional: if PyMuPDF is installed, `engine="pymupdf"` uses its block-level text as the text source.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

HEADING_RE = re.compile(r"^(第[一二三四五六七八九十\d]+[章节条]|[一二三四五六七八九十]+、|\d+(\.\d+)*\s|表\d+[：:]|图\d+[：:]|"
                        r"投资要点|核心观点|风险提示|事件|点评|免责声明)")
SENT_END = "。！？；：）」”%"


@dataclass
class Block:
    doc_id: str
    page: int
    type: str  # text | table
    text: str
    section: str = ""
    title: str = ""
    meta: Dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _table_to_markdown(rows: List[List[Optional[str]]]) -> str:
    rows = [[(c or "").replace("\n", "") for c in r] for r in rows if any(c for c in r)]
    if not rows:
        return ""
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * n]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _inside(obj, bbox, pad=1.0):
    x0, top, x1, bottom = bbox
    return (obj["x0"] >= x0 - pad and obj["x1"] <= x1 + pad and obj["top"] >= top - pad and obj["bottom"] <= bottom + pad)


class NaiveParser:
    name = "naive"

    def parse(self, path: str, doc_id: Optional[str] = None) -> List[Block]:
        import pdfplumber

        doc_id = doc_id or os.path.splitext(os.path.basename(path))[0]
        blocks = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                blocks.append(Block(doc_id, i, "text", page.extract_text() or ""))
        return blocks


class LayoutParser:
    name = "layout"

    def __init__(self, header_h: float = 42, footer_h: float = 42, min_gutter: float = 8.0,
                 repeat_ratio: float = 0.5, engine: str = "pdfplumber"):
        self.header_h, self.footer_h = header_h, footer_h
        self.min_gutter = min_gutter
        self.repeat_ratio = repeat_ratio
        self.engine = engine
        self.stats = Counter()

    # ------------------------------------------------------------------ columns
    def _find_gutter(self, chars, width) -> Optional[float]:
        """Return x of the vertical whitespace band separating two columns, or None for single column."""
        if len(chars) < 50:
            return None
        bins = [0] * int(width // 2 + 1)
        for c in chars:
            for b in range(int(c["x0"] // 2), min(len(bins), int(c["x1"] // 2) + 1)):
                bins[b] += 1
        lo, hi = int(width * 0.35 // 2), int(width * 0.65 // 2)
        best, run, best_mid = 0, 0, None
        for b in range(lo, hi):
            if bins[b] == 0:
                run += 1
                if run > best:
                    best, best_mid = run, (b - run / 2) * 2
            else:
                run = 0
        if best * 2 < self.min_gutter or best_mid is None:
            return None
        left = sum(1 for c in chars if c["x1"] <= best_mid)
        right = len(chars) - left
        return best_mid if min(left, right) > 0.15 * len(chars) else None

    @staticmethod
    def _lines(chars, tol=3.0) -> List[Tuple[float, float, str]]:
        """Group chars into lines -> (top, x1_max, text)."""
        chars = sorted(chars, key=lambda c: (round(c["top"] / tol), c["x0"]))
        lines, cur, cur_top = [], [], None
        for c in chars:
            if cur_top is None or abs(c["top"] - cur_top) <= tol:
                cur.append(c)
                cur_top = c["top"] if cur_top is None else cur_top
            else:
                lines.append(cur)
                cur, cur_top = [c], c["top"]
        if cur:
            lines.append(cur)
        out = []
        for ln in lines:
            ln = sorted(ln, key=lambda c: c["x0"])
            txt = ""
            for j, c in enumerate(ln):
                if j and c["x0"] - ln[j - 1]["x1"] > c.get("size", 9) * 0.6:
                    txt += " "
                txt += c["text"]
            out.append((ln[0]["top"], max(c["x1"] for c in ln), txt.strip()))
        return out

    @staticmethod
    def _merge_lines(lines, col_right: float) -> List[Tuple[float, str]]:
        """Merge visual lines into paragraphs (CJK: no space when joining). Returns (top, paragraph)."""
        paras, buf, buf_top = [], "", None
        for top, x1, txt in lines:
            if not txt:
                continue
            if HEADING_RE.match(txt) and buf:
                paras.append((buf_top, buf))
                buf = ""
            if not buf:
                buf_top = top
            buf += txt
            short_line = x1 < col_right - 25
            if HEADING_RE.match(txt) or (txt[-1] in SENT_END and short_line):
                paras.append((buf_top, buf))
                buf = ""
        if buf:
            paras.append((buf_top, buf))
        return paras

    # ------------------------------------------------------------------ main
    def parse(self, path: str, doc_id: Optional[str] = None) -> List[Block]:
        import pdfplumber

        doc_id = doc_id or os.path.splitext(os.path.basename(path))[0]
        raw_pages = []
        with pdfplumber.open(path) as pdf:
            for pno, page in enumerate(pdf.pages, 1):
                W, H = page.width, page.height
                tables = []
                for t in page.find_tables():
                    md = _table_to_markdown(t.extract())
                    if md:
                        tables.append((t.bbox, md))
                chars = [c for c in page.chars if c["text"].strip()]
                body = [c for c in chars if self.header_h < c["top"] and c["bottom"] < H - self.footer_h]
                self.stats["header_footer_chars_removed"] += len(chars) - len(body)
                body = [c for c in body if not any(_inside(c, bb) for bb, _ in tables)]
                gutter = self._find_gutter(body, W)
                if gutter:
                    self.stats["two_column_pages"] += 1
                    cols = [([c for c in body if c["x1"] <= gutter], gutter),
                            ([c for c in body if c["x1"] > gutter], W)]
                else:
                    cols = [(body, W)]
                items = []  # (column, top, kind, payload) -> sorted into reading order
                for ci, (cchars, right) in enumerate(cols):
                    lines = self._lines(cchars)
                    right_edge = max((x for _, x, _ in lines), default=right)
                    for top, para in self._merge_lines(lines, right_edge):
                        items.append((ci, top, "text", para))
                # tables: caption = text line just above the table bbox
                for bb, md in tables:
                    self.stats["tables"] += 1
                    above = [c for c in chars if bb[1] - 18 < c["bottom"] <= bb[1] + 1]
                    cap = "".join(c["text"] for c in sorted(above, key=lambda c: c["x0"])).strip()
                    items.append((0, bb[1], "table", (cap, md)))
                items.sort(key=lambda x: (x[0], x[1]))
                raw_pages.append((pno, items))

        # cross-page repeated lines (e.g. "请务必阅读正文之后的免责声明")
        cnt = Counter(t for _, items in raw_pages for _, _, k, t in items if k == "text" and len(t) < 60)
        n_pages = max(1, len(raw_pages))
        repeated = {t for t, v in cnt.items() if v >= max(2, self.repeat_ratio * n_pages)}

        blocks, section, title = [], "", ""
        for pno, items in raw_pages:
            for _, _, kind, payload in items:
                if kind == "table":
                    cap, md = payload
                    blocks.append(Block(doc_id, pno, "table", (cap + "\n" if cap else "") + md, section=cap or section))
                    continue
                t = payload
                if t in repeated:
                    self.stats["repeated_lines_removed"] += 1
                    continue
                if re.match(r"^表\d+[：:]", t) or t.startswith("资料来源"):
                    continue  # table caption / source already bound to the table
                if not title and pno == 1:
                    title = t
                if HEADING_RE.match(t) and len(t) < 40:
                    section = t
                blocks.append(Block(doc_id, pno, "text", t, section=section))
        for b in blocks:
            b.title = title
        return blocks


def get_parser(name: str, **kw):
    return LayoutParser(**kw) if name == "layout" else NaiveParser()
