"""Chunking strategies.

* chunk_size / overlap  in characters (中文 1 字 ≈ 0.6~1 token for Qwen/BGE tokenizers)
* sentence-aware recursive splitting: paragraph -> sentence -> hard cut
* table protection: 表格不被切断；表格作为独立 chunk，带表格标题；超长表按行切分并在每块重复表头
* title prefix: embedding 文本前拼接《文档标题》+ 章节，解决“chunk 里只有数字没有主语”的问题
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List

SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    type: str
    text: str
    section: str = ""
    title: str = ""

    def embed_text(self, title_prefix: bool = True) -> str:
        if not title_prefix:
            return self.text
        head = f"《{self.doc_id}》"
        if self.section and self.section not in self.text[:40]:
            head += self.section
        return head + "\n" + self.text

    def to_dict(self):
        return asdict(self)


def _split_long(text: str, size: int) -> List[str]:
    sents = [s for s in SENT_SPLIT.split(text) if s]
    out = []
    for s in sents:
        while len(s) > size:
            out.append(s[:size])
            s = s[size:]
        if s:
            out.append(s)
    return out


def _pack(pieces: List[str], size: int, overlap: int) -> List[str]:
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) > size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap > 0 else ""
            # 尽量从句子边界开始 overlap
            m = re.search(r"[。！？；]", tail)
            if m and m.end() < len(tail):
                tail = tail[m.end():]
            cur = tail
        cur += p
    if cur.strip():
        chunks.append(cur)
    return chunks


def split_table(text: str, size: int) -> List[str]:
    lines = text.split("\n")
    if len(text) <= int(size * 1.5):
        return [text]
    cap = lines[0] if not lines[0].startswith("|") else ""
    body = lines[1:] if cap else lines
    header, rows = body[:2], body[2:]
    out, cur = [], []
    for r in rows:
        if cur and len("\n".join([cap] + header + cur + [r])) > size:
            out.append("\n".join([x for x in [cap] + header + cur if x]))
            cur = []
        cur.append(r)
    if cur:
        out.append("\n".join([x for x in [cap] + header + cur if x]))
    return out


def chunk_blocks(blocks: List[Dict], size: int = 512, overlap: int = 50, table_protect: bool = True) -> List[Chunk]:
    """blocks: parsed Block dicts in reading order (from pdf_parser)."""
    chunks: List[Chunk] = []
    by_doc: Dict[str, List[Dict]] = {}
    for b in blocks:
        by_doc.setdefault(b["doc_id"], []).append(b)

    for doc_id, bs in by_doc.items():
        title = bs[0].get("title", "")
        stream: List[tuple] = []  # (page, section, piece)
        for b in bs:
            if b["type"] == "table" and table_protect:
                for i, t in enumerate(split_table(b["text"], size)):
                    chunks.append(Chunk(f"{doc_id}#t{len(chunks)}", doc_id, b["page"], "table", t,
                                        b.get("section", ""), title))
                continue
            text = b["text"] if b["type"] == "text" else b["text"].replace("|", " ")
            for piece in _split_long(text, size):
                stream.append((b["page"], b.get("section", ""), piece))
        # pack text pieces, remembering the page/section where each chunk starts
        cur, cur_page, cur_sec = "", None, ""
        for page, sec, p in stream:
            if cur and len(cur) + len(p) > size:
                chunks.append(Chunk(f"{doc_id}#c{len(chunks)}", doc_id, cur_page, "text", cur, cur_sec, title))
                tail = cur[-overlap:] if overlap > 0 else ""
                m = re.search(r"[。！？；]", tail)
                if m and m.end() < len(tail):
                    tail = tail[m.end():]
                cur, cur_page, cur_sec = tail, page, sec
            if cur_page is None or not cur:
                cur_page, cur_sec = page, sec
            cur += p
        if cur.strip():
            chunks.append(Chunk(f"{doc_id}#c{len(chunks)}", doc_id, cur_page, "text", cur, cur_sec, title))
    return chunks
