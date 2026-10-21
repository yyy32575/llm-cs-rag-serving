from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterable, List

import yaml


def read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path: str, rows: Iterable[Dict]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_WS = re.compile(r"\s+")


def norm_text(s: str) -> str:
    """Whitespace-insensitive normalization used for evidence matching."""
    return _WS.sub("", s or "").replace("，", ",").replace("（", "(").replace("）", ")")


def find_cjk_font() -> str:
    cands = [
        os.environ.get("CJK_FONT", ""),
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("未找到中文字体，请设置环境变量 CJK_FONT=/path/to/font.ttf")
