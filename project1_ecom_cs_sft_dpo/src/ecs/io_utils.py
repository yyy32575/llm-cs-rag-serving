from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterable, List


def read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def filter_kwargs(cls, kwargs: Dict) -> Dict:
    """Keep only kwargs accepted by a (dataclass) config class.

    TRL renames arguments frequently (max_seq_length -> max_length, tokenizer -> processing_class...).
    This keeps the scripts runnable across minor versions.
    """
    import dataclasses
    import inspect

    if dataclasses.is_dataclass(cls):
        names = {f.name for f in dataclasses.fields(cls)}
    else:
        names = set(inspect.signature(cls.__init__).parameters)
    dropped = sorted(k for k in kwargs if k not in names)
    if dropped:
        print(f"[filter_kwargs] {cls.__name__} ignores unsupported args: {dropped}")
    return {k: v for k, v in kwargs.items() if k in names}
