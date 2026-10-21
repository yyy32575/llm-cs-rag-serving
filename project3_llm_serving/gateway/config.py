from __future__ import annotations

import os
from typing import Any, Dict

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_settings(path: str = None) -> Dict[str, Any]:
    path = path or os.environ.get("GATEWAY_CONFIG", os.path.join(ROOT, "configs", "gateway.yaml"))
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in list(cfg.items()):
        env = os.environ.get(k.upper())
        if env is not None and not isinstance(v, (dict, list)):
            cfg[k] = type(v)(env) if v is not None and not isinstance(v, bool) else env
    return cfg
