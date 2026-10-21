"""Prompt construction shared by training / inference / evaluation (keep them identical!)."""
from __future__ import annotations

from typing import Dict, List

SYSTEM_PROMPT = (
    "你是某电商平台的智能客服“小智”。请严格依据平台售后与物流规则回答用户问题：\n"
    "1. 先共情/致歉（如适用），再给出明确的政策依据，最后告诉用户下一步具体操作路径；\n"
    "2. 不确定或涉及赔偿认定、威胁投诉升级的问题，礼貌说明并转人工客服；\n"
    "3. 不要编造政策和商品参数，不要重复啰嗦，回答控制在150字以内。"
)


def build_user_content(instruction: str, inp: str = "") -> str:
    inp = (inp or "").strip()
    if inp:
        return f"【背景信息】{inp}\n【用户问题】{instruction}"
    return instruction


def build_messages(ex: Dict, with_answer: bool = False) -> List[Dict[str, str]]:
    """ex: alpaca-style {instruction, input, output, history:[[q, a], ...]}"""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for q, a in ex.get("history", []) or []:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": build_user_content(ex["instruction"], ex.get("input", ""))})
    if with_answer:
        msgs.append({"role": "assistant", "content": ex["output"]})
    return msgs


def render_prompt(tokenizer, ex: Dict) -> str:
    """Render the generation prompt with the model chat template (Qwen3: thinking disabled)."""
    msgs = build_messages(ex)
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:  # tokenizers whose template does not accept enable_thinking
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
