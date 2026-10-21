"""Generation backends shared by candidate sampling (DPO) and evaluation.

  * hf     : transformers + (optional) LoRA adapter, batched left-padded generation
  * vllm   : vLLM offline engine (fast; supports LoRA adapters via LoRARequest)
  * openai : any OpenAI-compatible server (e.g. the vLLM server of project 3)
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from .prompts import build_messages, render_prompt


class Generator:
    def generate(self, examples: List[Dict], n: int = 1, temperature: float = 0.0, top_p: float = 1.0,
                 max_new_tokens: int = 256) -> List[List[str]]:
        raise NotImplementedError


class HFGenerator(Generator):
    def __init__(self, model_path: str, adapter: Optional[str] = None, load_4bit: bool = False, batch_size: int = 8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(adapter or model_path, trust_remote_code=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # 非 CUDA 设备（Apple MPS / CPU）：fp32 + 先挂 adapter 再整体 .to(device)
        dev = None
        if torch.cuda.is_available():
            kw = dict(torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        else:
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            kw = dict(torch_dtype=torch.float32, trust_remote_code=True, attn_implementation="eager")
        if load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        if dev:
            self.model = self.model.to(dev)
        self.model.eval()
        self.bs = batch_size

    def generate(self, examples, n=1, temperature=0.0, top_p=1.0, max_new_tokens=256):
        outs: List[List[str]] = []
        prompts = [render_prompt(self.tok, e) for e in examples]
        for i in range(0, len(prompts), self.bs):
            batch = prompts[i:i + self.bs]
            enc = self.tok(batch, return_tensors="pt", padding=True).to(self.model.device)
            gen_kw = dict(max_new_tokens=max_new_tokens, num_return_sequences=n, pad_token_id=self.tok.pad_token_id,
                          repetition_penalty=1.05)
            if temperature > 0:
                gen_kw.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                gen_kw.update(do_sample=False)
            with self.torch.no_grad():
                ids = self.model.generate(**enc, **gen_kw)
            ids = ids[:, enc["input_ids"].shape[1]:]
            texts = self.tok.batch_decode(ids, skip_special_tokens=True)
            for j in range(len(batch)):
                outs.append([t.strip() for t in texts[j * n:(j + 1) * n]])
        return outs


class VLLMGenerator(Generator):
    def __init__(self, model_path: str, adapter: Optional[str] = None, max_lora_rank: int = 64,
                 gpu_memory_utilization: float = 0.9, quantization: Optional[str] = None):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(model=model_path, enable_lora=adapter is not None, max_lora_rank=max_lora_rank,
                       gpu_memory_utilization=gpu_memory_utilization, quantization=quantization,
                       max_model_len=4096, trust_remote_code=True)
        self.lora = None
        if adapter:
            from vllm.lora.request import LoRARequest

            self.lora = LoRARequest("adapter", 1, adapter)

    def generate(self, examples, n=1, temperature=0.0, top_p=1.0, max_new_tokens=256):
        from vllm import SamplingParams

        sp = SamplingParams(n=n, temperature=temperature, top_p=top_p if temperature > 0 else 1.0,
                            max_tokens=max_new_tokens, repetition_penalty=1.05)
        prompts = [render_prompt(self.tok, e) for e in examples]
        res = self.llm.generate(prompts, sp, lora_request=self.lora)
        return [[o.text.strip() for o in r.outputs] for r in res]


class OpenAIGenerator(Generator):
    def __init__(self, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 concurrency: int = 16):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
                             api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self.model = model
        self.concurrency = concurrency

    def _one(self, e, n, temperature, top_p, max_new_tokens):
        r = self.client.chat.completions.create(
            model=self.model, messages=build_messages(e), n=n, temperature=temperature, top_p=top_p,
            max_tokens=max_new_tokens, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return [(c.message.content or "").strip() for c in r.choices]

    def generate(self, examples, n=1, temperature=0.0, top_p=1.0, max_new_tokens=256):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(self.concurrency) as ex:
            return list(ex.map(lambda e: self._one(e, n, temperature, top_p, max_new_tokens), examples))


def get_generator(backend: str, model: str, adapter: Optional[str] = None, **kw) -> Generator:
    if backend == "hf":
        return HFGenerator(model, adapter, load_4bit=kw.get("load_4bit", False), batch_size=kw.get("batch_size", 8))
    if backend == "vllm":
        return VLLMGenerator(model, adapter, quantization=kw.get("quantization"))
    if backend == "openai":
        return OpenAIGenerator(model, kw.get("base_url"))
    raise ValueError(backend)
