"""Qwen loading and deterministic answer inference."""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ExperimentConfig


def load_model_and_tokenizer(config: ExperimentConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype="auto",
        device_map=config.device_map,
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_response(model, tokenizer, prompt: str, config: ExperimentConfig) -> tuple[str, int, float]:
    """Generate one response with Qwen thinking explicitly disabled."""
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
    started = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    elapsed = time.perf_counter() - started
    new_ids = output_ids[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True), len(new_ids), elapsed
