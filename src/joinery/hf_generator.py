"""Batched greedy generation with Hugging Face transformers (fp16 or 4-bit on T4/P100)."""

from __future__ import annotations

import platform
from collections.abc import Sequence
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from joinery.config import GenerationConfig, ModelConfig
from joinery.generation import Generation
from joinery.prompts import Message

_DTYPES = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (all devices)."""
    transformers.set_seed(seed)


class HFGenerator:
    """Greedy, left-padded batched decoding with a causal LM.

    Decoding uses the model's own generation config (for Qwen2.5 that includes its repetition
    penalty) with sampling turned off. The effective settings are recorded by :meth:`describe`.
    """

    def __init__(self, model_config: ModelConfig, generation_config: GenerationConfig) -> None:
        self._model_config = model_config
        self._generation_config = generation_config
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.name_or_path, revision=model_config.revision
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = _DTYPES[model_config.dtype]
        quantization = (
            BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
            )
            if model_config.load_in_4bit
            else None
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.name_or_path,
            revision=model_config.revision,
            torch_dtype=dtype,
            quantization_config=quantization,
            device_map="auto",
        )
        self.model.eval()
        eos = self.model.generation_config.eos_token_id
        self._eos_ids = set(eos if isinstance(eos, list) else [eos])

    def render_prompt(self, messages: list[Message]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    @torch.inference_mode()
    def generate(self, prompts: Sequence[str]) -> list[Generation]:
        batch = self.tokenizer(
            list(prompts), return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)
        output = self.model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=self._generation_config.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = output[:, batch["input_ids"].shape[1] :].tolist()
        results = []
        for ids in new_tokens:
            stop = next((i for i, token in enumerate(ids) if token in self._eos_ids), None)
            kept = ids if stop is None else ids[:stop]
            text = self.tokenizer.decode(kept, skip_special_tokens=True)
            results.append(Generation(text=text, num_tokens=len(kept), truncated=stop is None))
        return results

    def describe(self) -> dict[str, Any]:
        return {
            "model": self._model_config.name_or_path,
            "revision": self._model_config.revision,
            "commit_hash": getattr(self.model.config, "_commit_hash", None),
            "dtype": self._model_config.dtype,
            "load_in_4bit": self._model_config.load_in_4bit,
            "max_new_tokens": self._generation_config.max_new_tokens,
            "batch_size": self._generation_config.batch_size,
            "do_sample": False,
            "model_generation_config": self.model.generation_config.to_diff_dict(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
