# file: agentmesh/models/qwen.py
"""Qwen model provider using HuggingFace transformers."""

import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentmesh.config import settings
from agentmesh.models.base import BaseModelProvider, ModelResponse


class QwenModelProvider(BaseModelProvider):
    """Loads and serves a Qwen instruct model for chat completion.

    Lazy-loads the model on first generate() call to avoid consuming
    GPU memory at import time.
    """

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or settings.model_name
        self._model = None
        self._tokenizer = None
        self._device = settings.device

    def _ensure_loaded(self) -> None:
        """Load model and tokenizer into memory if not already loaded."""
        if self._model is not None:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()

    def is_loaded(self) -> bool:
        return self._model is not None

    def get_model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate a chat completion from message history."""
        self._ensure_loaded()

        temp = temperature if temperature is not None else settings.temperature
        max_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else settings.max_new_tokens
        )

        # Apply chat template to format messages for the model
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[1]

        start_time = time.perf_counter()

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp if temp > 0 else None,
                top_p=settings.top_p if temp > 0 else None,
                do_sample=temp > 0,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Decode only the generated tokens (not the input)
        generated_ids = output_ids[0][input_len:]
        response_text = self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        tokens_out = len(generated_ids)

        return ModelResponse(
            text=response_text.strip(),
            tokens_in=input_len,
            tokens_out=tokens_out,
            latency_ms=round(latency_ms, 2),
            model_name=self._model_name,
        )