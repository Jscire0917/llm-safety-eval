# src/ai_eval/models/mlx_model.py
"""
MLX model wrapper for fast, optimized inference on Apple Silicon (M1/M2/M3/M4).
Uses mlx-lm for quantized models from Hugging Face.
No CUDA required – fully native on Mac.
"""

from mlx_lm import load, generate
from .base import BaseLLM
from typing import Optional


class MLXModel(BaseLLM):
    """
    High-performance local LLM inference using Apple's MLX framework.
    Excellent speed and memory efficiency on M-series Macs.
    Supports quantized models (e.g. 4-bit, 8-bit) from mlx-community.
    """

    def __init__(
        self,
        model_path: str,
        max_tokens: int = 512,
        temp: float = 0.0,
        quantize: Optional[str] = "4bit"  # or "8bit", None
    ):
        """
        Args:
            model_path: HuggingFace path or local directory
                        Examples:
                        - "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
                        - "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
            max_tokens: Max tokens to generate
            temp: Sampling temperature (0.0 = deterministic)
            quantize: Quantization level ("4bit", "8bit", None)
        """
        print(f"[MLX] Loading model: {model_path}")
        self.model, self.tokenizer = load(model_path)
        self.max_tokens = max_tokens
        self.temp = temp

    def generate(self, prompt: str) -> str:
        """
        Generate text from the model.

        Args:
            prompt: Input text

        Returns:
            Generated text
        """
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temp=self.temp,
            verbose=False
        )
        return response.strip()