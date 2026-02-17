# src/ai_eval/models/factory.py
"""
Factory for creating LLM instances based on provider name and model identifier.
"""

from typing import Optional
from .base import BaseLLM
from .openai_model import OpenAIModel

# Import local model wrappers (add more providers here later)
try:
    from .ollama_model import OllamaModel
except ImportError:
    OllamaModel = None


def get_model(provider: str, model_name: str) -> BaseLLM:
    """
    Create and return an LLM instance for the requested provider and model.

    Args:
        provider:   'openai', 'ollama', 'vllm', etc.
        model_name: Model identifier (e.g. 'gpt-4o-mini', 'llama3.1:8b')

    Returns:
        Initialized BaseLLM subclass

    Raises:
        ValueError: Unsupported provider
        ImportError: Required package for provider not installed
    """
    provider = provider.lower()

    if provider in ("openai", "azure"):
        return OpenAIModel(model_name)

    elif provider == "ollama":
        if OllamaModel is None:
            raise ImportError(
                "Ollama support requires the 'openai' client library.\n"
                "Ollama server must be running:  ollama serve\n"
                "Model must be pulled:          ollama pull llama3.1:8b"
            )
        return OllamaModel(model_name)
    else:
        raise ValueError(
            f"Unsupported provider: {provider!r}\n"
            f"Currently supported: 'openai', 'ollama'"
        )