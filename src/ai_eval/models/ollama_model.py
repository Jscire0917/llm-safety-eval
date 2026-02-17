# src/ai_eval/models/ollama_model.py
"""
Ollama model wrapper that uses the OpenAI-compatible API endpoint provided by Ollama.
Requires Ollama running locally (ollama serve) and a model pulled (e.g. ollama pull llama3.1:8b).
"""

from openai import OpenAI
from .base import BaseLLM


class OllamaModel(BaseLLM):
    """
    LLM wrapper for models served via Ollama[](http://localhost:11434/v1).
    Uses OpenAI client with custom base_url.
    """

    def __init__(self, model_name: str):
        """
        Args:
            model_name: Name of the model as known to Ollama (e.g. 'llama3.1:8b', 'phi3:mini')
        """
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",           # dummy value – Ollama ignores the key
        )
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """
        Generate text from the model using the chat completions endpoint.

        Args:
            prompt: User prompt string

        Returns:
            Model-generated text (assistant reply)
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,           # deterministic output
            max_tokens=512,            # reasonable limit – adjust as needed
        )
        return response.choices[0].message.content.strip()