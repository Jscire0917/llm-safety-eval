from openai import OpenAI
from .base import BaseLLM 
# This class is a wrapper around the Ollama API, which is compatible with the OpenAI API.
# It allows us to use Ollama as a language model in our evaluation framework.
class OllamaModel(BaseLLM):
    def __init__(self, model_name: str):
        # Ollama runs on http://localhost:11434 by default
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama", # dummy key - Ollama ignores it
            )
        self.model_name = model_name 
        
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()