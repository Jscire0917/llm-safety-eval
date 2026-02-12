from openai import OpenAI
from ai_eval.models.base import BaseLLM
import os 
# This class is a wrapper around the OpenAI API to provide a consistent interface for generating text using different models.
# It initializes the OpenAI client using an API key from the environment variables and defines a method to generate text based on a given prompt.
class OpenAIModel(BaseLLM):
    def __init__(self, model_name: str):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI models/embeddings")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name


    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
