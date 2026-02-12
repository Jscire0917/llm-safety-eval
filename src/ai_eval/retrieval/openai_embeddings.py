from openai import OpenAI
from ai_eval.retrieval.embedding_base import EmbeddingProvider
import os 
# This class provides an implementation of the EmbeddingProvider interface using OpenAI's embedding API. 
# It initializes an OpenAI client with the provided API key and defines a method to generate embeddings for a given text input using the specified model.
class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model="text-embedding-3-small"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI models/embeddings")
        self.client = OpenAI(api_key=api_key)
        
        
        
        self.model = model

    def embed(self, text):
        return self.client.embeddings.create(
            model=self.model,
            input=text
        ).data[0].embedding
