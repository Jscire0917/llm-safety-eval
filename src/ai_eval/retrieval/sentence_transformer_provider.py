# src/ai_eval/retrieval/sentence_transformer_provider.py
"""
Local embedding provider using sentence-transformers.
Fast, offline, and good quality for English text.
No API key required — works with Ollama fully local.
"""

from sentence_transformers import SentenceTransformer
from .embedding_base import EmbeddingProvider


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Embedding provider using pre-trained sentence-transformers models.
    Ideal for local/offline use with Ollama.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Pre-trained model from HuggingFace
                        - "all-MiniLM-L6-v2": Fast, general-purpose (384 dims)
                        - "all-mpnet-base-v2": Higher accuracy (768 dims, slower)
        """
        print(f"[INFO] Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding vector for the input text.

        Args:
            text: Input string to embed

        Returns:
            List of floats (embedding vector)
        """
        embedding = self.model.encode(text)
        return embedding.tolist()  # Convert to native list for JSON/serialization