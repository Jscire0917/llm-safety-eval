from abc import ABC, abstractmethod
# This is the base class for embedding providers. It defines the interface for embedding providers.
class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        pass
