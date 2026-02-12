from abc import ABC, abstractmethod
# This is the base class for all LLMs. It defines the interface that all LLMs must implement.
class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
