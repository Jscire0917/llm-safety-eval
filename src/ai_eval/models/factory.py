from typing import Optional
from .base import BaseLLM 
from .openai_model import OpenAIModel
# We import OllamaModel in a try-except block to handle cases where the required dependencies for OllamaModel are not installed.
try:
    from .ollama_model import OllamaModel
except ImportError:
    OllamaModel = None 
    
def get_model(provider: str, model_name: str) -> BaseLLM: 
    """ Simple factory method to get LLM model instances based on provider and model name. """
    if provider.lower() in ("openai", "azure"):
        return OpenAIModel(model_name)
    
    elif provider.lower() == "ollama":
        if OllamaModel is None:
            raise ImportError("OllamaModel is not available. Please install the required dependencies.")
        return OllamaModel(model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported: openai, ollama.")
    