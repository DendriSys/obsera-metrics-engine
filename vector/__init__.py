"""Vector embeddings and storage module"""

from .ollama_client import OllamaClient, OllamaConfig
from .scalable_vector_store import ScalableVectorStore, MetricEmbedding

# Use scalable version as default
VectorStore = ScalableVectorStore

__all__ = [
    "OllamaClient",
    "OllamaConfig",
    "VectorStore",
    "ScalableVectorStore",
    "MetricEmbedding"
]
