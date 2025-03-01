from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings


def get_embeddings_model() -> Embeddings:
    return OllamaEmbeddings(model="nomic-embed-text")
