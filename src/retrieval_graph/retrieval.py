"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Elasticsearch, Pinecone, and MongoDB.

The retrievers support filtering results by user_id to ensure data isolation between users.
"""

import os
from contextlib import contextmanager
from typing import Generator

import weaviate
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever, VectorStore
from langchain_weaviate import WeaviateVectorStore

from retrieval_graph.configuration import Configuration, IndexConfiguration

## Encoder constructors


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "ollama":
            from langchain_ollama import OllamaEmbeddings

            return OllamaEmbeddings(model=model)
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)
        case "cohere":
            from langchain_cohere import CohereEmbeddings

            return CohereEmbeddings(model=model)  # type: ignore
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


## Retriever constructors


@contextmanager
def make_weaviate_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    with weaviate.connect_to_local() as weaviate_client:
        store = WeaviateVectorStore(
            client=weaviate_client,
            index_name=os.environ["WEAVIATE_DOCS_INDEX_NAME"],
            text_key="text",
            embedding=embedding_model,
            attributes=["source", "title"],
        )
        search_kwargs = configuration.search_kwargs or {}

        yield store.as_retriever(search_kwargs=search_kwargs)


@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = IndexConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    user_id = configuration.user_id
    if not user_id:
        raise ValueError("Please provide a valid user_id in the configuration.")
    match configuration.retriever_provider:
        case "weaviate":
            with make_weaviate_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(Configuration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )


@contextmanager
def make_weaviate_indexer(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStore, None, None]:
    with weaviate.connect_to_local() as weaviate_client:
        store = WeaviateVectorStore(
            client=weaviate_client,
            index_name=os.environ["WEAVIATE_DOCS_INDEX_NAME"],
            text_key="text",
            embedding=embedding_model,
            attributes=["source", "title"],
        )

        yield store


@contextmanager
def make_indexer(
    config: RunnableConfig,
) -> Generator[VectorStore, None, None]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = IndexConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    user_id = configuration.user_id
    if not user_id:
        raise ValueError("Please provide a valid user_id in the configuration.")
    match configuration.retriever_provider:
        case "weaviate":
            with make_weaviate_indexer(configuration, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(Configuration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )
