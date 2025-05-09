"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""
import os
from typing import Optional, Sequence

from langchain.indexes import SQLRecordManager, index
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from retrieval_graph import retrieval
from retrieval_graph.state import InputIndexState, IndexState
from retrieval_graph.configuration import IndexConfiguration
from retrieval_graph.custom_epubloader import load_epub_docs



def ensure_docs_have_user_id(docs: Sequence[Document], config: RunnableConfig) -> list[Document]:
    """Ensure that all documents have a user_id in their metadata."""
    user_id = config["configurable"]["user_id"]

    return [Document(page_content=doc.page_content, metadata={**doc.metadata, "user_id": user_id}) for doc in docs]


def ingest_docs(state: IndexState, *, config: Optional[RunnableConfig] = None) -> dict[str, str]:
    if "reset" == state.file_path:
        WEAVIATE_DOCS_INDEX_NAME = os.environ["WEAVIATE_DOCS_INDEX_NAME"]
        RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]
        record_manager = SQLRecordManager(f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL)

        keys = record_manager.list_keys()
        record_manager.delete_keys(keys)
        retrieval.reset_retrieval_db(config)

        return {"docs": "delete"}

    if not os.path.isfile(state.file_path):
        raise ValueError(f"invalid input file path: {state.file_path}")

    # parse file path
    docs = load_epub_docs(state.file_path)

    return {"docs": docs}


async def index_docs(state: IndexState, *, config: Optional[RunnableConfig] = None) -> dict[str, str]:
    """Asynchronously index documents in the given state using the configured retriever.

    Args:
        state (IndexState): The current state containing documents and retriever.
        config (Optional[RunnableConfig]): Configuration for the indexing process.r
    """
    if not config:
        raise ValueError("Configuration required to run index_docs.")

    WEAVIATE_DOCS_INDEX_NAME = os.environ["WEAVIATE_DOCS_INDEX_NAME"]
    RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]

    with retrieval.make_retriever(config) as retriever:
        stamped_docs = ensure_docs_have_user_id(state.docs, config)
        record_manager = SQLRecordManager(
            f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
        )
        record_manager.create_schema()

        index(
            docs_source=stamped_docs,
            record_manager=record_manager,
            vector_store=retriever.vectorstore,         # your VectorStoreRetriever
            cleanup="incremental",
            source_id_key="source",        # ensure your Document.metadata["source"] is set
        )

    return {"docs": "delete"}


# Define a new graph
builder = StateGraph(IndexState, input=InputIndexState, config_schema=IndexConfiguration)
builder.add_node(ingest_docs)
builder.add_node(index_docs)
builder.add_edge(START, "ingest_docs")
builder.add_edge("ingest_docs", "index_docs")
builder.add_edge("index_docs", END)

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile()
graph.name = "IndexGraph"
