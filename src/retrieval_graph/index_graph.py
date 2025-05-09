"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""
import os
from pathlib import Path
from typing import Optional, Sequence, Literal

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


def gather_files(state: IndexState, *, config: Optional[RunnableConfig] = None) -> dict[str, str]:
    """Gather the files to be indexed."""
    list_files = []

    if "reset" == state.input_file_path:
        WEAVIATE_DOCS_INDEX_NAME = os.environ["WEAVIATE_DOCS_INDEX_NAME"]
        RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]
        record_manager = SQLRecordManager(f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL)

        keys = record_manager.list_keys()
        record_manager.delete_keys(keys)
        retrieval.reset_retrieval_db(config)

        return {"docs": "delete", "working_dir": list_files}


    target = Path(state.input_file_path)

    if(target.is_file() and target.suffix.lower() == ".epub"):
        list_files.append(str(target.resolve()))
    elif(target.is_dir()):
        pattern = "**/*.epub"
        list_files.extend([str(p.resolve()) for p in target.glob(pattern)])
    else:
        raise ValueError(f"Invalid file path: {target}")

    return {"docs": "delete", "working_dir": list_files}


def file_streaming(state: IndexState, *, config: Optional[RunnableConfig] = None) -> dict[str, str]:
    if (state.working_dir):
        next_file = state.working_dir[0]
    else:
        next_file = "DONE"

    return {"target_file": next_file}


def ingest_docs(state: IndexState, *, config: Optional[RunnableConfig] = None) -> dict[str, str]:

    # Load the EPUB document
    docs = load_epub_docs(state.target_file)

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
            vector_store=retriever.vectorstore,
            cleanup="incremental",
            source_id_key="source",
        )

    return {"docs": "delete", "working_dir": state.working_dir[1:]}


def check_finished(state: IndexState, *, config: RunnableConfig) -> Literal["ingest_docs", END]: # type: ignore
    """Determine if the research process is complete or if more research is needed."""

    if (state.working_dir):
        return "ingest_docs"
    else:
        return END


# Define a new graph
builder = StateGraph(IndexState, input=InputIndexState, config_schema=IndexConfiguration)

builder.add_node(gather_files)
builder.add_node(file_streaming)
builder.add_node(ingest_docs)
builder.add_node(index_docs)

builder.add_edge(START, "gather_files")
builder.add_edge("gather_files", "file_streaming")
builder.add_conditional_edges("file_streaming", check_finished)
builder.add_edge("ingest_docs", "index_docs")
builder.add_edge("index_docs", "file_streaming")

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile()
graph.name = "IndexGraph"
