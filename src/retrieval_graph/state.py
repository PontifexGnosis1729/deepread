"""State management for the retrieval graph.

This module defines the state structures and reduction functions used in the
retrieval graph. It includes definitions for document indexing, retrieval,
and conversation management.

Classes:
    IndexState: Represents the state for document indexing operations.
    RetrievalState: Represents the state for document retrieval operations.
    ConversationState: Represents the state of the ongoing conversation.

Functions:
    reduce_docs: Processes and reduces document inputs into a sequence of Documents.
    reduce_retriever: Updates the retriever in the state.
    reduce_messages: Manages the addition of new messages to the conversation state.
    reduce_retrieved_docs: Handles the updating of retrieved documents in the state.

The module also includes type definitions and utility functions to support
these state management operations.
"""

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Optional, Sequence, Union

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from retrieval_graph.utils import reduce_docs

############################  Index State  #############################


# The index state defines the simple IO for the single-node index graph
@dataclass(kw_only=True)
class IndexState:
    """Represents the state for document indexing and retrieval.

    This class defines the structure of the index state, which includes
    the documents to be indexed and the retriever used for searching
    these documents.
    """

    file_path: str
    docs: Annotated[Sequence[Document], reduce_docs]
    """A list of documents that the agent can index."""


#############################  Agent State  ###################################


# Optional, the InputState is a restricted version of the State that is used to
# define a narrower interface to the outside world vs. what is maintained
# internally.
@dataclass(kw_only=True)
class InputState:
    """Represents the input state for the agent.

    This class defines the structure of the input state, which includes
    the messages exchanged between the user and the agent. It serves as
    a restricted version of the full State, providing a narrower interface
    to the outside world compared to what is maintained internally.
    """

    file_path: str
    raw_search_qry: str


# This is the primary state of your agent, where you can store any information


def add_queries(existing: Sequence[str], new: Sequence[str]) -> Sequence[str]:
    """Combine existing queries with new queries.

    Args:
        existing (Sequence[str]): The current list of queries in the state.
        new (Sequence[str]): The new queries to be added.

    Returns:
        Sequence[str]: A new list containing all queries from both input sequences.
    """
    return list(existing) + list(new)


@dataclass(kw_only=True)
class State(InputState):
    """The state of your graph / agent."""

    steps: list[str] = field(default_factory=list)
    """A list of steps in the research plan."""

    llm_generated_qry: Annotated[list[str], add_queries] = field(default_factory=list)
    """A list of search queries that the agent has generated."""

    retrieved_docs: Annotated[list[Document], reduce_docs] = field(default_factory=list)
    """Populated by the retriever. This is a list of documents that the agent can reference."""

    reranked_docs: list[Document] = field(default_factory=list)
    """Populated by the reranker. This is a list of documents that have been reranked."""

    messages: Annotated[Sequence[AnyMessage], add_messages]
    """Messages track the primary execution state of the agent."""

    # Feel free to add additional attributes to your state as needed.
    # Common examples include retrieved documents, extracted entities, API connections, etc.
