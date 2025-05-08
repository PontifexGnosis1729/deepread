"""Researcher graph used in the conversational retrieval system as a subgraph.

This module defines the core structure and functionality of the researcher graph,
which is responsible for generating search queries and retrieving relevant documents.
"""

from typing import cast

from pydantic import BaseModel
from weaviate.classes.query import Filter
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from retrieval_graph import retrieval
from retrieval_graph.configuration import Configuration
from retrieval_graph.researcher_graph.state import ResearcherState
from retrieval_graph.utils import load_chat_model


async def generate_queries(state: ResearcherState, *, config: RunnableConfig) -> dict[str, list[str]]:
    """Generate search queries based on a step in the research plan and the users query.

    Args:
        state (ResearcherState): The current state of the researcher, including the user's question.
        config (RunnableConfig): Configuration with the model used to generate queries.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing the list of generated search queries.
    """
    class SearchQuery(BaseModel):
        queries: list[str]


    configuration = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_template(configuration.generate_queries_system_prompt)
    model = load_chat_model(configuration.query_model).with_structured_output(SearchQuery)

    message_value = await prompt.ainvoke(
        {
            "original_query": state.raw_search_qry,
            "research_step": state.research_step,
        },
        config,
    )
    response = cast(SearchQuery, await model.ainvoke(message_value, config))

    return {"llm_generated_queries": response.queries}


async def generate_hyde_passages(state: ResearcherState, *, config: RunnableConfig) -> dict[str, list[str]]:
    """Generate a search query based on the current state and configuration.

    This function analyzes the messages in the state and generates an appropriate
    search query. For the first message, it uses the user's input directly.
    For subsequent messages, it uses a language model to generate a refined query.

    Args:
        state (State): The current state containing messages and other information.
        config (RunnableConfig | None, optional): Configuration for the query generation process.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing a list of generated queries.

    Behavior:
        - If there's only one message (first user input), it uses that as the query.
        - For subsequent messages, it uses a language model to generate a refined query.
        - The function uses the configuration to set up the prompt and model for query generation.
    """

    configuration = Configuration.from_runnable_config(config)

    prompt = ChatPromptTemplate.from_template(configuration.generate_hyde_passage_system_prompt)
    model = load_chat_model(configuration.query_model)
    hyde_passages = []

    for qry in state.llm_generated_queries:
        message_value = await prompt.ainvoke(
            {
                "current_query": qry,
            },
            config,
        )
        response = await model.ainvoke(message_value, config)
        hyde_passages.append(response.content)

    return {"llm_hyde_passages": hyde_passages}


async def retrieve_documents(state: ResearcherState, *, config: RunnableConfig) -> dict[str, list[Document]]:
    """Retrieve documents based on a given query.

    This function uses a retriever to fetch relevant documents for a given query.

    Args:
        state (QueryState): The current state containing the query string.
        config (RunnableConfig): Configuration with the retriever used to fetch documents.

    Returns:
        dict[str, list[Document]]: A dictionary with a 'documents' key containing the list of retrieved documents.
    """
    out = []

    #TODO: filter out duplicate docs
    with retrieval.make_retriever(config) as retriever:
        where_filter = Filter.by_property("source").equal(state.file_path)
        retriever.search_kwargs["filters"] = where_filter
        retriever.search_kwargs["k"] = 3

        for qry in state.llm_hyde_passages:
            response = await retriever.ainvoke(qry, config)
            stamped_docs = [Document(page_content=doc.page_content, metadata={**doc.metadata, "research_step": state.research_step}) for doc in response]
            out.extend(stamped_docs)


        return {"research_documents": out}


# Define the graph
builder = StateGraph(ResearcherState)
builder.add_node(generate_queries)
builder.add_node(generate_hyde_passages)
builder.add_node(retrieve_documents)

builder.add_edge(START, "generate_queries")
builder.add_edge("generate_queries", "generate_hyde_passages")
builder.add_edge("generate_hyde_passages", "retrieve_documents")
builder.add_edge("retrieve_documents", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "ResearcherGraph"
