"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

from typing import Any, Literal, cast

from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from retrieval_graph.configuration import Configuration
from retrieval_graph.state import InputState, State
from retrieval_graph.researcher_graph.graph import graph as researcher_graph
from retrieval_graph.utils import format_docs, load_chat_model, deduplicate_documents

# Define the function that calls the model


async def create_research_plan(state: State, *, config: RunnableConfig) -> dict[str, list[str]]:
    """Create a step-by-step research plan for finding passages from a book.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used to generate the plan.

    Returns:
        dict[str, list[str]]: A dictionary with a 'research_plan' key containing the list of research steps.
    """

    class ResearchPlan(BaseModel):
        """Generate research plan."""

        steps: list[str]

    configuration = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_template(configuration.research_plan_system_prompt)
    model = load_chat_model(configuration.query_model).with_structured_output(ResearchPlan)

    message_value = await prompt.ainvoke(
        {
            "current_query": state.raw_search_qry,
        },
        config,
    )
    response = cast(ResearchPlan, await model.ainvoke(message_value, config))

    return {"research_plan": response.steps}


async def conduct_research(state: State, *, config: RunnableConfig) -> dict[str, Any]:
    """Execute the first step of the research plan.

    Args:
        state (AgentState): The current state of the agent, including the research plan steps.

    Returns:
        dict[str, list[str]]: A dictionary with 'documents' containing the research results and
                              'research_plan' containing the remaining research steps.
    """

    result = await researcher_graph.ainvoke({"research_step": state.research_plan[0], "raw_search_qry": state.raw_search_qry, "file_path": state.file_path})

    return {"retrieved_docs": result["research_documents"], "research_plan": state.research_plan[1:]}


def check_finished(state: State, *, config: RunnableConfig) -> Literal["rerank", "conduct_research"]:
    """Determine if the research process is complete or if more research is needed.

    This function checks if there are any remaining steps in the research plan:
        - If there are, route back to the `conduct_research` node
        - Otherwise, route to the `respond` node

    Args:
        state (AgentState): The current state of the agent, including the remaining research steps.

    Returns:
        Literal["respond", "conduct_research"]: The next step to take based on whether research is complete.
    """
    if len(state.research_plan or []) > 0:
        return "conduct_research"
    else:
        return "rerank"


async def rerank(state: State, *, config: RunnableConfig) -> dict[str, list[Document]]:
    """Rerank documents based on relevance to the query in the state."""

    class RelavanceScore(BaseModel):
        score: int

    configuration = Configuration.from_runnable_config(config)

    query = state.raw_search_qry
    unsorted_docs = deduplicate_documents(state.retrieved_docs)

    # Example using LLM-based reranking
    scored_docs = []
    prompt = ChatPromptTemplate.from_template(configuration.rerank_system_prompt)
    model = load_chat_model(configuration.rerank_model).with_structured_output(RelavanceScore)

    for doc in unsorted_docs:
        message_value = await prompt.ainvoke(
            {
                "current_query": query,
                "retrieved_doc": doc.page_content,
            },
            config,
        )

        generated = cast(RelavanceScore, await model.ainvoke(message_value, config))
        scored_docs.append((doc, generated.score))

    # Sort by score descending
    sorted_tp = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    reranked_docs = [tp[0] for tp in sorted_tp]


    return {"reranked_docs": reranked_docs}


async def respond(state: State, *, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
    """Call the LLM powering our "agent"."""
    configuration = Configuration.from_runnable_config(config)

    prompt = ChatPromptTemplate.from_template(configuration.response_system_prompt)
    model = load_chat_model(configuration.response_model)

    reranked_docs = format_docs(state.reranked_docs)
    message_value = await prompt.ainvoke(
        {
            "current_query": state.raw_search_qry,
            "reranked_docs": reranked_docs,
        },
        config,
    )
    response = await model.ainvoke(message_value, config)

    return {"messages": [response], "retrieved_docs": "delete"}


# Define a new graph (It's just a pipe)


builder = StateGraph(State, input=InputState, config_schema=Configuration)

builder.add_node(create_research_plan)
builder.add_node(conduct_research)
builder.add_node(rerank)
builder.add_node(respond)

builder.add_edge(START, "create_research_plan")
builder.add_edge("create_research_plan", "conduct_research")
builder.add_conditional_edges("conduct_research", check_finished)
builder.add_edge("rerank", "respond")
builder.add_edge("respond", END)

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile()
graph.name = "RetrievalGraph"
