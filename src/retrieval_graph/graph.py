"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

from typing import Any, Literal, cast

from pydantic import BaseModel
from weaviate.classes.query import Filter
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from retrieval_graph import retrieval
from retrieval_graph.configuration import Configuration
from retrieval_graph.state import InputState, State
from retrieval_graph.researcher_graph.graph import graph as researcher_graph
from retrieval_graph.utils import format_docs, load_chat_model

# Define the function that calls the model


async def create_research_plan(state: State, *, config: RunnableConfig) -> dict[str, list[str]]:
    """Create a step-by-step research plan for answering a LangChain-related query.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used to generate the plan.

    Returns:
        dict[str, list[str]]: A dictionary with a 'steps' key containing the list of research steps.
    """

    class Plan(BaseModel):
        """Generate research plan."""

        steps: list[str]

    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model).with_structured_output(Plan)

    #TODO: fix prompt
    messages = [
        {"role": "system", "content": configuration.research_plan_system_prompt}
    ] + state.messages
    response = cast(Plan, await model.ainvoke(messages))

    return {"steps": response.steps}


async def conduct_research(state: State, *, config: RunnableConfig) -> dict[str, Any]:
    """Execute the first step of the research plan.

    This function takes the first step from the research plan and uses it to conduct research.

    Args:
        state (AgentState): The current state of the agent, including the research plan steps.

    Returns:
        dict[str, list[str]]: A dictionary with 'documents' containing the research results and
                              'steps' containing the remaining research steps.

    Behavior:
        - Invokes the researcher_graph with the first step of the research plan.
        - Updates the state with the retrieved documents and removes the completed step.
    """

    #TODO: double check documents dont create duplicates
    result = await researcher_graph.ainvoke({"question": state.steps[0]})

    return {"retrieved_docs": result["research_documents"], "steps": state.steps[1:]}


def check_finished(state: State, *, config: RunnableConfig) -> Literal["respond", "conduct_research"]:
    """Determine if the research process is complete or if more research is needed.

    This function checks if there are any remaining steps in the research plan:
        - If there are, route back to the `conduct_research` node
        - Otherwise, route to the `respond` node

    Args:
        state (AgentState): The current state of the agent, including the remaining research steps.

    Returns:
        Literal["respond", "conduct_research"]: The next step to take based on whether research is complete.
    """
    if len(state.steps or []) > 0:
        return "conduct_research"
    else:
        return "rerank"


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


async def generate_query(state: State, *, config: RunnableConfig) -> dict[str, list[str]]:
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
    human_input = state.raw_search_qry
    configuration = Configuration.from_runnable_config(config)
    # Feel free to customize the prompt, model, and other logic!
    prompt = ChatPromptTemplate.from_template(configuration.query_system_prompt)

    model = load_chat_model(configuration.query_model).with_structured_output(SearchQuery)

    message_value = await prompt.ainvoke(
        {
            "current_query": human_input,
        },
        config,
    )
    generated = cast(SearchQuery, await model.ainvoke(message_value, config))

    return {"llm_generated_qry": [generated.query]}


async def retrieve(state: State, *, config: RunnableConfig) -> dict[str, list[Document]]:
    """Retrieve documents based on the latest query in the state.

    This function takes the current state and configuration, uses the latest query
    from the state to retrieve relevant documents using the retriever, and returns
    the retrieved documents.

    Args:
        state (State): The current state containing queries and the retriever.
        config (RunnableConfig | None, optional): Configuration for the retrieval process.

    Returns:
        dict[str, list[Document]]: A dictionary with a single key "retrieved_docs"
        containing a list of retrieved Document objects.
    """
    with retrieval.make_retriever(config) as retriever:
        where_filter = Filter.by_property("source").equal(state.file_path)
        retriever.search_kwargs["filters"] = where_filter
        retriever.search_kwargs["k"] = 10

        response = await retriever.ainvoke(state.llm_generated_qry[-1], config)
        return {"retrieved_docs": response}


async def rerank(state: State, *, config: RunnableConfig) -> dict[str, list[Document]]:
    """Rerank documents based on relevance to the query in the state."""

    class RelavanceScore(BaseModel):
        score: int

    configuration = Configuration.from_runnable_config(config)

    query = state.raw_search_qry
    unsorted_docs = state.retrieved_docs

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

    return {"messages": [response]}


# Define a new graph (It's just a pipe)


builder = StateGraph(State, input=InputState, config_schema=Configuration)

#builder.add_node(generate_query)
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
