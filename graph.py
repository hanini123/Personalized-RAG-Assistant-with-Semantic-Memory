
import os
import uuid
import operator
import streamlit as st
from typing import Annotated

from langchain_core.messages import SystemMessage, merge_message_runs
from langchain_core.runnables.config import RunnableConfig
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.document_loaders import WikipediaLoader
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore

from pydantic import BaseModel, Field
from trustcall import create_extractor

from prompts import MODEL_SYSTEM_MESSAGE,search_instructions,answer_template,TRUSTCALL_INSTRUCTION 

# ---------------------------------------------------------------------
# Graph State Definition
# ---------------------------------------------------------------------
# Extends the default MessagesState by adding:
# - search_query: used when performing retrieval
# - context: stores retrieved documents (merged across retrieval nodes)
class MessagesState(MessagesState):
    search_query: str
    context: Annotated[list, operator.add]

# ---------------------------------------------------------------------
# Initialize environment variables
# ---------------------------------------------------------------------
def keys_init(NVIDIA_API_KEY=None, TAVILY_API_KEY=None):
    os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# ---------------------------------------------------------------------
# Main Assistant Node
# ---------------------------------------------------------------------
# - Loads long-term memory from the store
# - Injects memory into system prompt
# - Calls the main chat model
def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):

    keys_init(config["configurable"]["NVIDIA_API_KEY"], config["configurable"]["TAVILY_API_KEY"])

    temperature = config["configurable"]["temperature"]
    top_p = config["configurable"]["top_p"]

    assistant_model = ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct",
        temperature=temperature,
        top_p=top_p,
        )
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Retrieve memory from the store
    namespace = ("memories", user_id)
    memories = store.search(namespace)

    # Format the memories for the system prompt
    info = "\n".join(f"- {mem.value['content']}" for mem in memories)
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=info)

    # Respond using memory as well as the chat history
    response = assistant_model.invoke([SystemMessage(content=system_msg)]+state["messages"])

    return {"messages": response}

# ---------------------------------------------------------------------
# Conditional Router
# ---------------------------------------------------------------------
# Decides whether to trigger web search or move directly to memory writing.
# The decision is based on a marker string inside the last model response.
def route_messages(state: MessagesState):
    last_msg = state["messages"][-1].content.strip()

    if "NEED_WEB_SEARCH" in last_msg:
        return "generate_search_query"

    return 'write_memory'


# ---------------------------------------------------------------------
# Search Query Generator
# ---------------------------------------------------------------------
# Reformulates the user's question into a search-engine-friendly query.
def generate_search_query(state: MessagesState,config: RunnableConfig):

    temperature = config["configurable"]["temperature"]
    top_p = config["configurable"]["top_p"]

    search_model = ChatNVIDIA(
        model="openai/gpt-oss-20b",
        temperature=temperature,
        top_p=top_p ,
        )

    # Search query
    search_query = search_model.invoke([search_instructions] + state["messages"]).content
    
    return {'search_query': search_query}

# ---------------------------------------------------------------------
# Web Search (Tavily)
# ---------------------------------------------------------------------
# Retrieves real-time information from the web.
# Formats results into structured document blocks.
def search_web(state: MessagesState):

    tavily_search = TavilySearch(max_results=3)
    data = tavily_search.invoke({"query": state["search_query"]})

    results = data.get("results", data)

    formatted_docs = []

    for doc in results:
        if isinstance(doc, dict):
            formatted_docs.append(
                f'<Document href="{doc.get("url", "")}">\n{doc.get("content", "")}\n</Document>'
            )
        else:
            formatted_docs.append(
                f"<Document>\n{doc}\n</Document>"
            )

    return {"context": ["\n\n---\n\n".join(formatted_docs)]}

# ---------------------------------------------------------------------
# Wikipedia Search
# ---------------------------------------------------------------------
# Retrieves up to 2 Wikipedia documents based on the search query.
def search_wikipedia(state: MessagesState):

    search_docs = WikipediaLoader(
        query=state['search_query'],
        load_max_docs=2
    ).load()

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}">\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 


# ---------------------------------------------------------------------
# Answer Generator (RAG step)
# ---------------------------------------------------------------------
# Uses retrieved context + chat history to generate final response.
def generate_answer(state: MessagesState,config: RunnableConfig):

    temperature = config["configurable"]["temperature"]
    top_p = config["configurable"]["top_p"]

    answer_generator = ChatNVIDIA(
            model="openai/gpt-oss-20b",
            temperature=temperature,
            top_p=top_p,
            )
    
    context = state["context"]

    answer_instructions = answer_template.format(context=context)    
  
    answer = answer_generator.invoke([SystemMessage(content=answer_instructions)] + state["messages"])

    return {"messages": answer}

# ---------------------------------------------------------------------
# Memory Schema
# ---------------------------------------------------------------------
# Defines the structure of a stored memory entry.

class Memory(BaseModel):
    content: str = Field(description="The main content of the memory. For example: User expressed interest in learning about French.")



# ---------------------------------------------------------------------
# Long-Term Memory Writer
# ---------------------------------------------------------------------
# Uses Trustcall to extract structured memories from chat history.
# Filters out useless or duplicate entries before storing.
def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):

    temperature = config["configurable"]["temperature"]
    top_p = config["configurable"]["top_p"]

    memory_model= ChatNVIDIA(
        model="meta/llama-3.3-70b-instruct",
        temperature=temperature,
        top_p=top_p,
        )

    trustcall_extractor = create_extractor(
            memory_model,
            tools=[Memory],
            tool_choice="Memory",
            enable_inserts=True,
        )
        
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Define the namespace for the memories
    namespace = ("memories", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)
    
    # Format the existing memories for the Trustcall extractor
    tool_name = "Memory"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )
    
    # Merge the chat history and the instruction
    updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION)] + state["messages"]))

    # Invoke the extractor
    result = trustcall_extractor.invoke({"messages": updated_messages, 
                                        "existing": existing_memories})

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):

        memory_dict = r.model_dump(mode="json")
        content = memory_dict.get("content", "").strip().lower()
    
        # 🚫 Skip useless / meta memories
        if (
            not content
            or "no new information" in content
            or "nothing to store" in content
            or "no relevant information" in content
            or "hello" in content
        ):
            continue
        # 🚫 Skip duplicate memories
        if content in existing_items:
            continue
    
        # ✅ Store only factual memory
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            memory_dict,
        )


# ---------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------
builder = StateGraph(MessagesState)
builder.add_node("assistant", call_model)
builder.add_node("generate_search_query", generate_search_query)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("search_web", search_web)
builder.add_node("generate_answer", generate_answer)
builder.add_node('write_memory',write_memory)

builder.add_edge(START, "assistant")

builder.add_conditional_edges("assistant",route_messages,["generate_search_query",'write_memory'])
builder.add_edge("generate_search_query","search_wikipedia")
builder.add_edge("generate_search_query","search_web")
builder.add_edge(["search_wikipedia","search_web"],"generate_answer")
builder.add_edge("generate_answer",'write_memory')
builder.add_edge('write_memory',END)

# ---------------------------------------------------------------------
# Memory Systems
# ---------------------------------------------------------------------

# Short-term (thread-level) memory
within_thread_memory = MemorySaver()

# Compile graph with:
# - checkpointer (short-term memory)
# - persistent store (long-term semantic memory)
graph = builder.compile(checkpointer=within_thread_memory, store=st.session_state.memory_store)