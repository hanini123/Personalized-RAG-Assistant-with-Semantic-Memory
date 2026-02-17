import streamlit as st
import os
import requests
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv() 
os.getenv("NVIDIA_API_KEY")
os.getenv("TAVILY_API_KEY")
# https://personalized-rag-assistant-with-semantic-memory.streamlit.app/
# Initialize session state
if "NVIDIA_API_KEY" in os.environ and "TAVILY_API_KEY" in os.environ:
    st.session_state.NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
    st.session_state.TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", 
                                      "content":"👋 Welcome! I can personalize responses and search the web in real time.\n\nHow can I help you today?"})

if "thread_id" not in st.session_state:
    st.session_state.thread_id = 1

if "user_id" not in st.session_state:
    st.session_state.user_id = "1"

if "memory_store" not in st.session_state:
    st.session_state.memory_store = InMemoryStore()

from graph import graph

def new_chat():
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant",
                                       "content":"👋 Welcome! I can personalize responses and search the web in real time.\n\nHow can I help you today?"})
    st.session_state.thread_id += st.session_state.thread_id

@st.dialog("🧠 Stored Memories")
def memory_dialog():

    user_id = st.session_state.user_id
    namespace = ("memories", user_id)
    store = st.session_state.memory_store

    memories = store.search(namespace)

    if not memories:
        st.info("No memories stored yet.")
        return

    for mem in memories:
        col1, col2 = st.columns([5, 1])

        with col1:
            st.markdown(f"• {mem.value.get('content', '')}")

        with col2:
            if st.button("🗑️", key=f"delete_{mem.key}"):
                store.delete(namespace, mem.key)
                st.rerun()

    st.divider()

    if st.button("🔥 Clear all memories"):
        for mem in memories:
            store.delete(namespace, mem.key)
        st.success("All memories cleared.")
        st.rerun()


def reset_keys():
    st.session_state.pop("NVIDIA_API_KEY")
    st.session_state.pop("TAVILY_API_KEY")
    del os.environ["NVIDIA_API_KEY"]
    del os.environ["TAVILY_API_KEY"]

with st.sidebar:
    missing_keys = "NVIDIA_API_KEY" not in st.session_state and "TAVILY_API_KEY" not in st.session_state

    if missing_keys:
        st.error("⚠️ This demo supports advanced reasoning, web search, and long-term memory.\n\n"
        "To unlock the full experience, please provide your own API keys:\n\n"
        "🔹 **NVIDIA API Key** (for LLM reasoning):\n"
        "https://build.nvidia.com/models\n\n"
        "🔹 **Tavily API Key** (for web search):\n"
        "https://www.tavily.com/\n\n")

        NVIDIA_KEY = st.text_input(
            "NVIDIA_API_KEY",
            type="password"
        )
        TAVILY_KEY = st.text_input(
            "TAVILY_API_KEY",
            type="password"
        )
        if NVIDIA_KEY and TAVILY_KEY:
            # strip spaces and surrounding quotes
            NVIDIA_KEY = NVIDIA_KEY.strip()
            TAVILY_KEY = TAVILY_KEY.strip()

            if (NVIDIA_KEY.startswith('"') and NVIDIA_KEY.endswith('"')) or (NVIDIA_KEY.startswith("'") and NVIDIA_KEY.endswith("'")):
                NVIDIA_KEY = NVIDIA_KEY[1:-1]

            if (TAVILY_KEY.startswith('"') and TAVILY_KEY.endswith('"')) or (TAVILY_KEY.startswith("'") and TAVILY_KEY.endswith("'")):
                TAVILY_KEY = TAVILY_KEY[1:-1]

            st.session_state.NVIDIA_API_KEY = NVIDIA_KEY
            st.session_state.TAVILY_API_KEY = TAVILY_KEY
            os.environ["NVIDIA_API_KEY"] = st.session_state.NVIDIA_API_KEY
            os.environ["TAVILY_API_KEY"] = st.session_state.TAVILY_API_KEY

            st.success("✅ Keys loaded successfully")
            st.rerun()  # 🔥 important
    else:
        st.success("🔐 API keys loaded")
        st.divider()
        
        st.button("Create New Chat", on_click=new_chat)
        st.button("Display Your Memory", on_click=memory_dialog)
        st.button('Reset Your Keys', on_click=reset_keys)

    st.divider()
    st.session_state.temperature = st.slider(
            "temperature", 0.0, 1.0, 0.2, 0.05
        )
    st.session_state.top_p = st.slider(
            "top_p", 0.0, 1.0, 0.7, 0.05
        )


if "NVIDIA_API_KEY" in st.session_state and "TAVILY_API_KEY" in st.session_state:
    config = {
            "configurable": {
                "thread_id": st.session_state.thread_id,
                "user_id": st.session_state.user_id,
                "temperature": st.session_state.temperature,
                "top_p": st.session_state.top_p,
                "NVIDIA_API_KEY": st.session_state.NVIDIA_API_KEY,
                "TAVILY_API_KEY": st.session_state.TAVILY_API_KEY
            }
        }

# Display all previous messages (from earlier runs)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box
if prompt := st.chat_input("What is up?"):
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    if "NVIDIA_API_KEY" not in st.session_state and "TAVILY_API_KEY" not in st.session_state:
        ai_msg = (
            "👋 Hi! I'm an assistant designed to help you.\n\n"
        "This demo supports advanced reasoning, web search, and long-term memory.\n\n"
        "To unlock the full experience, please provide your own API keys:\n\n"
        "🔹 **NVIDIA API Key** (for LLM reasoning):\n"
        "https://build.nvidia.com/models\n\n"
        "🔹 **Tavily API Key** (for web search):\n"
        "https://www.tavily.com/\n\n"
        "👉 Once the keys are added in the sidebar, the assistant will switch to full mode."
        )
    else:
        try:
            result = graph.invoke(
                {"messages": HumanMessage(content=prompt)},
                config
            )

            ai_msg = result["messages"][-1].content

        except requests.exceptions.ConnectionError:
            ai_msg = (
                "⚠️ Connection issue detected.\n\n"
                "I couldn’t reach the AI service right now. This usually happens due to:\n"
                "- Temporary internet issues\n"
                "- NVIDIA service being unavailable\n\n"
                "👉 Please try again in a moment."
            )

        except requests.exceptions.Timeout:
            ai_msg = (
                "⏳ The request took too long to respond.\n\n"
                "This may be caused by a slow network or a busy AI service.\n"
                "👉 Please retry in a few seconds."
            )

        except PermissionError:
            ai_msg = (
                "🔑 API key error.\n\n"
                "Your NVIDIA or Tavily API key appears to be invalid or unauthorized.\n\n"
                "👉 Please double-check your keys in the sidebar:\n"
                "- https://build.nvidia.com/models\n"
                "- https://www.tavily.com/"
            )

        except Exception as e:
            ai_msg = (
                "❌ Something unexpected happened.\n\n"
                "This demo couldn’t process your request right now.\n"
                "👉 "
                f"{e}"
            )



    # Display AI response immediately
    with st.chat_message("assistant"):
        st.markdown(ai_msg)

    # Store AI response
    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
