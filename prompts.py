
from langchain_core.messages import SystemMessage
# ---------------------------------------------------------------------
# Main Assistant System Prompt
# ---------------------------------------------------------------------
# This prompt defines:
# - The assistant's role and personality
# - The existence of long-term memory
# - The routing rule for triggering web search
#
# The model will:
# - Answer normally if it can
# - Return ONLY "NEED_WEB_SEARCH" if external info is required
# ---------------------------------------------------------------------
MODEL_SYSTEM_MESSAGE = """You are a helpful chatbot. You are designed to be a companion to a user. 

You have a long term memory which keeps track of information about the user over time to personalize your response.

Current long term memory (it may be empty): 
{memory}

Your task:
- If you can fully answer the user using your own knowledge, answer normally.
- If the user requires real-time, factual, or external information, respond ONLY with:
  NEED_WEB_SEARCH
Do not call any tools.
"""
# ---------------------------------------------------------------------
# Search Query Reformulation Instructions
# ---------------------------------------------------------------------
# Used in the "generate_search_query" node.
#
# Purpose:
# - Analyze the full conversation 
# - Focus especially on the latest user question
# - Prepare query suitable for web search or retrieval
#
# Output:
# - A single well-formed search query string
# ---------------------------------------------------------------------
search_instructions = SystemMessage(content=f"""You will be given a conversation between an AI assistant and a user. 

Your goal is to generate a well-structured query for use in retrieval and/or web search related to the conversation.
        
First, analyze the full conversation.

Pay particular attention to the final question posed by the user.

Convert this final question into a well-structured web search query.
""")

# ---------------------------------------------------------------------
# RAG Answer Generation Template
# ---------------------------------------------------------------------
# Used in the "generate_answer" node.
#
# Purpose:
# - Inject retrieved web context into the system prompt
# - Force the model to rely on retrieved data
# - Generate a clear and relevant final answer
#
# {context} will be dynamically filled with:
# - Wikipedia results
# - Tavily web search results
# ---------------------------------------------------------------------
answer_template = """You will be given a conversation between an AI assistant and a user. 

The assistant cannot answer the final user question using only its general knowledge,
        
so the following context has been extracted from the web:
Context: {context}
        
Your goal is to answer the user's question.
                
First, analyze the full conversation.
        
Pay particular attention to the final question posed by the user.
        
Provide a clear and relevant answer.
"""

# ---------------------------------------------------------------------
# Trustcall Memory Extraction Instruction
# ---------------------------------------------------------------------
# Used in the "write_memory" node.
#
# Purpose:
# - Extract structured long-term memories from conversation
# - Insert new memories
# - Update existing memories
# - Avoid storing useless or trivial data
#
# Important constraints:
# - No greetings
# - No small talk
# - No assumptions
# - Only factual user-stated information
# ---------------------------------------------------------------------
TRUSTCALL_INSTRUCTION = """
You are collecting information about the user to personalize your responses.
User memory is a collection of memories, so update existing ones or insert new ones based on the following instructions.

Reflect on the following interaction.

Do NOT store trivial or generic interactions such as greetings, politeness, small talk, or acknowledgements (e.g., "hello", "hi", "thanks").

If the information already exists in memory, skip it or update it if needed.

Use the provided tools to store any relevant memories about the user.

Use parallel tool calling to handle updates and insertions simultaneously.

Remember: Only include factual information directly stated by the user. Do not make assumptions or inferences.

Based on the chat history below, please update the user information:
"""