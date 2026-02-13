# Personalized RAG Assistant with Semantic Memory

## Overview

This project implements a **Personalized Retrieval-Augmented Generation (RAG) Assistant** with semantic memory and graph-based orchestration.

The system is built as a structured multi-model workflow rather than a single LLM call. It follows a branching execution logic:

1. A primary assistant model first observes the user request.
2. If the request can be answered using its internal knowledge, it responds directly.
3. If external information is required, the workflow triggers a secondary LLM responsible for generating a well-structured search query.
4. The system retrieves relevant information from the web.
5. Another model generates a response grounded strictly in the retrieved context.
6. Regardless of the branch taken, the interaction is passed to a memory extraction model that identifies factual information about the user.
7. Extracted facts are stored as semantic memory to personalize future responses.

This design separates:
- Decision-making
- Query generation
- Retrieval
- Grounded response generation
- Memory extraction

The entire workflow is orchestrated using a graph-based execution model.

---

## Architecture Flow

### Step 1 — Request Observation
The main assistant model evaluates the user query and decides:
- Direct answer branch (internal knowledge)
- Retrieval branch (requires external information)

### Step 2 — Retrieval Branch (if triggered)

- A dedicated LLM generates a structured search query.
- The system retrieves relevant information from web sources.
- A response-generation model produces an answer grounded in the retrieved context.

### Step 3 — Memory Extraction (Both Branches)

After generating the response:
- A separate model extracts factual user information from the interaction.
- Extracted facts are stored as semantic memory.
- Memory is later injected into future prompts to enable personalization.

---

## Key Characteristics

- Multi-model architecture
- Conditional execution flow
- Graph-based orchestration (LangGraph)
- Retrieval-augmented reasoning
- Post-response semantic memory extraction
- Personalization through stored factual memory

---

## Tech Stack

- Python
- Streamlit
- LangChain
- LangGraph
- LangChain NVIDIA AI Endpoints
- LangChain Tavily 
- Wikipedia
- Trustcall

---

## Project Structure

Personalized-RAG-Assistant-with-Semantic-Memory/
│
├── app.py # Streamlit interface and application entry point
├── graph.py # Graph-based workflow definition and orchestration logic
├── prompts.py # Prompt templates for different models (assistant, query generator, memory extractor,etc.)
├── requirements.txt
└──README.md

---

## Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/hanini123/Personalized-RAG-Assistant-with-Semantic-Memory.git
cd Personalized-RAG-Assistant-with-Semantic-Memory
```
### 2️⃣ Create a Virtual Environment
```bash
python -m venv .venv
```
#### Windows
```bash
.venv\Scripts\activate
```
#### Mac/Linux
```bash
source .venv/bin/activate

```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt

```
### 4️⃣ Configure Environment Variables

You can either:
- Create a `.env` file in the root directory and add your API keys:

```bash
NVIDIA_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```
or
- Provide the API keys directly through the graphical user interface when running the application.
### Running the Application
```bash
streamlit run app.py
```
---

## Author

**Mbarek Hanini**  
PhD Researcher | Data Scientist  
AI Multi-Agent Systems | Workflow Automation | RAG and GraphRAG Architectures