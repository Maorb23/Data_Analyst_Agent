# Data Analyst Agent

A modern, agentic data analysis platform powered by LLMs, LangGraph, and Streamlit.

## Overview

**Data_Analyst_Agent** is an interactive, agent-based data analysis tool that leverages large language models (LLMs), LangGraph for agentic workflows, and Streamlit for a beautiful, user-friendly UI. It is designed to:
- Answer both structured (statistical, quantitative) and unstructured (textual, qualitative) data questions.
- Integrate a suite of analysis tools (statistical tests, summarization, pattern analysis, etc.).
- Support follow-up queries and persistent user preferences via memory.
- Visualize agent flows and results interactively.

## Features
- **Agentic Data Analysis**: Uses LangGraph to route queries to the right agent (structured/unstructured/refusal).
- **LLM-Powered**: Uses OpenAI-compatible LLMs for reasoning, summarization, and tool orchestration.
- **Rich Toolset**: Includes tools for statistical tests, word analysis, pattern analysis, summarization, and more.
- **Streamlit UI**: Modern, interactive web interface for chat, results, plots, and debugging.
- **Memory & Context**: Remembers user preferences and conversation history for follow-up queries.
- **Graph Visualization**: Visualizes the agent workflow using Graphviz in the UI.
- **Debugging & Logging**: Advanced debug logs, tool call tracking, and agent iteration inspection.

## Architecture
- **Streamlit**: Frontend UI and app runner.
- **LangGraph**: Agent workflow and routing (classifier, structured, unstructured, refusal, summary nodes).
- **LangChain Agents**: Tool-calling agents for both structured and unstructured queries.
- **Tools**: Modular Python functions for analytics, statistics, summarization, and more.
- **MemoryStore**: SQLite-backed persistent memory for user preferences and context.

## Setup

### Prerequisites
- Python 3.9+
- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [OpenAI-compatible LLM API key](https://platform.openai.com/)
- (Optional) [Graphviz](https://graphviz.gitlab.io/download/) for PNG export

### Installation
1. Clone the repo:
   ```sh
   git clone https://github.com/Maorb23/Data_Analyst_Agent.git
   cd Data_Analyst_Agent
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Add your LLM API key to `Nebius_api_key.txt` (or set as env var).

### Running the App
```sh
streamlit run Data_Analyst/app_lang.py
```

## Usage
- Open the Streamlit UI in your browser.
- Ask questions about the dataset (structured or unstructured).
- View answers, generated plots, and debug info.
- Use the sidebar to inspect memory and session state.
- Visualize the agent workflow graph in the UI.

## Contributing
Pull requests and issues are welcome! To contribute:
1. Fork the repo
2. Create a feature branch
3. Commit your changes
4. Open a pull request

## License
MIT License

## Acknowledgements
- [LangChain](https://python.langchain.com/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Streamlit](https://streamlit.io/)
- [OpenAI](https://openai.com/)
- [Graphviz](https://graphviz.gitlab.io/)

---

For more, see the code and issues at [https://github.com/Maorb23/Data_Analyst_Agent](https://github.com/Maorb23/Data_Analyst_Agent)
