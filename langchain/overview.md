# LangChain v1.0 - Overview

## What is LangChain?

LangChain is a framework for building agents and LLM-powered applications. It allows you to:
- Connect to OpenAI, Anthropic, Google and other providers in less than 10 lines of code
- Create agents with ready-made architecture and model integrations
- Use LangGraph for complex combinations of deterministic and agentic workflows

## Key Changes in v1.0

- **LangChain v1.0 is available!**
- Full changelog: [release notes](https://python.langchain.com/oss/python/releases/langchain-v1)
- Migration guide: [migration guide](https://python.langchain.com/oss/python/migrate/langchain-v1)

## Installation

### Basic Installation

```bash
# pip
pip install -U langchain

# uv
uv add langchain
```

### With Providers

```bash
# With Anthropic
pip install -U "langchain[anthropic]"

# With OpenAI
pip install -U "langchain[openai]"

# Full bundle
pip install -U langchain langchain-anthropic langchain-openai langchain-community langgraph
```

## Quick Start: Creating an Agent

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# Create agent (less than 10 lines!)
agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

## Key Benefits

### 1. Standardized Model Interface
- Uniform interaction with models from different providers
- Easy to switch providers without vendor lock-in
- Standard request and response format

### 2. Simple and Flexible Agents
- Create an agent in 10 lines of code
- Flexible enough for any customization
- Control over context and prompt engineering

### 3. Built on LangGraph
- Leverages LangGraph capabilities:
  - Long-running execution
  - Human-in-the-loop
  - Persistence
  - Streaming
- No need to know LangGraph for basic usage

### 4. Debugging with LangSmith
- Visualize execution traces
- Capture state transitions
- Detailed timing metrics

## When to Use LangChain vs LangGraph

### Use LangChain when:
- You need to quickly create agents and autonomous applications
- Standard agent architecture is sufficient
- You value simplicity and development speed

### Use LangGraph when:
- You need a combination of deterministic and agentic workflows
- Deep customization is required
- Latency control is important
- You need complex multi-agent systems

## Package Architecture

### langchain
Core package focused on agents:
- `langchain.agents` - agent creation, AgentState
- `langchain.messages` - message types, content blocks, trim_messages
- `langchain.tools` - @tool, BaseTool, injection helpers
- `langchain.chat_models` - init_chat_model, BaseChatModel
- `langchain.embeddings` - init_embeddings, Embeddings

### langchain-classic
Legacy code no longer in the main package:
- Legacy chains (LLMChain, ConversationChain, etc.)
- Retrievers
- Indexing API
- Hub module
- Embeddings modules

## Important Concepts

### Messages
- Uniform format for communicating with models
- Roles: system, user, assistant, tool
- Multimodal content support

### Tools
- Functions that an agent can call
- Automatic schema generation from function signature
- Complex input data support via Pydantic

### State
- Data passed through execution
- Can be extended with custom fields
- Managed through middleware and hooks

### Context
- Immutable configuration data
- Available during runtime
- Useful for user_id, session_id, etc.

## Core Patterns

1. **ReAct (Reasoning + Acting)**
   - Model reasons and selects tools
   - Tools are executed
   - Results are used for next step

2. **Human-in-the-Loop**
   - Pauses for human approval
   - Editing model responses
   - Control over critical operations

3. **Memory Management**
   - Short-term memory (within session)
   - Long-term memory (across sessions)
   - Context window management

## Resources

- Documentation: https://python.langchain.com/
- GitHub: https://github.com/langchain-ai/langchain
- Discord: https://discord.gg/langchain
- LangSmith for debugging: https://smith.langchain.com/
