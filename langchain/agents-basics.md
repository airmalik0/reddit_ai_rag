# Agents in LangChain

## What is an Agent?

An Agent is a system that combines a language model with tools, allowing the model to reason, make decisions, and iteratively work toward solving tasks.

**Key idea:** An agent works in a loop until a stopping condition is reached (final answer or iteration limit).

## Agent Architecture

```
┌─────────────────────────────────────────────┐
│               AGENT LOOP                     │
│                                             │
│  ┌──────┐    ┌───────┐    ┌──────────┐    │
│  │ User │ -> │ Model │ -> │   Tools  │    │
│  │Input │    │  LLM  │    │Execution │    │
│  └──────┘    └───────┘    └──────────┘    │
│                  │              │           │
│                  └──────────────┘           │
│                  (iterate until done)       │
└─────────────────────────────────────────────┘
```

## Creating a Basic Agent

```python
from langchain.agents import create_agent

# Define a tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# Create agent
agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

## Core Components

### 1. Model

The reasoning engine of the agent. Selects tools and generates responses.

#### Static Model

```python
# String identifier
agent = create_agent(
    model="openai:gpt-4o",
    tools=tools
)

# Or model instance with parameters
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    max_tokens=1000
)
agent = create_agent(model, tools=tools)
```

#### Dynamic Model

Choose model based on state/context:

```python
from langchain.agents.middleware import wrap_model_call
from langchain_openai import ChatOpenAI

basic_model = ChatOpenAI(model="gpt-4o-mini")
advanced_model = ChatOpenAI(model="gpt-4o")

@wrap_model_call
def dynamic_model_selection(request, handler):
    """Select model based on conversation length."""
    if len(request.state["messages"]) > 10:
        request.model = advanced_model
    else:
        request.model = basic_model

    return handler(request)

agent = create_agent(
    model=basic_model,
    tools=tools,
    middleware=[dynamic_model_selection]
)
```

### 2. Tools

Functions that the agent can call.

```python
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search, get_weather]
)
```

Empty tools list = LLM-only agent without tool calling.

### 3. System Prompt

Instructions for the agent:

```python
agent = create_agent(
    model="openai:gpt-4o",
    tools=tools,
    system_prompt="You are a helpful assistant. Be concise and accurate."
)
```

#### Dynamic Prompt

Based on runtime context:

```python
from langchain.agents.middleware import dynamic_prompt
from typing import TypedDict

class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request):
    """Generate prompt based on user role."""
    user_role = request.runtime.context.get("user_role", "user")
    base = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base} Explain concepts simply and avoid jargon."

    return base

agent = create_agent(
    model="openai:gpt-4o",
    tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context
)

# Invoke with context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain machine learning"}]},
    context={"user_role": "expert"}
)
```

## Invoking the Agent

```python
# Basic invocation
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in SF?"}]}
)

# With streaming
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Search for AI news"}]},
    stream_mode="values"
):
    print(chunk["messages"][-1].content)
```

## ReAct Pattern (Reasoning + Acting)

Agents follow the ReAct pattern:

```
1. User: "Find weather in SF and check if it's good for hiking"

2. Agent thinks: "Need to get weather first"
   -> Calls: get_weather("San Francisco")

3. Result: "Sunny, 72°F"

4. Agent thinks: "Weather is good, but need to check conditions"
   -> Calls: check_hiking_conditions("San Francisco")

5. Result: "Excellent hiking conditions"

6. Agent responds: "The weather in SF is perfect for hiking today!"
```

## Advanced Concepts

### Structured Output

Force agent to return data in a specific format:

```python
from pydantic import BaseModel, Field
from langchain.agents.structured_output import ToolStrategy

class ContactInfo(BaseModel):
    name: str = Field(description="Person's name")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract: John Doe, john@example.com, (555) 123-4567"}]
})

# result["structured_response"] contains ContactInfo
print(result["structured_response"])
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

### Memory

Agents automatically maintain message history. For long-term memory:

```python
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver

# Custom state
class CustomState(AgentState):
    user_id: str
    preferences: dict

agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_user_info],
    state_schema=CustomState,
    checkpointer=InMemorySaver()  # For persistence
)

# Use with custom state
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",
        "preferences": {"theme": "dark"}
    },
    config={"configurable": {"thread_id": "1"}}
)
```

## Best Practices

1. **Clear prompts**: Describe agent tasks clearly and specifically
2. **Right tools**: Provide relevant tools with good docstrings
3. **Manage context**: Use middleware to manage history length
4. **Error handling**: Add error handling in tools
5. **Testing**: Test agent on various scenarios

## Common Patterns

### Model Call Limit

```python
from langchain.agents.middleware import ModelCallLimitMiddleware

agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[
        ModelCallLimitMiddleware(
            run_limit=5,  # Max 5 calls per invoke
            exit_behavior="end"  # Graceful exit
        )
    ]
)
```

### Human-in-the-Loop

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="openai:gpt-4o",
    tools=[send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                }
            }
        )
    ]
)
```

## Next Steps

- Explore Tools for creating custom tools
- Learn about Middleware for customizing behavior
- Read about Memory for managing state
- See Streaming for real-time updates
