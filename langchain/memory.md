# Memory in LangChain

## Short-term Memory

Memory within a single session/thread. Automatically managed through agent state.

### Adding Memory to Agent

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    "openai:gpt-5",
    tools=[get_user_info],
    checkpointer=InMemorySaver()  # Enable memory
)

# First call
agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    {"configurable": {"thread_id": "1"}}
)

# Second call - agent remembers the name
agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    {"configurable": {"thread_id": "1"}}
)
# Answer: "Your name is Bob."
```

### Production - PostgreSQL

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:pass@localhost:5432/db"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    agent = create_agent(
        "openai:gpt-5",
        tools=[get_user_info],
        checkpointer=checkpointer
    )
```

## Custom State

```python
from langchain.agents import AgentState

class CustomState(AgentState):
    user_id: str
    preferences: dict

agent = create_agent(
    "openai:gpt-5",
    tools=[],
    state_schema=CustomState,
    checkpointer=InMemorySaver()
)

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",
        "preferences": {"theme": "dark"}
    },
    {"configurable": {"thread_id": "1"}}
)
```

## Managing History Length

### 1. Trim Messages

```python
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

@before_model
def trim_messages(state, runtime):
    """Keep only the last N messages."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None

    first_msg = messages[0]  # System message
    recent = messages[-3:]
    new_messages = [first_msg] + recent

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

agent = create_agent(
    model,
    tools=tools,
    middleware=[trim_messages],
    checkpointer=InMemorySaver()
)
```

### 2. Delete Messages

```python
@before_model
def delete_old_messages(state, runtime):
    """Delete old messages."""
    messages = state["messages"]
    if len(messages) > 2:
        # Delete first 2 messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None
```

### 3. Summarization

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            max_tokens_before_summary=4000,  # Threshold for summarization
            messages_to_keep=20  # How many recent messages to keep
        )
    ],
    checkpointer=InMemorySaver()
)
```

## Accessing Memory from Tools

```python
from langchain.tools import tool, ToolRuntime

@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """Look up user info from state."""
    user_id = runtime.state["user_id"]
    return f"User ID: {user_id}"

@tool
def greet(runtime: ToolRuntime) -> str:
    """Greet the user by name."""
    user_name = runtime.state.get("user_name", "Guest")
    return f"Hello {user_name}!"
```

## Long-term Memory

Memory across sessions via Store:

```python
from langgraph.store.memory import InMemoryStore

@tool
def save_preference(key: str, value: str, runtime: ToolRuntime) -> str:
    """Save user preference."""
    store = runtime.store
    user_id = runtime.context["user_id"]
    store.put(("users", user_id), key, value)
    return f"Saved {key}={value}"

@tool
def get_preference(key: str, runtime: ToolRuntime) -> str:
    """Get user preference."""
    store = runtime.store
    user_id = runtime.context["user_id"]
    value = store.get(("users", user_id), key)
    return value or "Not set"

store = InMemoryStore()
agent = create_agent(
    model,
    tools=[save_preference, get_preference],
    store=store
)
```

## Best Practices

1. **Use checkpointer** for persistence
2. **Manage history length** with middleware
3. **Summarize** instead of deleting to preserve context
4. **Custom state** for additional data
5. **Store** for long-term memory across sessions

## Troubleshooting

### Context Overflow

```python
# Use SummarizationMiddleware
middleware=[SummarizationMiddleware(max_tokens_before_summary=4000)]
```

### Losing Important Information

```python
# Don't delete the first (system) message
first_msg = messages[0]
new_messages = [first_msg] + recent_messages
```

### Slow Performance with Large History

```python
# Trim history regularly
@before_model
def trim_history(state, runtime):
    if len(state["messages"]) > 50:
        # Keep first + last 20
        messages = [state["messages"][0]] + state["messages"][-20:]
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]}
```
