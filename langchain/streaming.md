# Streaming in LangChain

## What is Streaming?

Real-time data streaming instead of waiting for complete response.

**Benefits:**
- Improved UX (user sees progress)
- Fast feedback
- Less perceived latency

## Streaming Types

### 1. Agent Progress (stream_mode="updates")

Get updates after each agent step:

```python
agent = create_agent(
    model="openai:gpt-5-nano",
    tools=[get_weather]
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="updates"
):
    for step, data in chunk.items():
        print(f"Step: {step}")
        print(f"Content: {data['messages'][-1].content_blocks}")
```

Output:
```
Step: model
Content: [{'type': 'tool_call', 'name': 'get_weather', 'args': {'city': 'SF'}}]

Step: tools
Content: [{'type': 'text', 'text': "It's sunny in SF!"}]

Step: model
Content: [{'type': 'text', 'text': "It's sunny in SF!"}]
```

### 2. LLM Tokens (stream_mode="messages")

Get tokens as they're generated:

```python
for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="messages"
):
    print(f"Node: {metadata['langgraph_node']}")
    print(f"Content: {token.content_blocks}")
```

Output:
```
Node: model
Content: [{'type': 'text', 'text': 'The'}]

Node: model
Content: [{'type': 'text', 'text': ' weather'}]

Node: model
Content: [{'type': 'text', 'text': ' in'}]
...
```

### 3. Custom Updates (stream_mode="custom")

Stream custom data from tools:

```python
from langgraph.config import get_stream_writer

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's sunny in {city}!"

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[get_weather]
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="custom"
):
    print(chunk)
```

Output:
```
Looking up data for city: San Francisco
Acquired data for city: San Francisco
```

## Multiple Modes

```python
for stream_mode, chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode=["updates", "custom"]
):
    print(f"Mode: {stream_mode}")
    print(f"Content: {chunk}")
```

## Practical Examples

### Streaming Text to UI

```python
def stream_to_ui(user_message: str):
    """Stream tokens to user interface."""
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        stream_mode="messages"
    ):
        # Send token to UI
        yield chunk.text
```

### Execution Progress

```python
def stream_progress(user_message: str):
    """Show agent progress."""
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        stream_mode="updates"
    ):
        for step, data in chunk.items():
            if step == "model":
                print("🤖 Thinking...")
            elif step == "tools":
                print("🔧 Using tools...")
```

### Custom Updates from Tools

```python
@tool
def long_running_task(task_id: str) -> str:
    """Execute long-running task with progress updates."""
    writer = get_stream_writer()

    writer("Starting task...")
    time.sleep(1)

    writer("Processing step 1/3...")
    time.sleep(1)

    writer("Processing step 2/3...")
    time.sleep(1)

    writer("Processing step 3/3...")
    time.sleep(1)

    return "Task completed!"

# Usage
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Run task 123"}]},
    stream_mode="custom"
):
    print(f"Progress: {chunk}")
```

## Disabling Streaming

For individual models:

```python
model = init_chat_model(
    "openai:gpt-4o",
    streaming=False  # Disable streaming
)
```

## Best Practices

1. **Use stream_mode="messages"** for UI with tokens
2. **Use stream_mode="updates"** for logging steps
3. **Use stream_mode="custom"** for progress indicators
4. **Combine modes** for full control
5. **Handle errors** in streaming callbacks

## Troubleshooting

### Streaming Not Working

Check that:
- Model supports streaming
- `streaming=True` (default)
- Using correct stream_mode

### Slow Streaming

```python
# Use faster model
model = init_chat_model("openai:gpt-4o-mini")
```

### Missing Updates

```python
# Process all chunks immediately
for chunk in agent.stream(...):
    process_immediately(chunk)  # Don't accumulate
```
