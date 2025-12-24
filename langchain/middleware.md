# Middleware in LangChain

## What is Middleware?

Middleware provides hooks to control and customize agent execution at each step.

**Capabilities:**
- Monitoring and logging
- Modifying prompts, tools, responses
- Retries, fallbacks, early termination
- Rate limiting, guardrails, PII detection

## Built-in Middleware

### 1. SummarizationMiddleware

Automatic history summarization when token limit exceeded:

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            max_tokens_before_summary=4000,
            messages_to_keep=20
        )
    ]
)
```

### 2. HumanInTheLoopMiddleware

Pause for human approval:

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="openai:gpt-4o",
    tools=[read_email, send_email],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {"allowed_decisions": ["approve", "edit", "reject"]},
                "read_email": False  # Auto-approve
            }
        )
    ]
)
```

### 3. ModelCallLimitMiddleware

Limit model calls:

```python
from langchain.agents.middleware import ModelCallLimitMiddleware

agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=10,  # Max 10 calls per thread
            run_limit=5,  # Max 5 calls per run
            exit_behavior="end"  # Graceful exit
        )
    ]
)
```

### 4. ToolCallLimitMiddleware

Limit tool calls:

```python
from langchain.agents.middleware import ToolCallLimitMiddleware

agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[
        ToolCallLimitMiddleware(thread_limit=20, run_limit=10),  # All tools
        ToolCallLimitMiddleware(tool_name="search", run_limit=3)  # Specific tool
    ]
)
```

### 5. ModelFallbackMiddleware

Fallback to other models on errors:

```python
from langchain.agents.middleware import ModelFallbackMiddleware

agent = create_agent(
    model="openai:gpt-4o",  # Primary
    tools=[...],
    middleware=[
        ModelFallbackMiddleware(
            "openai:gpt-4o-mini",  # First fallback
            "anthropic:claude-3-5-sonnet"  # Second fallback
        )
    ]
)
```

### 6. PIIMiddleware

PII detection and handling:

```python
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block")
    ]
)
```

### 7. ToolRetryMiddleware

Automatic retries for tools:

```python
from langchain.agents.middleware import ToolRetryMiddleware

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool, db_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
            jitter=True
        )
    ]
)
```

## Custom Middleware

### With Decorators

#### @before_model

```python
from langchain.agents.middleware import before_model

@before_model
def log_before_model(state, runtime):
    print(f"About to call model with {len(state['messages'])} messages")
    return None  # No changes
```

#### @after_model

```python
from langchain.agents.middleware import after_model

@after_model(can_jump_to=["end"])
def validate_output(state, runtime):
    last_message = state["messages"][-1]
    if "BLOCKED" in last_message.content:
        return {
            "messages": [AIMessage("I cannot respond to that.")],
            "jump_to": "end"
        }
    return None
```

#### @wrap_model_call

```python
from langchain.agents.middleware import wrap_model_call

@wrap_model_call
def retry_model(request, handler):
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/3 after error: {e}")
```

#### @wrap_tool_call

```python
from langchain.agents.middleware import wrap_tool_call

@wrap_tool_call
def monitor_tool_calls(request, handler):
    print(f"Executing: {request.tool_call['name']}")
    print(f"Args: {request.tool_call['args']}")
    try:
        result = handler(request)
        print("Tool completed successfully")
        return result
    except Exception as e:
        print(f"Tool failed: {e}")
        raise
```

#### @dynamic_prompt

```python
from langchain.agents.middleware import dynamic_prompt

@dynamic_prompt
def personalized_prompt(request):
    user_id = request.runtime.context.get("user_id", "guest")
    return f"You are a helpful assistant for user {user_id}."
```

### Middleware Classes

```python
from langchain.agents.middleware import AgentMiddleware

class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        print(f"Messages: {len(state['messages'])}")
        return None

    def after_model(self, state, runtime):
        print(f"Response: {state['messages'][-1].content}")
        return None

    def wrap_model_call(self, request, handler):
        start = time.time()
        result = handler(request)
        print(f"Model call took {time.time() - start:.2f}s")
        return result
```

## Execution Order

```python
agent = create_agent(
    model="openai:gpt-4o",
    middleware=[mw1, mw2, mw3],
    tools=[...]
)
```

1. `mw1.before_model()` → `mw2.before_model()` → `mw3.before_model()`
2. `mw1.wrap_model_call()` → `mw2.wrap_model_call()` → `mw3.wrap_model_call()` → model
3. `mw3.after_model()` → `mw2.after_model()` → `mw1.after_model()`

## Practical Examples

### Dynamic Model Selection

```python
@wrap_model_call
def select_model(request, handler):
    if len(request.state["messages"]) > 10:
        request.model = advanced_model
    else:
        request.model = basic_model
    return handler(request)
```

### Response Validation

```python
@after_model(can_jump_to=["end"])
def validate_response(state, runtime):
    """Check for blocked words."""
    STOP_WORDS = ["password", "secret"]
    last = state["messages"][-1]
    if any(word in last.content for word in STOP_WORDS):
        return {"messages": [RemoveMessage(id=last.id)]}
    return None
```

### Timing Monitoring

```python
class TimingMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        start = time.time()
        result = handler(request)
        duration = time.time() - start
        if duration > 5.0:
            print(f"⚠️  Slow model call: {duration:.2f}s")
        return result
```

## Best Practices

1. **Use built-in middleware** when possible
2. **Decorators** for simple cases
3. **Classes** for complex logic
4. **Order matters** - critical middleware first
5. **Handle errors** gracefully
6. **Test** middleware separately

## Troubleshooting

### Middleware Not Firing

Check:
- Correct decorator (@before_model, @after_model, etc.)
- Middleware added to list
- Return value (None or dict)

### Middleware Conflicts

```python
# Use correct order
middleware=[
    PIIMiddleware(...),  # Security first
    ModelCallLimitMiddleware(...),  # Then limits
    LoggingMiddleware(...)  # Logging last
]
```
