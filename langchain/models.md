# Models in LangChain

## What is a Model?

A Model (LLM) is the reasoning engine that interprets text, generates responses, calls tools, and makes decisions.

**Key capabilities:**
- Text generation
- Tool calling (function calling)
- Structured output
- Multimodal (working with images, audio)
- Reasoning (multi-step reasoning)

## Model Initialization

### init_chat_model (recommended)

```python
from langchain.chat_models import init_chat_model

# Simple initialization
model = init_chat_model("openai:gpt-4.1")
model = init_chat_model("anthropic:claude-sonnet-4-5")

# With parameters
model = init_chat_model(
    "anthropic:claude-sonnet-4-5",
    temperature=0.7,
    timeout=30,
    max_tokens=1000
)
```

### Model Class (for fine-tuning)

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# OpenAI
model = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.1,
    max_tokens=1000
)

# Anthropic
model = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0.7,
    max_tokens=2000
)
```

## Core Parameters

```python
model = init_chat_model(
    model="anthropic:claude-sonnet-4-5",  # Model name
    api_key="...",                         # API key (or via env var)
    temperature=0.7,                       # Creativity (0-1)
    timeout=30,                            # Request timeout (sec)
    max_tokens=1000,                       # Max tokens in response
    max_retries=3                          # Retry attempts on error
)
```

## Calling the Model

### invoke() - Full Response
```python
# Single message
response = model.invoke("Why do parrots talk?")
print(response.content)

# Message history
conversation = [
    {"role": "system", "content": "You are a helpful translator."},
    {"role": "user", "content": "Translate: I love programming."},
    {"role": "assistant", "content": "J'adore la programmation."},
    {"role": "user", "content": "Translate: I love AI."}
]
response = model.invoke(conversation)
print(response.content)  # "J'adore l'IA."
```

### stream() - Streaming Output
```python
# Token streaming
for chunk in model.stream("Why do parrots talk?"):
    print(chunk.text, end="|", flush=True)
# Output: "Par|rots| can| talk|..."

# Streaming tool calls and reasoning
for chunk in model.stream("What color is the sky?"):
    for block in chunk.content_blocks:
        if block["type"] == "reasoning":
            print(f"Reasoning: {block['reasoning']}")
        elif block["type"] == "text":
            print(block["text"])
```

### batch() - Parallel Processing
```python
responses = model.batch([
    "Why do parrots talk?",
    "How do airplanes fly?",
    "What is quantum computing?"
])

for response in responses:
    print(response.content)
```

## Tool Calling

Models can call functions/tools:

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get weather at a location."""
    return f"It's sunny in {location}."

# Bind tool to model
model_with_tools = model.bind_tools([get_weather])

# Model can call the tool
response = model_with_tools.invoke("What's the weather in Boston?")

# Check tool calls
for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
# Output:
# Tool: get_weather
# Args: {'location': 'Boston'}
```

### Parallel Tool Calls
```python
response = model_with_tools.invoke(
    "What's the weather in Boston and Tokyo?"
)

print(response.tool_calls)
# [
#   {'name': 'get_weather', 'args': {'location': 'Boston'}, 'id': 'call_1'},
#   {'name': 'get_weather', 'args': {'location': 'Tokyo'}, 'id': 'call_2'},
# ]
```

### Forced Tool Calling
```python
# Force use of any tool
model_with_tools = model.bind_tools([tool1], tool_choice="any")

# Force use of specific tool
model_with_tools = model.bind_tools([tool1], tool_choice="tool_1")
```

## Structured Output

Model returns data in a specific format:

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(description="Movie title")
    year: int = Field(description="Release year")
    director: str = Field(description="Director name")
    rating: float = Field(description="Rating out of 10")

# Bind schema
model_with_structure = model.with_structured_output(Movie)

# Get structured response
response = model_with_structure.invoke(
    "Give me details about Inception"
)

print(response)
# Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)
```

## Main Providers

### OpenAI
```bash
pip install -U "langchain[openai]"
```
```python
model = init_chat_model("openai:gpt-4.1")
# Models: gpt-4.1, gpt-4.1-mini, gpt-5, gpt-5-nano
```

### Anthropic (Claude)
```bash
pip install -U "langchain[anthropic]"
```
```python
model = init_chat_model("anthropic:claude-sonnet-4-5")
# Models: claude-sonnet-4-5, claude-opus-4, claude-haiku-3-5
```

### Google (Gemini)
```bash
pip install -U "langchain[google-genai]"
```
```python
model = init_chat_model("google_genai:gemini-2.5-flash-lite")
```

### Azure OpenAI
```bash
pip install -U "langchain[openai]"
```
```python
import os
os.environ["AZURE_OPENAI_API_KEY"] = "..."
os.environ["AZURE_OPENAI_ENDPOINT"] = "..."

model = init_chat_model(
    "azure_openai:gpt-4.1",
    azure_deployment="your-deployment-name"
)
```

## Advanced Features

### Multimodal (images, audio)
```python
from langchain.messages import HumanMessage

message = HumanMessage(content_blocks=[
    {"type": "text", "text": "Describe this image."},
    {"type": "image", "url": "https://example.com/image.jpg"},
])

response = model.invoke([message])
```

### Reasoning
```python
# Streaming reasoning process
for chunk in model.stream("Why do parrots have colorful feathers?"):
    reasoning_steps = [r for r in chunk.content_blocks if r["type"] == "reasoning"]
    if reasoning_steps:
        print(reasoning_steps)
```

### Prompt Caching
```python
# Anthropic - automatic caching
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4-latest"),
    system_prompt=LONG_PROMPT,  # Will be cached
    middleware=[AnthropicPromptCachingMiddleware(ttl="5m")]
)
```

### Rate Limiting
```python
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # 1 request every 10 sec
    max_bucket_size=10
)

model = init_chat_model(
    "openai:gpt-4o",
    rate_limiter=rate_limiter
)
```

### Token Usage
```python
response = model.invoke("Hello!")

# Token metadata
usage = response.usage_metadata
print(f"Input tokens: {usage['input_tokens']}")
print(f"Output tokens: {usage['output_tokens']}")
print(f"Total tokens: {usage['total_tokens']}")
```

### Configurable Models
```python
# Create model that can be reconfigured at runtime
configurable_model = init_chat_model(temperature=0)

# Use with different models
configurable_model.invoke(
    "what's your name",
    config={"configurable": {"model": "gpt-5-nano"}}
)

configurable_model.invoke(
    "what's your name",
    config={"configurable": {"model": "claude-sonnet-4-5"}}
)
```

## Best Practices

1. **Use init_chat_model** for simplicity and flexibility
2. **Manage temperature**: low (0.0-0.3) for precision, high (0.7-1.0) for creativity
3. **Set max_tokens** to control response length
4. **Use timeout** to prevent hanging
5. **Handle errors** with try-except
6. **Cache prompts** to save tokens and time
7. **Monitor token usage** to control costs

## Model Comparison

### OpenAI GPT
- Excellent performance
- Good tool calling support
- Fast responses
- Reasoning in new models (gpt-5)

### Anthropic Claude
- Large context windows (200k tokens)
- Excellent long context
- Prompt caching
- Strong reasoning capabilities

### Google Gemini
- Multimodal out of the box
- Good performance/price ratio
- Built-in tools (search, code execution)

## Troubleshooting

### Rate Limit Errors
```python
# Use InMemoryRateLimiter
rate_limiter = InMemoryRateLimiter(requests_per_second=0.5)
model = init_chat_model("openai:gpt-4o", rate_limiter=rate_limiter)
```

### Timeout Errors
```python
# Increase timeout
model = init_chat_model("openai:gpt-4o", timeout=60)
```

### Context Length Errors
```python
# Use middleware to manage length
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="openai:gpt-4o",
    middleware=[SummarizationMiddleware(max_tokens_before_summary=4000)]
)
```
