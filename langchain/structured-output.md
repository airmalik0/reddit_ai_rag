# Structured Output in LangChain

## What is Structured Output?

Return data in predictable formats (JSON, Pydantic) instead of parsing natural text.

**Benefits:**
- Reliable data extraction
- Automatic validation
- Easy code integration

## Two Strategies

### 1. ProviderStrategy (Recommended)

Uses provider's native structured output support (OpenAI, Grok):

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy

class ContactInfo(BaseModel):
    """Contact information."""
    name: str = Field(description="Person's name")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")

agent = create_agent(
    model="openai:gpt-5",
    tools=[],
    response_format=ProviderStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract: John, john@ex.com, (555) 123-4567"}]
})

print(result["structured_response"])
# ContactInfo(name='John', email='john@ex.com', phone='(555) 123-4567')
```

### 2. ToolStrategy (Universal)

Uses tool calling for structured output (works with any model):

```python
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model="openai:gpt-5",
    tools=[...],
    response_format=ToolStrategy(ContactInfo)
)
```

## Schema Types

### Pydantic Models (Recommended)

```python
from pydantic import BaseModel, Field
from typing import Literal

class ProductReview(BaseModel):
    """Product review analysis."""
    rating: int | None = Field(description="Rating 1-5", ge=1, le=5)
    sentiment: Literal["positive", "negative"]
    key_points: list[str] = Field(description="Key points, lowercase, 1-3 words each")

agent = create_agent(
    model="openai:gpt-5",
    tools=[],
    response_format=ProductReview  # Auto-selects best strategy
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Review: Great product! 5/5. Fast shipping, expensive."}]
})

print(result["structured_response"])
# ProductReview(rating=5, sentiment='positive', key_points=['fast shipping', 'expensive'])
```

### TypedDict

```python
from typing import Literal
from typing_extensions import TypedDict

class ProductReviewDict(TypedDict):
    """Product review analysis."""
    rating: int | None  # Rating 1-5
    sentiment: Literal["positive", "negative"]
    key_points: list[str]  # Key points

agent = create_agent(
    model="openai:gpt-5",
    tools=[],
    response_format=ProductReviewDict
)
```

### JSON Schema

```python
review_schema = {
    "type": "object",
    "description": "Product review analysis",
    "properties": {
        "rating": {"type": "integer", "minimum": 1, "maximum": 5},
        "sentiment": {"type": "string", "enum": ["positive", "negative"]},
        "key_points": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["sentiment", "key_points"]
}

agent = create_agent(
    model="openai:gpt-5",
    tools=[],
    response_format=review_schema
)
```

### Union Types (Multiple Schemas)

```python
from typing import Union

class ProductReview(BaseModel):
    rating: int
    sentiment: str

class CustomerComplaint(BaseModel):
    issue_type: Literal["product", "service", "shipping"]
    severity: Literal["low", "medium", "high"]

# Model will choose appropriate schema
agent = create_agent(
    model="openai:gpt-5",
    tools=[],
    response_format=ToolStrategy(Union[ProductReview, CustomerComplaint])
)
```

## Customization

### Tool Message Content

```python
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model="openai:gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=ContactInfo,
        tool_message_content="✅ Contact info captured successfully!"
    )
)
```

### Error Handling

```python
# Default: handle_errors=True (automatic retries)
agent = create_agent(
    model="openai:gpt-5",
    tools=[],
    response_format=ToolStrategy(ProductReview)
)

# Custom error message
agent = create_agent(
    model="openai:gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=ProductReview,
        handle_errors="Please provide valid rating 1-5 and sentiment."
    )
)

# Handle only specific errors
agent = create_agent(
    model="openai:gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=ProductReview,
        handle_errors=ValueError  # Only ValueError
    )
)

# Custom error handler
def custom_error_handler(error: Exception) -> str:
    if isinstance(error, ValidationError):
        return "Validation failed. Check your data format."
    return f"Error: {str(error)}"

agent = create_agent(
    model="openai:gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=ProductReview,
        handle_errors=custom_error_handler
    )
)

# No error handling (raise exception)
agent = create_agent(
    model="openai:gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=ProductReview,
        handle_errors=False
    )
)
```

## Practical Examples

### Contact Extraction

```python
class Contact(BaseModel):
    name: str
    email: str
    phone: str | None
    company: str | None

agent = create_agent(
    model="openai:gpt-5",
    tools=[],
    response_format=Contact
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract: John Doe, CTO at Acme Inc, john@acme.com"}]
})
# Contact(name='John Doe', email='john@acme.com', phone=None, company='Acme Inc')
```

### Sentiment Analysis

```python
class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    key_phrases: list[str]

agent = create_agent(
    model="openai:gpt-5",
    tools=[],
    response_format=SentimentAnalysis
)
```

### Event Extraction

```python
class Event(BaseModel):
    title: str
    date: str
    location: str | None
    attendees: list[str]

agent = create_agent(
    model="openai:gpt-5",
    tools=[],
    response_format=Event
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Team meeting on Monday at 2pm in Room 301. Attendees: Alice, Bob, Carol"}]
})
```

### Nested Structures

```python
class Actor(BaseModel):
    name: str
    role: str

class MovieDetails(BaseModel):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]

agent = create_agent(
    model="openai:gpt-5",
    tools=[],
    response_format=MovieDetails
)
```

## Best Practices

1. **Use Pydantic** for validation and types
2. **Add Field descriptions** for better model understanding
3. **ProviderStrategy** when available (OpenAI, Grok)
4. **ToolStrategy** for universality
5. **Validate** with ge, le, min_length, etc.
6. **Union types** for multiple formats

## When to Use

✅ **Use Structured Output when:**
- Need guaranteed data formats
- Extracting information from text
- API integrations
- Validation is critical

❌ **Don't use when:**
- Need natural user responses
- Creative text generation
- Open-ended dialogues

## Troubleshooting

### Model Not Following Schema

```python
# Use ProviderStrategy if available
response_format=ProviderStrategy(YourSchema)

# Or add more detailed descriptions
class Contact(BaseModel):
    name: str = Field(description="Full name of the person (first and last)")
    email: str = Field(description="Valid email address (user@domain.com)")
```

### Validation Errors

```python
# Add error handling
response_format=ToolStrategy(
    schema=YourSchema,
    handle_errors=True  # Automatic retries
)
```

### Incomplete Data

```python
# Use Optional fields
class Contact(BaseModel):
    name: str
    email: str | None = None  # Optional
    phone: str | None = None  # Optional
```
