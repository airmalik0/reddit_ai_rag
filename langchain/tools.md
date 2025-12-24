# Tools in LangChain

## What is a Tool?

A Tool is a function that an agent can call to perform actions:
- Fetch data from APIs
- Query databases
- Perform calculations
- Send emails
- And any other operations

**Important:** A tool consists of:
1. **Schema** - name, description, parameters
2. **Function** - code to execute

## Basic Tool Creation

### With @tool Decorator

```python
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"

# Use with agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_database]
)
```

**Key rules:**
- Type hints are REQUIRED (str, int, list, etc.)
- Docstring is used by model to understand the tool
- First line of docstring - brief description
- Args in docstring help the model

### Custom Name

```python
@tool("web_search")  # Custom name
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

print(search.name)  # "web_search"
```

### Custom Description

```python
@tool(
    "calculator",
    description="Performs arithmetic calculations. Use this for any math problems."
)
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))
```

## Complex Input Data

### Pydantic Models

```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp}°{units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result
```

### JSON Schema

```python
weather_schema = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "units": {"type": "string"},
        "include_forecast": {"type": "boolean"}
    },
    "required": ["location"]
}

@tool(args_schema=weather_schema)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    return f"Weather in {location}"
```

## Access to Context (ToolRuntime)

### Reading State

```python
from langchain.tools import tool, ToolRuntime

@tool
def summarize_conversation(runtime: ToolRuntime) -> str:
    """Summarize the conversation so far."""
    messages = runtime.state["messages"]

    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")

    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses"

@tool
def get_user_preference(pref_name: str, runtime: ToolRuntime) -> str:
    """Get a user preference value."""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "Not set")
```

**Important:** `runtime: ToolRuntime` is NOT visible to the model - it's a hidden parameter for context access.

### Updating State

```python
from langgraph.types import Command
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

@tool
def clear_conversation() -> Command:
    """Clear the conversation history."""
    return Command(
        update={
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        }
    )

@tool
def update_user_name(new_name: str, runtime: ToolRuntime) -> Command:
    """Update the user's name."""
    return Command(update={"user_name": new_name})
```

### Access to Context (immutable context)

```python
from dataclasses import dataclass

USER_DATABASE = {
    "user123": {"name": "Alice", "balance": 5000},
    "user456": {"name": "Bob", "balance": 1200}
}

@dataclass
class UserContext:
    user_id: str

@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information."""
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return f"Account holder: {user['name']}\nBalance: ${user['balance']}"
    return "User not found"

# Use with context
agent = create_agent(
    model,
    tools=[get_account_info],
    context_schema=UserContext
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my balance?"}]},
    context=UserContext(user_id="user123")
)
```

### Access to Store (long-term memory)

```python
from typing import Any
from langgraph.store.memory import InMemoryStore

@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info."""
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

@tool
def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str:
    """Save user info."""
    store = runtime.store
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."

store = InMemoryStore()
agent = create_agent(
    model,
    tools=[get_user_info, save_user_info],
    store=store
)

# First session: save
agent.invoke({"messages": [{"role": "user", "content": "Save: userid abc123, name Foo, age 25"}]})

# Second session: read
agent.invoke({"messages": [{"role": "user", "content": "Get user info for 'abc123'"}]})
# "Name: Foo, Age: 25"
```

### Stream Writer (custom updates)

```python
@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer

    # Stream custom updates
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")

    return f"It's always sunny in {city}!"

# Use in agent stream
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "weather in SF"}]},
    stream_mode="custom"
):
    print(chunk)
# Output:
# Looking up data for city: San Francisco
# Acquired data for city: San Francisco
```

## Error Handling

### Middleware for Error Handling

```python
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return custom message to model
        return ToolMessage(
            content=f"Tool error: Please check your input ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search, get_weather],
    middleware=[handle_tool_errors]
)
```

## Best Practices

### 1. Good Docstrings

```python
# ❌ Bad
@tool
def get_data(x):
    """Get data"""
    return "data"

# ✅ Good
@tool
def get_user_data(user_id: str) -> dict:
    """Retrieve comprehensive user profile data from the database.

    Args:
        user_id: Unique identifier for the user (format: UUID)

    Returns:
        Dictionary containing user profile with keys: name, email, age, preferences
    """
    return {"name": "John", "email": "john@example.com"}
```

### 2. Parameter Typing

```python
# ❌ Bad - no type hints
@tool
def search(query, limit):
    """Search database"""
    return f"Results for {query}"

# ✅ Good - clear typing
@tool
def search_database(query: str, limit: int = 10) -> list[dict]:
    """Search customer database for matching records.

    Args:
        query: Search terms (e.g., customer name, email)
        limit: Maximum number of results (default 10, max 100)
    """
    return [{"id": 1, "name": "John"}]
```

### 3. Input Validation

```python
from pydantic import BaseModel, Field, validator

class SearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=10, ge=1, le=100)

    @validator('query')
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v

@tool(args_schema=SearchInput)
def search_database(query: str, limit: int = 10) -> list[dict]:
    """Search database with validated inputs."""
    return [{"id": 1, "name": "John"}]
```

### 4. Error Handling Inside Tools

```python
@tool
def api_call(endpoint: str) -> str:
    """Make API call to external service."""
    try:
        response = requests.get(f"https://api.example.com/{endpoint}")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        # Return clear message instead of raising exception
        return f"API error: {str(e)}. Please try again or check endpoint."
```

## Real-World Tool Examples

### Database Query Tool

```python
@tool
def query_database(sql: str, runtime: ToolRuntime) -> str:
    """Execute SQL query on the database.

    Args:
        sql: SQL query to execute (SELECT only)

    Returns:
        Query results as JSON string
    """
    # Security check
    if not sql.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries allowed"

    try:
        # Execute query
        results = execute_sql(sql)
        return json.dumps(results)
    except Exception as e:
        return f"Database error: {str(e)}"
```

### Web Search Tool

```python
@tool
def web_search(query: str, num_results: int = 5) -> list[dict]:
    """Search the web for information.

    Args:
        query: Search query
        num_results: Number of results to return (max 10)

    Returns:
        List of search results with title, url, snippet
    """
    # Integration with search API
    results = search_api.search(query, limit=min(num_results, 10))
    return [
        {
            "title": r.title,
            "url": r.url,
            "snippet": r.snippet
        }
        for r in results
    ]
```

### File Operations Tool

```python
from pathlib import Path

@tool
def read_file(file_path: str) -> str:
    """Read content from a file.

    Args:
        file_path: Path to file (must be in allowed directory)

    Returns:
        File content as string
    """
    path = Path(file_path)

    # Security check
    if not path.is_relative_to("/allowed/directory"):
        return "Error: Access denied to this path"

    try:
        return path.read_text()
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"
```

## Troubleshooting

### Tool Not Being Called

**Problem:** Model doesn't use the tool

**Solutions:**
1. Improve docstring - describe WHEN to use the tool
2. Give a more specific name
3. Simplify parameters
4. Use tool_choice="any" to force tool use

```python
# Before
@tool
def search(q: str) -> str:
    """Search"""
    return "results"

# After
@tool
def search_knowledge_base(search_query: str) -> str:
    """Search internal knowledge base for company information.

    Use this when the user asks about company policies, procedures, or documentation.

    Args:
        search_query: What to search for (e.g., 'vacation policy', 'expense reports')
    """
    return "results"
```

### Type Errors

**Problem:** Model passes wrong types

**Solution:** Use Pydantic for validation

```python
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    expression: str = Field(description="Math expression like '2 + 2'")
    precision: int = Field(default=2, ge=0, le=10)

@tool(args_schema=CalculatorInput)
def calculator(expression: str, precision: int = 2) -> float:
    """Calculate mathematical expression."""
    result = eval(expression)
    return round(result, precision)
```
