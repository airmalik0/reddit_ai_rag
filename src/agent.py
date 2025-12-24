"""RAG Agent implementation using LangChain."""

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from .config import DEFAULT_MODEL
from .tools import create_search_tool


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Reddit search assistant. Your job is to find relevant posts
and experiences from Reddit communities based on user queries.

## How You Work
1. When user asks a question, use the search_posts tool to find relevant Reddit posts
2. Summarize what people on Reddit shared about this topic
3. Include post titles and subreddit names as sources
4. Present different perspectives and experiences from the community

## Response Style
- Focus on what real people shared, not your own advice
- Quote or paraphrase actual experiences from posts
- Mention which subreddit each insight comes from (r/learnprogramming, r/Coffee, r/explainlikeimfive)
- If posts have conflicting opinions, present both sides

## Language
Respond in the same language the user writes in."""


# ---------------------------------------------------------------------------
# Agent Class
# ---------------------------------------------------------------------------

class RAGAgent:
    """RAG Agent for searching Reddit posts.

    Supported models:
    - OpenAI: "openai:gpt-5.2", "openai:gpt-4o", "openai:gpt-4o-mini"
    - Anthropic: "anthropic:claude-sonnet-4-5", "anthropic:claude-opus-4"
    """

    def __init__(self, model_name: str | None = None):
        """Initialize the RAG agent.

        Args:
            model_name: Model identifier (e.g., "openai:gpt-5.2")
        """
        self.model_name = model_name or DEFAULT_MODEL

        # Initialize model
        self.model = init_chat_model(
            self.model_name,
            temperature=0.3,
            max_tokens=2000
        )

        # Create search tool (uses Qdrant + Postgres internally)
        search_tool = create_search_tool()

        # Create agent
        self.agent = create_agent(
            model=self.model,
            tools=[search_tool],
            system_prompt=SYSTEM_PROMPT
        )

        self.conversation_history: list[dict] = []

    def reset(self):
        """Reset conversation history."""
        self.conversation_history = []

    def invoke(self, user_message: str) -> str:
        """Run agent synchronously.

        Args:
            user_message: User's input message

        Returns:
            Agent's response text
        """
        self.conversation_history.append({"role": "user", "content": user_message})

        response = self.agent.invoke({"messages": self.conversation_history})

        assistant_message = response['messages'][-1].content
        self.conversation_history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def stream(self, user_message: str):
        """Stream agent response.

        Args:
            user_message: User's input message

        Yields:
            Text tokens as they are generated
        """
        self.conversation_history.append({"role": "user", "content": user_message})

        full_response = ""
        for mode, chunk in self.agent.stream(
            {"messages": self.conversation_history},
            stream_mode=["messages", "custom"]
        ):
            if mode == "messages":
                token, metadata = chunk
                node = metadata.get("langgraph_node", "")

                # Only process model node outputs (not tool results)
                if node == "model" and hasattr(token, "content"):
                    content = token.content

                    if isinstance(content, str) and content:
                        full_response += content
                        yield content

                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "")
                                if text:
                                    full_response += text
                                    yield text

        self.conversation_history.append({"role": "assistant", "content": full_response})


# ---------------------------------------------------------------------------
# Graph Export for LangGraph Studio
# ---------------------------------------------------------------------------

def create_graph():
    """Create the agent graph for LangGraph Studio."""
    from langchain.chat_models import init_chat_model
    from langchain.agents import create_agent

    model = init_chat_model(DEFAULT_MODEL, temperature=0.3, max_tokens=2000)
    search_tool = create_search_tool()

    return create_agent(
        model=model,
        tools=[search_tool],
        system_prompt=SYSTEM_PROMPT
    )


# Export graph for LangGraph Studio
graph = create_graph()
