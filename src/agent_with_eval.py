"""Hybrid RAG: Tool-calling Agent + Evaluator pattern.

Combines:
- Agentic RAG: Agent with tools decides when/how to search
- Evaluator-Optimizer: Quality check with feedback loop

Architecture:
    User Query → Agent (with tools) ←─────────────┐
                      │                           │
              ┌───────┴───────┐                   │
              ▼               ▼                   │
         [tool call]    [ready to answer]         │
              │               │                   │
              ▼               ▼                   │
         Tool Node       Evaluator                │
              │               │                   │
              │         ┌─────┴─────┐             │
              │         ▼           ▼             │
              │       pass        fail            │
              │         │           │             │
              └────→ Agent ←────────┘ (feedback)
                        │
                      [END]
"""

from typing import Literal, Annotated
from operator import add

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START, END

from .config import DEFAULT_MODEL, DEFAULT_NUM_RESULTS
from .tools import create_search_tool


# ---------------------------------------------------------------------------
# Evaluator Schema
# ---------------------------------------------------------------------------

class EvaluatorOutput(BaseModel):
    """Schema for evaluator decision."""
    grade: Literal["pass", "fail"] = Field(
        description="Whether the response meets quality criteria"
    )
    feedback: str = Field(
        default="",
        description="Specific feedback on how to improve the response"
    )


# ---------------------------------------------------------------------------
# Default Prompts
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """You are a Reddit search assistant. Your job is to find relevant posts
and experiences from Reddit communities based on user queries.

## Available Subreddits
You have access to posts from these communities:
- r/learnprogramming - Programming questions, learning resources, career advice
- r/Coffee - Coffee brewing, equipment, beans, cafes
- r/explainlikeimfive - Simple explanations of complex topics

## When to Search
- USE search_posts when user asks questions that benefit from real experiences/opinions
  (product recommendations, "how do I...", comparisons, reviews, advice)
- DO NOT search for greetings, small talk, clarifying questions, or general knowledge
  ("hi", "thanks", "what can you do?", "how are you?")

## How You Work
1. Decide if the query needs Reddit search
2. If yes: search, then summarize what people shared with sources
3. If no: respond naturally without searching

## Response Style (when searching)
- Focus on what real people shared, not your own advice
- Quote or paraphrase actual experiences from posts
- Always mention which subreddit each insight comes from
- If posts have conflicting opinions, present both sides

## Language
Respond in the same language the user writes in."""

DEFAULT_EVALUATOR_PROMPT = """You are a quality evaluator. Check ONLY these 2 criteria:

## 1. Hallucination Check
If tools were used: Does the response accurately reflect the tool results?
- FAIL if the agent invented information not in tool results
- FAIL if the agent misrepresented what the tools returned
- PASS if response is grounded in actual tool results (or no tools were needed)

## 2. Safety Check
- FAIL if the agent was rude, offensive, or disrespectful
- FAIL if the agent gave dangerous/harmful advice
- PASS if the agent behaved professionally

## Conversation to Evaluate:
{conversation}

Grade "pass" if both criteria are met.
Grade "fail" with specific feedback explaining which criterion failed."""


# ---------------------------------------------------------------------------
# Graph Configuration (editable in LangGraph Studio)
# ---------------------------------------------------------------------------

class GraphConfig(BaseModel):
    """Configuration for the Hybrid RAG agent.

    These settings can be modified in LangGraph Studio UI
    without restarting the server.
    """
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        description="System prompt that defines agent behavior and response style"
    )
    evaluator_prompt: str = Field(
        default=DEFAULT_EVALUATOR_PROMPT,
        description="Prompt for quality evaluation (must contain {conversation} placeholder)"
    )
    max_evaluations: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum evaluation attempts before accepting response"
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class HybridState(MessagesState):
    """State for Hybrid RAG agent."""
    # Evaluator state
    evaluation_count: int
    grade: str
    feedback: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def create_agent_node(model, tools):
    """Create the agent node that can call tools."""
    llm_with_tools = model.bind_tools(tools)

    def agent(state: HybridState, config: RunnableConfig) -> dict:
        messages = state["messages"]
        feedback = state.get("feedback", "")

        # Get system prompt from config (editable in LangGraph Studio)
        system_prompt = config.get("configurable", {}).get(
            "system_prompt", DEFAULT_SYSTEM_PROMPT
        )

        # Check if this is a new user turn (last message is HumanMessage)
        # Reset evaluation_count on new turn to allow fresh evaluation attempts
        is_new_turn = len(messages) > 0 and isinstance(messages[-1], HumanMessage)

        # If we have feedback from evaluator, add it as system message
        if feedback:
            feedback_msg = SystemMessage(
                content=f"[Quality Feedback] Your previous response was rejected. Please improve based on this feedback:\n{feedback}\n\nTry searching again or provide more detailed information with proper citations."
            )
            response = llm_with_tools.invoke(
                [SystemMessage(content=system_prompt)] + messages + [feedback_msg]
            )
        else:
            response = llm_with_tools.invoke(
                [SystemMessage(content=system_prompt)] + messages
            )

        # Reset evaluation_count on new user turn, clear feedback always
        result = {"messages": [response], "feedback": ""}
        if is_new_turn:
            result["evaluation_count"] = 0

        return result

    return agent


def create_tool_node(tools):
    """Create the tool execution node."""
    tools_by_name = {tool.name: tool for tool in tools}

    def tool_node(state: HybridState) -> dict:
        messages = state["messages"]
        last_message = messages[-1]

        results = []
        for tool_call in last_message.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            result = tool.invoke(tool_call["args"])
            results.append(
                ToolMessage(content=result, tool_call_id=tool_call["id"])
            )

        return {"messages": results}

    return tool_node


def _format_conversation(messages) -> str:
    """Format messages into readable conversation for evaluator."""
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"[USER]: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(f"[AGENT TOOL CALL]: {tc['name']}({tc['args']})")
            if msg.content:
                lines.append(f"[AGENT]: {msg.content}")
        elif isinstance(msg, ToolMessage):
            lines.append(f"[TOOL RESULT]: {msg.content}")
    return "\n".join(lines)


def create_evaluator_node(model):
    """Create the evaluator node that checks response quality."""
    evaluator_llm = model.with_structured_output(EvaluatorOutput)

    def evaluator(state: HybridState, config: RunnableConfig) -> dict:
        messages = state["messages"]
        evaluation_count = state.get("evaluation_count", 0)

        # Get evaluator prompt from config (editable in LangGraph Studio)
        evaluator_prompt = config.get("configurable", {}).get(
            "evaluator_prompt", DEFAULT_EVALUATOR_PROMPT
        )

        # Get the last AI response (non-tool-call)
        last_response = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                last_response = msg.content
                break

        if not last_response:
            return {
                "grade": "pass",
                "feedback": "",
                "evaluation_count": evaluation_count + 1
            }

        # Format full conversation for evaluator
        conversation = _format_conversation(messages)

        # Evaluate
        result = evaluator_llm.invoke([
            {"role": "system", "content": "You are a quality evaluator."},
            {"role": "user", "content": evaluator_prompt.format(conversation=conversation)}
        ])

        return {
            "grade": result.grade,
            "feedback": result.feedback if result.grade == "fail" else "",
            "evaluation_count": evaluation_count + 1
        }

    return evaluator


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def should_continue(state: HybridState) -> Literal["tools", "evaluator"]:
    """Route after agent: to tools or to evaluator."""
    messages = state["messages"]
    last_message = messages[-1]

    # If agent made tool calls, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise, evaluate the response
    return "evaluator"


def after_evaluation(state: HybridState, config: RunnableConfig) -> Literal["agent", "__end__"]:
    """Route after evaluator: back to agent or finish."""
    grade = state.get("grade", "pass")
    evaluation_count = state.get("evaluation_count", 0)

    # Get max_evaluations from config (editable in LangGraph Studio)
    max_evaluations = config.get("configurable", {}).get("max_evaluations", 2)

    # Pass or max evaluations reached
    if grade == "pass" or evaluation_count >= max_evaluations:
        return "__end__"

    # Fail - go back to agent with feedback
    return "agent"


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_graph(model_name: str | None = None):
    """Build the Hybrid RAG graph.

    Args:
        model_name: Model identifier (e.g., "openai:gpt-5.2")

    Returns:
        Compiled StateGraph with configurable prompts (editable in LangGraph Studio)
    """
    model_name = model_name or DEFAULT_MODEL
    model = init_chat_model(model_name, temperature=0.3, max_tokens=2000)

    # Create tools
    search_tool = create_search_tool()
    tools = [search_tool]

    # Create nodes
    agent = create_agent_node(model, tools)
    tool_node = create_tool_node(tools)
    evaluator = create_evaluator_node(model)

    # Build graph
    builder = StateGraph(HybridState, config_schema=GraphConfig)

    # Add nodes
    builder.add_node("agent", agent)
    builder.add_node("tools", tool_node)
    builder.add_node("evaluator", evaluator)

    # Add edges
    builder.add_edge(START, "agent")

    # Agent -> tools OR evaluator
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "evaluator": "evaluator"
        }
    )

    # Tools -> agent (continue the loop)
    builder.add_edge("tools", "agent")

    # Evaluator -> end OR agent (with feedback)
    builder.add_conditional_edges(
        "evaluator",
        after_evaluation,
        {
            "__end__": END,
            "agent": "agent"
        }
    )

    return builder.compile()


# ---------------------------------------------------------------------------
# Agent Class (for programmatic use)
# ---------------------------------------------------------------------------

class HybridRAG:
    """Hybrid RAG Agent: Tool-calling + Evaluator pattern.

    Combines:
    - Agentic RAG: Agent decides when/how to use search tools
    - Evaluator: Checks response quality with feedback loop

    Supported models:
    - OpenAI: "openai:gpt-5.2", "openai:gpt-4o", "openai:gpt-4o-mini"
    - Anthropic: "anthropic:claude-sonnet-4-5", "anthropic:claude-opus-4"

    Note: For LangGraph Studio, use the exported `graph` directly.
    Prompts and settings are configurable via Studio UI.
    """

    def __init__(self, model_name: str | None = None):
        """Initialize the Hybrid RAG agent.

        Args:
            model_name: Model identifier
        """
        self.model_name = model_name or DEFAULT_MODEL
        self.graph = build_graph(model_name)
        self.conversation_history: list = []

    def reset(self):
        """Reset conversation history."""
        self.conversation_history = []

    def invoke(self, user_message: str, **config_overrides) -> str:
        """Run agent synchronously.

        Args:
            user_message: User's input message
            **config_overrides: Override default config (system_prompt, evaluator_prompt, max_evaluations)

        Returns:
            Agent's response text
        """
        self.conversation_history.append(HumanMessage(content=user_message))

        result = self.graph.invoke(
            {
                "messages": self.conversation_history,
                "evaluation_count": 0,
                "grade": "",
                "feedback": ""
            },
            config={"configurable": config_overrides} if config_overrides else None
        )

        # Get the last AI response
        response = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                response = msg.content
                break

        self.conversation_history.append(AIMessage(content=response))
        return response

    def stream(self, user_message: str, **config_overrides):
        """Stream agent execution with node updates.

        Args:
            user_message: User's input message
            **config_overrides: Override default config

        Yields:
            Tuple of (node_name, content)
        """
        self.conversation_history.append(HumanMessage(content=user_message))

        for event in self.graph.stream(
            {
                "messages": self.conversation_history,
                "evaluation_count": 0,
                "grade": "",
                "feedback": ""
            },
            config={"configurable": config_overrides} if config_overrides else None,
            stream_mode="updates"
        ):
            for node_name, node_output in event.items():
                # Yield node execution info
                if node_name == "agent":
                    msgs = node_output.get("messages", [])
                    for msg in msgs:
                        if isinstance(msg, AIMessage):
                            if msg.tool_calls:
                                for tc in msg.tool_calls:
                                    yield ("tool_call", f"Calling {tc['name']}: {tc['args']}")
                            elif msg.content:
                                yield ("response", msg.content)

                elif node_name == "tools":
                    msgs = node_output.get("messages", [])
                    for msg in msgs:
                        if isinstance(msg, ToolMessage):
                            yield ("tool_result", f"Got {len(msg.content)} chars")

                elif node_name == "evaluator":
                    grade = node_output.get("grade", "")
                    feedback = node_output.get("feedback", "")
                    yield ("evaluation", f"Grade: {grade}" + (f", Feedback: {feedback}" if feedback else ""))

    def get_trace(self, user_message: str, **config_overrides) -> dict:
        """Run agent and return execution trace.

        Args:
            user_message: User's input message
            **config_overrides: Override default config

        Returns:
            Dict with execution details
        """
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=user_message)],
                "evaluation_count": 0,
                "grade": "",
                "feedback": ""
            },
            config={"configurable": config_overrides} if config_overrides else None
        )

        # Extract info
        tool_calls = []
        response = ""
        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    tool_calls.extend(msg.tool_calls)
                elif msg.content:
                    response = msg.content

        return {
            "query": user_message,
            "tool_calls": len(tool_calls),
            "evaluation_count": result.get("evaluation_count", 0),
            "grade": result.get("grade", ""),
            "feedback": result.get("feedback", ""),
            "response": response
        }


# ---------------------------------------------------------------------------
# Graph Export for LangGraph Studio
# ---------------------------------------------------------------------------

def create_graph():
    """Create the agent graph for LangGraph Studio.

    The graph supports configurable parameters via Studio UI:
    - system_prompt: Agent's system prompt
    - evaluator_prompt: Quality evaluation prompt
    - max_evaluations: Max retry attempts (1-5)
    """
    return build_graph(DEFAULT_MODEL)


# Export graph for LangGraph Studio
graph = create_graph()
