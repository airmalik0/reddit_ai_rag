#!/usr/bin/env python3
"""Entry point for the Agentic RAG Agent."""

import argparse

from dotenv import load_dotenv
load_dotenv()

from src.config import DEFAULT_MODEL
from src.agent_with_eval import HybridRAG
from src.db import get_stats


def chat(model_name: str | None = None, debug: bool = False):
    """Interactive chat with the Hybrid RAG agent.

    Args:
        model_name: Model identifier (e.g., "openai:gpt-4.1")
        debug: Show debug information (node execution)
    """
    # Show database stats
    print("Connecting to databases...")
    stats = get_stats()
    print(f"  Posts: {stats['posts']}")
    print(f"  Chunks: {stats['chunks']}")
    print(f"  Subreddits: {stats['subreddits']}")

    print("\nInitializing Hybrid RAG...")
    agent = HybridRAG(model_name)

    print("\n" + "=" * 60)
    print(f"Hybrid RAG Agent ({agent.model_name})")
    print("Architecture: Agent -> Tools -> Evaluator (with feedback loop)")
    print("Type 'quit' to exit, 'reset' to start new conversation")
    print("Type 'trace' to see full execution trace of last query")
    print("=" * 60 + "\n")

    last_query = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if user_input.lower() == 'reset':
            agent.reset()
            last_query = None
            print("Conversation reset.\n")
            continue

        if user_input.lower() == 'trace' and last_query:
            print("\n--- Execution Trace ---")
            trace = agent.get_trace(last_query)
            print(f"Query: {trace.get('query')}")
            print(f"Tool Calls: {trace.get('tool_calls')}")
            print(f"Evaluation Count: {trace.get('evaluation_count')}")
            print(f"Final Grade: {trace.get('grade')}")
            if trace.get('feedback'):
                print(f"Feedback: {trace.get('feedback')}")
            print(f"Response: {trace.get('response')[:200]}..." if len(trace.get('response', '')) > 200 else f"Response: {trace.get('response')}")
            print("--- End Trace ---\n")
            continue

        last_query = user_input

        try:
            if debug:
                print("\n--- Debug Mode ---")
                for event_type, content in agent.stream(user_input):
                    if event_type == "debug":
                        print(f"  {content}")
                    elif event_type == "response":
                        print(f"\nAssistant: {content}")
                print()
            else:
                print("\nAssistant: ", end="", flush=True)
                response = agent.invoke(user_input)
                print(response)
                print()

        except Exception as e:
            print(f"\nError: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid RAG Agent")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help=f"Model to use (default: {DEFAULT_MODEL}). Examples: openai:gpt-4.1, anthropic:claude-sonnet-4-5"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Show debug information (node execution)"
    )
    args = parser.parse_args()

    chat(model_name=args.model, debug=args.debug)
