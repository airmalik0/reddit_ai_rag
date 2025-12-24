"""Tool definitions for the RAG agent."""

from pydantic import BaseModel, Field
from langchain.tools import tool

from .config import DEFAULT_NUM_RESULTS, MAX_NUM_RESULTS
from .vector_store import search_similar
from .db import get_posts_by_ids


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class SearchPostsInput(BaseModel):
    """Input schema for searching programming posts."""
    query: str = Field(
        description="Search query to find relevant programming posts (e.g., 'how to learn Python', 'career advice for junior developers')"
    )
    num_results: int = Field(
        default=DEFAULT_NUM_RESULTS,
        ge=1,
        le=MAX_NUM_RESULTS,
        description=f"Number of results to return (1-{MAX_NUM_RESULTS})"
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

def create_search_tool():
    """Create the search tool.

    Returns:
        Tool function for searching posts
    """

    @tool(args_schema=SearchPostsInput)
    def search_posts(query: str, num_results: int = DEFAULT_NUM_RESULTS) -> str:
        """Search through programming-related Reddit posts.

        Use this tool to find information about programming questions,
        learning experiences, career advice, and technical discussions.

        Args:
            query: Search query to find relevant posts
            num_results: Number of results to return (1-10)

        Returns:
            Relevant posts with titles, URLs, and content
        """
        # Search Qdrant for similar chunks (with deduplication by post_id)
        search_results = search_similar(query, k=num_results * 2)

        if not search_results:
            return "No relevant posts found."

        # Get post IDs
        post_ids = [r['post_id'] for r in search_results[:num_results]]

        # Fetch full post data from Postgres
        posts = get_posts_by_ids(post_ids)

        if not posts:
            return "No relevant posts found."

        # Create a map for ordering
        post_map = {p.id: p for p in posts}

        # Format results in search order (full content for agent and evaluator)
        results = []
        for i, post_id in enumerate(post_ids, 1):
            post = post_map.get(post_id)
            if post:
                results.append(f"--- Result {i} ---")
                results.append(f"Subreddit: r/{post.subreddit_name}")
                results.append(f"Title: {post.title}")
                results.append(f"URL: {post.url}")
                results.append(f"Content: {post.content if post.content else '(no content)'}")
                results.append("")

        return "\n".join(results)

    return search_posts
