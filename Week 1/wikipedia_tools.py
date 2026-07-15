"""
Researcher agent's tools: Wikipedia's public search + summary APIs.
No auth needed. Two tools:
  - wikipedia_search: find candidate page titles for a query
  - wikipedia_get_summary: get the intro/summary for a specific title

Kept to summary-length content (not full article dumps) so the LLM's
context window doesn't get blown out by one tool call.
"""
import re
import requests

USER_AGENT = "GenAI-MultiAgent-Demo/1.0 (learning project)"


def wikipedia_search(query: str, limit: int = 5) -> list:
    """Search Wikipedia, return a list of {title, snippet} candidates."""
    resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json().get("query", {}).get("search", [])
    return [
        {"title": r["title"], "snippet": _strip_html(r.get("snippet", ""))}
        for r in results
    ]


def wikipedia_get_summary(title: str) -> str:
    """Fetch the intro/summary section for a specific Wikipedia page title."""
    resp = requests.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}",
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    if resp.status_code == 404:
        return f"No Wikipedia page found for '{title}'."
    resp.raise_for_status()
    data = resp.json()
    return data.get("extract", "No summary available.")


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


# OpenAI-style function-calling schemas exposed to the LLM.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Search Wikipedia for pages matching a query. Returns candidate titles + snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_get_summary",
            "description": "Get the summary/intro section of a specific Wikipedia page by exact title.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Exact Wikipedia page title"},
                },
                "required": ["title"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "wikipedia_search": wikipedia_search,
    "wikipedia_get_summary": wikipedia_get_summary,
}
