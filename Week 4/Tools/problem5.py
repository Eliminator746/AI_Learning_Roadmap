"""
Problem 7: create_agent with ToolRuntime

Rebuilds Problem 3's tools (add, get_current_weather, search_wikipedia)
inside a create_agent(...) agent instead of a manual tool-call loop, and
adds a new tool (summarize_conversation) that uses ToolRuntime to read
runtime.state["messages"] -- i.e. it can see the full conversation history
without the LLM having to pass that history in as an argument.

Key API notes (current LangChain, mid/late-2026):
- `create_agent` lives in `langchain.agents` (this is the prebuilt
  ReAct-style agent constructor -- it wraps a LangGraph graph for you,
  so you don't need to hand-build the tool-call loop like in Problems 1-2).
- `ToolRuntime` lives in `langchain.tools`, alongside the `tool` decorator.
- A tool parameter typed as `ToolRuntime` is automatically injected by the
  framework at execution time -- it never appears in the tool's JSON
  schema, so the LLM never sees it and never has to "provide" it. This is
  dependency injection, not a normal tool argument.
- `runtime.state["messages"]` gives you the full list of conversation
  messages (HumanMessage/AIMessage/ToolMessage/...) accumulated so far in
  the agent's graph state -- this is the mechanism this problem is testing.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from dotenv import load_dotenv
from pathlib import Path
import requests
import wikipedia

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

MODEL = "gemini-3.6-flash"

model = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0,
)


# ---------------------------------------------------------------------------
# Problem 3's tools, rebuilt for use with create_agent
# (identical logic to before -- create_agent just needs plain @tool-decorated
# functions, same as bind_tools did)
# ---------------------------------------------------------------------------

@tool
def add(a: int, b: int) -> int:
    """Add two numbers and return the sum."""
    return a + b


@tool
def get_current_weather(city: str) -> str:
    """Get the current weather for a given city."""
    geo_resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=10,
    )
    geo_data = geo_resp.json()

    if not geo_data.get("results"):
        return f"Could not find location: {city}"

    location = geo_data["results"][0]
    lat, lon = location["latitude"], location["longitude"]

    weather_resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,wind_speed_10m,relative_humidity_2m",
        },
        timeout=10,
    )
    weather = weather_resp.json()["current"]

    return (
        f"Weather in {location['name']}: {weather['temperature_2m']}\u00b0C, "
        f"humidity {weather['relative_humidity_2m']}%, "
        f"wind {weather['wind_speed_10m']} km/h"
    )


@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return a short summary for a query."""
    try:
        return wikipedia.summary(query, sentences=3, auto_suggest=True)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"'{query}' is ambiguous. Possible matches: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'"


# ---------------------------------------------------------------------------
# NEW: tool that uses ToolRuntime to read conversation state directly,
# instead of the LLM having to pass the conversation in as an argument.
# ---------------------------------------------------------------------------

@tool
def summarize_conversation(runtime: ToolRuntime) -> str:
    """Summarize the conversation so far.
    Use this when the user asks you to recap or summarize what has been
    discussed. Do NOT pass conversation text as an argument -- this tool
    reads the conversation directly from agent state.
    """
    # `runtime` is injected automatically -- it is NOT part of the tool's
    # JSON schema, so the model never generates a value for it. This is
    # the "state injection" behavior the exercise is testing.
    messages = runtime.state["messages"]

    lines = []
    for m in messages:
        role = m.__class__.__name__.replace("Message", "")
        # message content can be a string OR a list of content blocks
        # (see the Gemini 3 "thought signature" behavior from earlier) --
        # normalize with .text where available.
        text = getattr(m, "text", None)
        content = text if isinstance(text, str) else str(m.content)
        if content.strip():
            lines.append(f"{role}: {content[:200]}")

    return "Conversation so far:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Build the agent -- this replaces the manual bind_tools + tool_calls loop
# from Problems 1-6 with a prebuilt ReAct-style agent.
# ---------------------------------------------------------------------------

agent = create_agent(
    model=model,
    tools=[add, get_current_weather, search_wikipedia, summarize_conversation],
)

# ---------------------------------------------------------------------------
# Run it
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "What's 15 + 27?"},
            {"role": "user", "content": "Now tell me the weather in Kolkata."},
            {"role": "user", "content": "Summarize what we've talked about so far."},
        ]
    })

    for m in result["messages"]:
        role = m.__class__.__name__
        text = getattr(m, "text", None)
        content = text if isinstance(text, str) else str(m.content)
        print(f"[{role}] {content}")
        print("-" * 40)