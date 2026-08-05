# Problem 8 (capstone): Two agents, one delegates to the other
# Build a research_agent (has search_wikipedia) and a math_agent (has add, multiply). Build a third "supervisor" tool/agent that, given a question, decides which sub-agent to call and passes the question along, then returns the sub-agent's answer. Don't reach for a prebuilt multi-agent framework yet — hand-build the routing logic yourself first, so you understand what a supervisor pattern is actually doing before using any higher-level abstraction for it.

from pydantic import BaseModel
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pathlib import Path
import wikipedia


env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

model="gemini-3.6-flash"

model = ChatGoogleGenerativeAI(
    model = model,
    temperature = 0
)

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return a short summary for a query."""
    try:
        summary = wikipedia.summary(query, sentences=3, auto_suggest=True)
        return summary
    except Exception as e:
        return f"Wikipedia lookup failed for '{query}': {e}"

@tool
def add(a:float, b:float) -> float:
    """Return the sum of two numbers"""
    return a + b

@tool
def multiply(a:float, b:float) -> float:
    """Return the sum of two numbers"""
    return a * b


research_agent = create_agent(
    model=model,
    tools=[search_wikipedia],
    system_prompt="You are a research agent. Use search_wikipedia and return a concise answer."
)

math_agent = create_agent(
    model=model,
    tools=[add, multiply],
    system_prompt="You are a math agent. Use add and multiply to solve arithmetic questions."
)


# Wrap sub-agents as tools

@tool("research_agent_tool", description="Handles research questions using Wikipedia.")
def call_research_agent(query: str) -> str:
    result = research_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content if result["messages"] else "No answer"

@tool("math_agent_tool", description="Handles math questions using add and multiply.")
def call_math_agent(query: str) -> str:
    result = math_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content if result["messages"] else "No answer"





supervisor_agent = create_agent(
    model=model,
    tools=[call_research_agent, call_math_agent],
    system_prompt="""
    You are a supervisor agent.
    Route research questions to research_agent_tool.
    Route arithmetic questions to math_agent_tool.
    If a question needs both, call both tools and combine the results.
    Do not answer from memory if a tool can help.
    """
)


result = supervisor_agent.invoke({
    "messages": [
        {"role": "user", "content": "What is 3*24 and what is the capital of Delhi?"}
    ]
})

print(result["messages"][-1].content)

print("\n\n")
print(result)