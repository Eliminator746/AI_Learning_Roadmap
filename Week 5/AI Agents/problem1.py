# Problem 1: Minimal create_agent
# Build a create_agent with 2-3 simple tools (reuse add, get_current_weather from before). Invoke it with a multi-step question that requires calling more than one tool in sequence (e.g. "What's the weather in Delhi, and what's that temperature times 2?"). Print every message in the result and trace exactly how many round-trips the agent made.

# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents import create_agent
from dotenv import load_dotenv
from pathlib import Path
import requests
import wikipedia

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0:featherless-ai",    
#     task="text-generation",
#     max_new_tokens=200,
# )
# model = ChatHuggingFace(llm=llm)

MODEL = "gemini-3.6-flash"

model = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0,
)

@tool
def add(a:int, b:int) -> int:
    """Addition of two numbers. Return the sum of two numbers"""
    return a+b

@tool
def get_current_weather(city: str) -> str:
    """Get the current weather of a city"""
    geo_resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1}
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
        }
    )
    weather = weather_resp.json()["current"]

    return (
        f"Weather in {location['name']}: {weather['temperature_2m']}°C, "
        f"humidity {weather['relative_humidity_2m']}%, "
        f"wind {weather['wind_speed_10m']} km/h"
    )
    
@tool
def search_wikipedia(query:str)-> str:
    """Search Wikipedia and return a short summary for a query."""
    summary=wikipedia.summary(query, sentences=3, auto_suggest=True)
    return summary



# question = "What's the weather in Delhi, what's that temperature times 2?, and also tell me the sum of 10 + 91"
question = "weather in kolkat, temperature of delhi, tell me my name, tell me about mushuko tensai"
messages = [HumanMessage(content=question)]



agent = create_agent(
    model=model,
    tools=[get_current_weather, search_wikipedia, add],
    system_prompt=(
        "You are a helpful assistant. "
        "For multi-part questions, use tools step by step and answer every part. "
        "If a tool is needed, call it. If multiple tools are needed, call them in sequence. "
        "If a required tool is not available, say so."
    )
)

# result = agent.invoke(messages) wrong
result = agent.invoke({"messages": messages})   # ✅
# create_agent returns — LangGraph's state schema

print(result)
print("---" * 20)

for m in result["messages"]:
    role = m.__class__.__name__
    text = getattr(m, "text", None)
    content = text if isinstance(text, str) else str(m.content)
    print(f"[{role}] {content}")
    print("-" * 40)