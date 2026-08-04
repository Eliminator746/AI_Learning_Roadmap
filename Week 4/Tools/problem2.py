# Level 2 — Multiple tools, schema control

# Tool selection
# Create three tools: add(a: int, b: int), get_current_weather(city: str), search_wikipedia(query: str). Bind all three. Ask a question that should only trigger one of them (e.g. "What's 15 + 27?"), then a question that should trigger a different one, then an ambiguous question that could go either way — see what the model picks and why.

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
from pathlib import Path
import requests
import wikipedia

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

llm = HuggingFaceEndpoint(
    repo_id="suayptalha/DeepSeek-R1-Distill-Llama-3B:featherless-ai",
    task="text-generation",
    max_new_tokens=200,
)
model = ChatHuggingFace(llm=llm)

# MODEL = "gemini-3.6-flash"

# model = ChatGoogleGenerativeAI(
#     model=MODEL,
#     temperature=0,
# )

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



question = "Tell me about Mushuko Tensai Season 2"
messages = [HumanMessage(content=question)]

# 1. bind tool
llm_with_tool = model.bind_tools([add, get_current_weather, search_wikipedia])

# 2. checks which tool to use
res = llm_with_tool.invoke(messages)
print(res)

# 3. invoke tool i.e fn with args
messages.append(res)

if res.tool_calls:
    
    tools_by_name = {t.name: t for t in [add, get_current_weather, search_wikipedia]} # key:str, value:fn
    tool_call = res.tool_calls[0]
    selected_tool = tools_by_name[tool_call['name']]
    tool_result = selected_tool.invoke(tool_call["args"])
        
    tool_mess = ToolMessage(
        content=tool_result,
        tool_call_id=tool_call['id']
    )
    messages.append(tool_mess)
    final_res = model.invoke(messages)
    print(f'{"--" * 20}\n{"--" * 20}')
    
    print(final_res.text)
else:
    print("No tool call done")