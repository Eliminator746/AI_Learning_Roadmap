# Problem 7: response_format for structured output
# Give your agent a Pydantic model via response_format= (e.g. a WeatherReport schema with city: str, temperature_c: float, summary: str). Ask a weather question and inspect the result's structured field vs. the raw message content — understand where LangChain slots the structured-output step into the agent loop (it's not the same mechanism as tool calling, even though both use schemas).

from pydantic import BaseModel
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pathlib import Path
import requests


env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

model = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash",
    temperature = 0
)


class WeatherReport(BaseModel):
    city:str
    temperature_c:float
    summary:str

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
    # return {
    #         "city": location["name"],
    #         "temperature_c": float(weather["temperature_2m"]),
    #         "summary": summary,
    #     }
    return (
        f"Weather in {location['name']}: {weather['temperature_2m']}°C, "
        f"humidity {weather['relative_humidity_2m']}%, "
        f"wind {weather['wind_speed_10m']} km/h"
    )

agent = create_agent(
    model=model,
    tools=[get_current_weather],
    response_format=WeatherReport,
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the weather of Kolkata?"}
    ]
})

print(result["structured_response"])
print(type(result["structured_response"]))  # <class '__main__.WeatherReport'>
print("--"*20)
print("\n\n")
print(result)
print("\n\n")

# Compare against the raw conversational messages, to see the two mechanisms
# side by side, as the exercise asks
for m in result["messages"]:
    print(m.__class__.__name__, "-", getattr(m, "text", m.content))