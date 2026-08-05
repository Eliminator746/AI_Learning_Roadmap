# Problem A1: Bare minimum — one endpoint, one agent call
# Build a FastAPI app with a single POST /chat endpoint that accepts {"message": str}, invokes a create_agent (reuse your weather/wikipedia tools), and returns {"response": str}. No memory yet, no streaming — just prove the plumbing works: FastAPI receiving a request, calling .invoke(), returning JSON.

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from fastapi import FastAPI
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel
import wikipedia


env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

model="gemini-3.6-flash"

model = ChatGoogleGenerativeAI(
    model = model,
    temperature = 0
)

app = FastAPI()


@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return a short summary for a query."""
    try:
        summary = wikipedia.summary(query, sentences=2, auto_suggest=True)
        return summary
    except Exception as e:
        return f"Wikipedia lookup failed for '{query}': {e}"


agent = create_agent(
    model = model,
    tools=[search_wikipedia],
    system_prompt="You are a helpful assistant. If a relevant tool is available, use it to answer the user’s request. If no suitable tool is available, answer from your own knowledge. Keep your response short, direct, and useful."
)



class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


# @app.post("/chat")
# def chat(query: str):
    
#     result = agent.invoke({
#         "messages": [{"role": "user", "content": query}]
#     })
    
#     return  {"response": result}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": request.message}]
    })
    return ChatResponse(response=result["messages"][-1].text)


# A plain str-typed parameter with no Pydantic model tells FastAPI to expect query as a URL query parameter (e.g. POST /chat?query=hello), not a JSON body field. Since the assignment specifically wants {"message": str} in the request body, you need a Pydantic model: