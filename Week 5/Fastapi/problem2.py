# Problem A2: Session-scoped memory via thread_id
# Add a session_id field to the request body. Map it directly to thread_id in your agent's config, using the checkpointer pattern you already built in Problem 3. Test with two different session_ids in a row and confirm they don't share memory — same test you already did manually, now behind a real HTTP interface. Think about where the agent + checkpointer should be instantiated: once at app startup (shared across requests) vs. per-request — and why one of those is clearly wrong for a checkpointer holding conversation state.

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from fastapi import FastAPI
from dotenv import load_dotenv
from pathlib import Path


env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

model="gemini-3.6-flash"

model = ChatGoogleGenerativeAI(
    model = model,
    temperature = 0
)

app = FastAPI()

agent = create_agent(
    model=model,
    tools=[],
    checkpointer=InMemorySaver()
)





@app.post("/chat")
async def chat(query:str, session_id:str):
    
    thread_config = {"configurable": {"thread_id": session_id}}
    res = agent.ainvoke(
        {"messages": {"role": "user", "content": query}},
        thread_config,
    )
    
    print(res)
    return {"response": res}