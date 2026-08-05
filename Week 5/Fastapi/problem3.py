# Problem A4: Background execution for long-running agent tasks
# Add an endpoint that kicks off an agent run as a BackgroundTask, immediately returns a task_id, and a separate GET /status/{task_id} endpoint to poll for completion. This mirrors a common backend interview question pattern (you're already prepping FastAPI CRUD + this general "async job" shape) applied specifically to a scenario where an agent might take 30+ seconds (e.g. a multi-tool research task).

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


