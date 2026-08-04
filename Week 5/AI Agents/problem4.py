# Problem 6: HumanInTheLoopMiddleware
# Add a send_email(to: str, body: str) tool (just print instead of actually sending). Wrap it with HumanInTheLoopMiddleware(interrupt_on={"send_email": True}), with a checkpointer + thread_id configured. Invoke the agent with a request that triggers send_email, observe the interrupt, then resume execution with Command(resume={"decisions": [{"type": "approve"}]}). Then try again but resume with a "reject" decision instead, and see how the agent reacts to being told no.


from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from pathlib import Path
import requests
import wikipedia

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0:featherless-ai",    
    task="text-generation",
    max_new_tokens=200,
)
model = ChatHuggingFace(llm=llm)

# MODEL = "gemini-3.6-flash"

# model = ChatGoogleGenerativeAI(
#     model=MODEL,
#     temperature=0,
# )


# mess -> MESS + HISTORY -> LLM -> O/P  (will be attached to history)

