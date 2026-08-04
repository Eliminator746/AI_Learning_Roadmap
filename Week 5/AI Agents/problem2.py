# Problem 3: Checkpointer + thread_id
# Add checkpointer=InMemorySaver() to your agent (from langgraph.checkpoint.memory). Call agent.invoke(...) twice with the same config={"configurable": {"thread_id": "abc"}}, asking a follow-up question the second time that depends on the first (e.g. "What's 10 + 5?" then "Multiply that by 3"). Confirm it works. Then call it a third time with a different thread_id and confirm the model has no memory of the earlier conversation — this proves you understand what thread_id actually scopes.

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

MODEL = "gemini-3.6-flash"

model = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0,
)



agent = create_agent(
    model=model,
    tools=[],  # no tools needed to test checkpointing
    checkpointer=InMemorySaver()
)

thread_config = {"configurable": {"thread_id": "1"}}

messages = {"messages": [HumanMessage(content="My name is Rohan. Age: 23, city: Amsterdam")]}

res = agent.invoke(
    messages,
    thread_config,
)

final_output = agent.invoke(
    {"messages": [HumanMessage(content="Where I live and what's my age?")]},
    thread_config,
)

print(final_output["messages"][-1].text)


thread_config = {"configurable": {"thread_id": "2"}}

messages = {"messages": [HumanMessage(content="What is my name?")]}

second_session = agent.invoke(
    messages,
    thread_config,
)

print(second_session["messages"][-1].text)

