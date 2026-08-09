from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from typing import TypedDict, Sequence, Annotated
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

model = "gemini-3.6-flash"

@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a"""
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

tools = [add, subtract, multiply]

model = ChatGoogleGenerativeAI(
    model=model,
    temperature=0,
).bind_tools(tools=tools)



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def model_chat_call(state: AgentState) -> AgentState:
    
    system_prompt = SystemMessage(
        content= "Use tool if available else fetch from website the latest data and tell in 2 sentences only at max"
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]} # SInce we've to return state so we need to return as dict


def should_continue(state: AgentState): 
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: 
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("my_agent", model_chat_call)
graph.add_edge(START, "my_agent")

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_conditional_edges(
    "my_agent",
    should_continue,
    {
        "continue" : "tools",
        "end" : END
    }
)

graph.add_edge("tools", "my_agent") # After the tools execute, you need to return to the agent. Otherwise, after a tool runs, the graph has nowhere to continue.


graph = graph.compile()

query = "What is 2*200+3-45 * 10 ? and who is Modi ?"
res = graph.invoke({"messages": [HumanMessage(content = query)]}) # Here message is List[mess]. Sequence means it can be list,tuple, or other ordered collections

for message in res["messages"]:
    message.pretty_print()
    print() # blank line between messages