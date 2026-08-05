from typing import TypedDict

from langchain.agents import AgentState
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    name:str
    age:int
    skills: list[str]
    result: str

def first_node(state:AgentState) -> AgentState:
    """Name field"""
    state["result"] = f"{state['name']}, welcome to the system!"
    return state

def second_node(state: AgentState) -> AgentState:
    state["result"] += f" You are {state['age']} years old!"
    return state


def third_node(state: AgentState) -> AgentState:
    skills = ", ".join(state["skills"])
    state["result"] += f" You have skills in {skills}."
    return state


graph = StateGraph(AgentState)

graph.add_node("first", first_node)
graph.add_node("second", second_node)
graph.add_node("third", third_node)

graph.add_edge("first", "second")
graph.add_edge("second", "third")

graph.set_entry_point("first")
graph.set_finish_point("third")

graph = graph.compile()

mess = graph.invoke({"name":"Tarun", "age":24, "skills": ["JS", "Python", "Machine Learning", "LangGraph"] })

print(mess["result"])