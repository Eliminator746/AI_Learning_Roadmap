# from langgraph.graph import StateGraph
# from typing import TypedDict
# from math import prod


# # schema
# class AgentState(TypedDict):
#     name: str
#     value: list[int]
#     operation: str
#     result: int
    
# def operation(state: AgentState) -> AgentState: # This is a Node, return only the fields it wants to update
#     """ Add or multiply and return the result"""
#     if state["operation"] == "+":
#         return {"result": sum(state["value"])}

#     elif state["operation"] == "*":
#         return {"result": prod(state["value"])}

#     return {"result": None}

# graph = StateGraph(AgentState)

# graph.add_node("operation", operation)
# graph.set_entry_point("operation")
# graph.set_finish_point("operation")

# graph = graph.compile()

# answers  = graph.invoke({"value": [1,2,3,4], "name": "Steve", "operation":"*"})
# print(answers)

from typing import TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    name: str
    age: str
    final: str

def first_node(state:AgentState) -> AgentState:
    """This is the first node of our sequence"""

    state["final"] = f"Hi {state["name"]}!"
    return state

def second_node(state:AgentState) -> AgentState:
    """This is the second node of our sequence"""

    state["final"] = state["final"] + f" You are {state["age"]} years old!"

    return state

graph = StateGraph(AgentState)

graph.add_node("first_node", first_node)
graph.add_node("second_node", second_node)

graph.set_entry_point("first_node")
graph.add_edge("first_node", "second_node")
graph.set_finish_point("second_node")
app = graph.compile()

from PIL import Image
import io

image = Image.open(io.BytesIO(app.get_graph().draw_mermaid_png()))
image.show()

result = app.invoke({"name": "Charlie", "age": 20})
print(result)