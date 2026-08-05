from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class AgentState(TypedDict):
    number1: int 
    operation: str
    number2: int
    finalNumber: int
    number3: int
    operation2: str
    number4: int
    finalNumber2: int

def add_node(state: AgentState) -> AgentState:
    """Add the two numbers"""
    state["finalNumber"] = state["number1"] + state["number2"]
    return state

def sub_node(state: AgentState) -> AgentState:
    """Substract two numbers"""
    state["finalNumber"] = state["number1"] - state["number2"]
    return state

def router_node(state: AgentState) -> str:
    """This node will select the next node of the graph"""
    if state["operation"] == "+":
        return "addition_operation"
    else:
        return "substraction_operation"


    

def add_node2(state: AgentState) -> AgentState:
    """Add the two numbers"""
    state["finalNumber2"] = state["number3"] + state["number4"]
    return state

def sub_node2(state: AgentState) -> AgentState:
    """Substract two numbers"""
    state["finalNumber2"] = state["number3"] - state["number4"]
    return state

def router_node2(state: AgentState) -> str:
    """This node will select the next node of the graph"""
    if state["operation2"] == "+":
        return "addition_operation"
    else:
        return "substraction_operation"
    


graph = StateGraph(AgentState)

graph.add_node("add", add_node)
graph.add_node("sub", sub_node)
graph.add_node("router", lambda state:state)

graph.add_conditional_edges(
    "router",
    router_node,
    {
        "addition_operation":"add",
        "substraction_operation":"sub"
    }
)



graph.add_node("router2", lambda state:state)
graph.add_edge("add", "router2")
graph.add_edge("sub", "router2")

graph.add_node("add2", add_node2)
graph.add_node("sub2", sub_node2)
graph.add_conditional_edges(
    "router2",
    router_node2,
    {
        "addition_operation":"add2",
        "substraction_operation":"sub2"
    }
)


graph.add_edge(START, "router")
graph.add_edge("add2", END)
graph.add_edge("sub2", END)

app = graph.compile()

initial_state = AgentState(number1 = 10, operation="-", number2 = 5, number3 = 7, number4=2, operation2="+", finalNumber= 0, finalNumber2 = 0)
print(app.invoke(initial_state))



from PIL import Image
import io

image = Image.open(io.BytesIO(app.get_graph().draw_mermaid_png()))
image.show()