from typing import TypedDict
from langgraph.graph import END, START, StateGraph

class AgentState(TypedDict):
    number1:int
    number2:int
    operation:str
    result:int

def add_node(state: AgentState) -> AgentState:
    """Add the two numbers"""
    state["result"] = state["number1"] + state["number2"]
    return state


def sub_node(state: AgentState) -> AgentState:
    """Substract two numbers"""
    state["result"] = state["number1"] - state["number2"]
    return state

def router_node(state: AgentState) -> AgentState:
    """This node will select the next node of the graph"""
    if state["operation"] == "+":
        return "addition_operation" # return edges
    else:
        return "substraction_operation"
    

graph = StateGraph(AgentState)
graph.add_node("add", add_node)
graph.add_node("sub", sub_node)
graph.add_node("router", lambda state:state) # passthrough function
graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    router_node,
    {
        "addition_operation" : "add",
        "substraction_operation" : "sub"
    }
)   
graph.add_edge("add", END)
graph.add_edge("sub", END)

app = graph.compile()

initial_state_1 = AgentState(number1 = 10, operation="-", number2 = 5)
final_output = app.invoke(initial_state_1)
print(final_output)


from PIL import Image
import io

image = Image.open(io.BytesIO(app.get_graph().draw_mermaid_png()))
image.show()