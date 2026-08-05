from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import random

class AgentState(TypedDict):
    name: str
    numbers: list[int]
    final: str
    counter: int

def greeting_node(state: AgentState) -> AgentState:
    state['final'] = f"Hi {state['name']} !!"
    state['counter'] = 0
    return state

def random_node(state: AgentState) -> AgentState:
    """Generates a random number from 0 to 10"""
    
    # state['numbers'] = [random.randint(0,10) for i in range(5)]
    state["numbers"].append(random.randint(0, 10))
    state["counter"] += 1
    
    return state


def should_continue(state: AgentState) -> AgentState:
    if state['counter'] < 5:
        print("ENTERING LOOP", state["counter"])
        return "loop"  # Continue looping
    else:
        return "exit"
    

graph = StateGraph(AgentState)
graph.add_node("greeting", greeting_node)
graph.add_node("random", random_node)

graph.add_edge("greeting", "random")

graph.add_conditional_edges(
    "random",
    should_continue,  # Action
    {
        "loop": "random",  
        "exit": END          
    }
)

graph.add_edge(START, "greeting")

app = graph.compile()

output = app.invoke({
    "name":"Tarun",
    "numbers":[],
    "counter":-2
})

print(output)