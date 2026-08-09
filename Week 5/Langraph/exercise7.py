from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Union
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

model = "gemini-3.6-flash"

model = ChatGoogleGenerativeAI(
    model=model,
    temperature=0,
)

class AgentState(TypedDict):
    messages: list[Union[HumanMessage, AIMessage]]


def chat(state: AgentState) -> AgentState:
    response = model.invoke(state["messages"])
    print("AI: ", response.text)
    state["messages"].append(AIMessage(content=response.text))
    return state


graph = StateGraph(AgentState)
graph.add_node("chat_node", chat)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

graph = graph.compile()




chat_history = []
while True:
    query = input("Chat with AI: ")
    
    if query == "end":
            break
    
    chat_history.append(HumanMessage(content = query))
    res = graph.invoke({"messages": chat_history})
    # chat_history.append(res)
    conversation_history = res["messages"]
    

with open("conversation.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    for chat in conversation_history:
        if isinstance(chat, HumanMessage):
            file.write(f"You: {chat.content}\n")
        elif isinstance(chat, AIMessage):
            file.write(f"AI: {chat.content}\n")
    file.write("End of Conversation")    

print(chat_history)
print("Conversation saved to conversation.txt")

    

# while loop job:
# Read user input.
# Invoke the graph.
# Print the response.
# Repeat.