from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage,BaseMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.tools import tool
from typing import TypedDict, Sequence, Annotated
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

model = "gemini-3.6-flash"

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


document_content = ""

@tool
def update(content: str) -> str:
    """Updates the document with the provided content."""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is:\n{document_content}"

@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process.
    
    Args:
        filename: Name for the text file.
    """
    
    global document_content

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"
        
    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    
    except Exception as e:
        return f"Error saving document: {str(e)}"
        



tools = [update, save]

model = ChatGoogleGenerativeAI(
    model=model,
    temperature=0,
)

llm_with_tool = model.bind_tools(tools=tools)

def call_llm(state:AgentState) -> AgentState:
    
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)
    
    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)
    
    
    res = llm_with_tool.invoke([system_prompt] + state["messages"] + [user_message])

    return {"messages": state["messages"] + [user_message, res]} # mess history by appending all



def should_continue(state: AgentState):
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]

    for message in reversed(messages):
        # ... and checks if this is a ToolMessage resulting from save
        if (isinstance(message, ToolMessage) and  
            "saved" in message.content.lower() and # you are searching inside the ToolMessage.content, not inside tool_calls.
            "document" in message.content.lower()):
            return "end" # goes to the end edge which leads to the endpoint
        
    return "continue"
    

# ToolMessage
# ├── name: "save"
# ├── content: "Document has been saved successfully to 'leave_request_email.txt'."
# └── tool_call_id: "..."

# AIMessage
#     └── tool_calls
#           └── name = "save"     ← LLM requested save
          


graph = StateGraph(AgentState)

graph.add_node("llm_agent", call_llm)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("llm_agent")

graph.add_edge("llm_agent", "tools")


graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "llm_agent",
        "end": END,
    },
)


app = graph.compile()


def run_document_agent():
    print("\n ===== DRAFTER =====")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            for message in step["messages"]:
                message.pretty_print()

    print("\n ===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_document_agent()