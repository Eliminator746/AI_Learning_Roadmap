# Problem 5: Custom @wrap_tool_call middleware
# Write a @wrap_tool_call middleware that logs every tool call's name and arguments before executing it (print to console), without changing behavior. Then modify it to implement a retry: if a tool call raises an exception, retry up to 3 times before giving up and returning an error ToolMessage. This is your Problem 6 error-handling concept, generalized to the agent/middleware level instead of inside one tool.

from langchain_core.messages import ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from langchain.agents import create_agent
from dotenv import load_dotenv
from pathlib import Path
import wikipedia


env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

MODEL = "gemini-3.6-flash"  # current model id, no change needed

model = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0,
)

def print_agent_trace(result):
    """Pretty-print an agent's result, showing each message's role,
    whether a tool was called (and with what args), and the content."""
    for msg in result["messages"]:
        role = msg.__class__.__name__

        # Get clean text content — handles both plain strings and the
        # Gemini 3 "list of content blocks" format
        text = getattr(msg, "text", None)
        content = text if isinstance(text, str) else str(msg.content)

        print(f"[{role}]")

        if content.strip():
            print(f"  content: {content}")

        # Show tool calls if this AIMessage requested any
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                print(f"  🔧 TOOL CALL -> {tc['name']}({tc['args']})  [id={tc['id']}]")

        # Show which tool this ToolMessage is a result for
        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id:
            tool_name = getattr(msg, "name", "unknown_tool")
            print(f"  ↩️  TOOL RESULT from {tool_name} [id={tool_call_id}]")

        print("-" * 60)
        

@tool
def search_wikipedia(query:str)-> str:
    """Search Wikipedia and return a short summary for a query."""
    raise ValueError("Forced failure for testing")
    summary=wikipedia.summary(query, sentences=3, auto_suggest=True)
    return summary

@wrap_tool_call
def handle_tool_log(request, handler):
    for attempt in range(3):
        try:
            print("Tool args: ", request.tool_call['args'], " --- Tool Name: ", request.tool_call['name'] )
            return handler(request) # Returns AI/Tool mess
        except Exception as e:
            last_error = e
            print(f"Retry {attempt + 1}/3 after error: {e}")
            # return  ToolMessage(
            #     content=f"Tool failed with error: {str(e)}",
            #     tool_call_id=request.tool_call["id"]
            # )
    
    # all 3 attempts failed — return an error ToolMessage instead of crashing
    return ToolMessage(
        content=f"Tool failed after 3 attempts. Last error: {last_error}",
        tool_call_id=request.tool_call["id"],
    )    
        
agent = create_agent(
    model=model,
    tools=[search_wikipedia],
    middleware=[handle_tool_log],
    system_prompt="You must use the search_wikipedia tool to answer any factual question, even if you think you already know the answer. Do not answer from your own knowledge.",
)

final_output = agent.invoke({
    "messages": [
        {"role": "user", "content": "Tell me about Python"}
    ]
})


print_agent_trace(final_output)