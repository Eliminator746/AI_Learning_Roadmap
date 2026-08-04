# Problem 6: Tool error handling
# Make one of your tools intentionally throw an exception for certain inputs (e.g. get_current_weather("Atlantis") raises ValueError). Handle it two ways: (a) let it crash and see what breaks, (b) catch it and return an error string as the ToolMessage content instead, so the model can react to the failure gracefully in its next turn. This maps directly to ToolNode's handle_tool_errors — build the manual version first, then try handle_tool_errors=True and compare.

from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

MODEL = "gemini-3.6-flash"

model = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0,
)

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0:featherless-ai",
#     task="text-generation",
#     max_new_tokens=200,
# )
# model = ChatHuggingFace(llm=llm)


@tool
def get_current_weather(location: str):
    "Return me the json formatted weather information"
    if location == "Kolkata":
        return {
            "location": "Kolkata",
            "temperature_c": 33,
            "condition": "Mostly cloudy"
        }
    elif location == "Delhi":
        return {
            "location": "Delhi",
            "temperature_c": 35,
            "condition": "Hot"
        }
    else:
        raise ValueError(f"Unknown location: {location}")

llm_with_tool = model.bind_tools([get_current_weather])

query = "What is the weather of Atlantis?"
messages = [
    HumanMessage(content=query)
]

ai_mess = llm_with_tool.invoke(messages)
messages.append(ai_mess)

if ai_mess.tool_calls:
    tool = ai_mess.tool_calls[0]

    # tool_result = get_current_weather.invoke(tool['args'])
    
    try:
        tool_result = get_current_weather.invoke(tool['args'])
    except Exception as e:
        tool_result = f"Error: {e}"   # becomes the ToolMessage content instead of crashing


    tool_mess = ToolMessage(
        content = tool_result,
        tool_call_id=tool['id']
    )
    
    messages.append(tool_mess)
    
    final_res = model.invoke(messages)
    print(final_res.text)
    
else:
    print("no tool call happened")
    print(ai_mess)