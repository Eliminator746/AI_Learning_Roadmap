# Problem 4: Pydantic schema with validation

# Build a tool for booking a meeting: book_meeting(title, start_time, duration_minutes, attendees: list[str]). Use args_schema=BookMeetingInput with a Pydantic model that has field-level validation (e.g. duration_minutes must be between 15 and 240 via a Literal or a custom validator). Ask the model something that would violate a constraint and see what happens at the schema level vs. what you'd need to check in your function body.

# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, Field
import requests
import wikipedia

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0:featherless-ai",
#     task="text-generation",
#     max_new_tokens=200,
# )
# model = ChatHuggingFace(llm=llm)

MODEL = "gemini-3.6-flash"

model = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0,
)

class BookMeetingInput(BaseModel):
    title: str = Field(..., description="A concise title for the meeting. Must be provided by the user — do not invent one.")
    start_time: str = Field(..., description="The meeting start time, as stated by the user.")
    duration_minutes: int = Field(..., ge=15, le=240, description="Meeting duration in minutes, between 15 and 240.")
    attendees: list[str] = Field(..., description="Names of people attending. Must be explicitly provided by the user.")
    
@tool(args_schema=BookMeetingInput)
def book_meeting(title: str, start_time: str, duration_minutes: int, attendees: list[str]) -> str:
    """
    Book a meeting with a title, start time, duration, and attendees.
    """
    return (
        f"Meeting booked: {title} at {start_time} for {duration_minutes} minutes "
        f"with attendees: {', '.join(attendees) if attendees else 'none'}"
    )

# 1. Bind tool
llm_with_tool = model.bind_tools([book_meeting])

# Message
query = "Please book a meeting for me at 12:30pm for 45mins"
# query = "Book a meeting titled 'Sprint Planning' at 12:30pm for 45 minutes with attendees Rahul and Priya"

messages = [
    SystemMessage(content="When booking a meeting, only call book_meeting once the user has explicitly provided all of: title, start_time, duration_minutes, and attendees. If any are missing, ask the user for them instead of guessing or using placeholder values."),   # System Mess -> extra layer of protection so that llm don't ask random things.
    HumanMessage(content=query)
]

# Call llm_with_tool with human query
ai_mess = llm_with_tool.invoke(messages)
print(ai_mess)
print("--" * 20)

# Append ai mess
messages.append(ai_mess)

if ai_mess.tool_calls:
    tool = ai_mess.tool_calls[0]
    
    # Invoke desired function with full input
    tool_result = book_meeting.invoke(tool['args'])
    
    # Write and Append ToolMessage
    tool_mess = ToolMessage(
        content=tool_result,
        tool_call_id=tool['id']
    )
    
    messages.append(tool_mess)
    
    # Do final call
    final_response = model.invoke(messages)
    print(final_response.text)
else:
    print("No tool call happened")
    print(ai_mess.text)