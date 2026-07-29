from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

MODEL="gemini-3.6-flash"


model = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0,
)

# Schema
# class Review(TypedDict):
#     summary:str
#     sentiment:str

class Review(TypedDict):
    summary: Annotated[ str, "A brief summary of the customer's review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]


review = "I had a disappointing experience. The product did not meet my expectations, and the issues I faced were not resolved quickly."

messages = [
    SystemMessage("You are a helpful reviewer assistant. You need to understand the review of Human given to a product and tell me the summary and sentiment of their review"),
    HumanMessage(content = review)
]

structured_model = model.with_structured_output(Review)
result = structured_model.invoke(messages)

print("AI reply: \n", result)
print("Type of result: ", type(result))