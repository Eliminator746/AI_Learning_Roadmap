from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

MODEL = "gemini-2.5-flash" 


model = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0,
)

response = model.invoke("What is the capital of India?")

print(response.content)