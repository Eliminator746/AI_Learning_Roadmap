from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import streamlit as st
from pathlib import Path

load_dotenv()
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

MODEL = "gemini-2.0-flash"  # double check this against your available models — "gemini-3.6-flash" doesn't match any released naming pattern

model = ChatGoogleGenerativeAI(model=MODEL, temperature=0)

SYSTEM_PROMPT = SystemMessage(
    content="You are a helpful assistant. Provide a short answer (maximum 4 sentences). "
            "If the context is insufficient, say you don't know."
)

# --- State initialization (runs once per session, not per rerun) ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SYSTEM_PROMPT]

st.title("Chat conversation window")
st.write("Let's chat!!")

# --- Render past turns (skip the SystemMessage) ---
for msg in st.session_state.chat_history[1:]:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# --- New input ---
question = st.chat_input("Enter your question:")

if question:
    st.session_state.chat_history.append(HumanMessage(content=question))
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            res = model.invoke(st.session_state.chat_history)
        st.write(res.content)

    st.session_state.chat_history.append(AIMessage(content=res.content))