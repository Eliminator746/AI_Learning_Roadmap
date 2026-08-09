from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

model = "gemini-3.6-flash"


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    
    
# Load docs
loader  = PyPDFLoader(file_path= r"C:\Users\anish\Downloads\Stock_Market_Performance_2024.pdf", mode='page')
docs = []
docs_lazy = loader.lazy_load() # since we are using lazy_load(), we need loop to create list

for doc in docs_lazy:
    docs.append(doc)
    
# Chunking Process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

pages_split = text_splitter.split_documents(docs) # We now apply this to our docs


embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

persist_directory = r"chroma_db"
collection_name = "stock_market"

# If our collection does not exist in the directory, we create using the os command
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


try:
    # create db + store docs i.e chunks
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store!")
    
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise


# Now we create our retriever 
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Stock Market Performance 2024 document.
    """
    
    retrieved_docs = retriever.invoke(query)
    
    if not retrieved_docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."

    content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return content



tools = [retriever_tool]

model = ChatGoogleGenerativeAI(
    model=model,
)

llm_with_tool = model.bind_tools(tools=tools)

prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    system_prompt = SystemMessage(content = prompt)
    message = llm_with_tool.invoke([system_prompt] + state["messages"])
    return {'messages': [message]}

# llm -> sm + ques + retrieved_doc in str


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""
    last_message = state["messages"][-1]

    tool_map = {tool.name: tool for tool in tools}

    results = []

    for tool_call in last_message.tool_calls:
        tool = tool_map[tool_call["name"]]

        result = tool.invoke(tool_call["args"])

        results.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=tool_call["name"]
            )
        )

    return {"messages": results}



def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0
    
    
    


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


# final_prompt = prompt.invoke({"context": content, "question": query})
# answer = model.invoke(final_prompt)
# print("\nAnswer:", answer.text)


while True:
    print("\n=== RAG AGENT===")
    query = input("What info do you want to retrieve from document: ")
    message = HumanMessage(content = query)
    
    if query.lower() in ['exit', 'quit', 'end']:
        break
    
    final_ouput = rag_agent.invoke({"messages" : [message]})
    
    print("\n=== ANSWER ===")
    print(final_ouput['messages'][-1].content)