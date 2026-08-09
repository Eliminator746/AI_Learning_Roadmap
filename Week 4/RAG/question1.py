# Problem B1: Basic RAG chain, one strategy
# Since this is your first LangChain RAG build (not the manual retriever scripts from before — this time wire it as a proper retriever | format_docs LCEL chain like we did for the YouTube Q&A case), start simple: load some documents, embed with HuggingFaceEmbeddings, store in Chroma, build a basic similarity retriever, and answer a question end-to-end.

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from pathlib import Path
import os

print(os.getcwd())

# Load environment variables
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

MODEL = "gemini-3.6-flash"

TESTING_MODE = False
MAX_PAGES_TO_READ = 1

# LLM
model = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0,
)

# PDF path
FILE_PATH = r"E:\AI_roadmap\Week 4\RAG\docs\Not a Penny More, Not a Penny Less by Jeffrey Archer_removed.pdf"

# Load PDF
loader = PyPDFLoader(FILE_PATH)

# Read all pages
all_docs = loader.load()



# Testing mode
if TESTING_MODE:
    print(f"Printing first {MAX_PAGES_TO_READ} page(s):\n")

    for i, doc in enumerate(all_docs[:MAX_PAGES_TO_READ]):
        print(f"Page {i + 1}")
        print(doc.page_content[:1000])
        print("-" * 80)

print(f"\nLoaded {len(all_docs)} pages.")



text_splitter = RecursiveCharacterTextSplitter( chunk_size=2000, chunk_overlap=200 )
chunks = text_splitter.split_documents(all_docs)

print(len(chunks))
print(chunks[0].page_content)

