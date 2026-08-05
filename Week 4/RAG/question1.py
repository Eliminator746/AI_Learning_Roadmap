# Problem B1: Basic RAG chain, one strategy
# Since this is your first LangChain RAG build (not the manual retriever scripts from before — this time wire it as a proper retriever | format_docs LCEL chain like we did for the YouTube Q&A case), start simple: load some documents, embed with HuggingFaceEmbeddings, store in Chroma, build a basic similarity retriever, and answer a question end-to-end.


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_docling.loader import DoclingLoader
from langchain_chroma import Chroma
from dotenv import load_dotenv
from pathlib import Path

import os

print(os.getcwd())

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

MODEL = "gemini-3.6-flash"  # current model id, no change needed

model = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0,
)


FILE_PATH = "E:\AI_roadmap\Week 4\RAG\docs\i want to eat your pancreas.pdf"
loader = DoclingLoader(file_path = FILE_PATH)

documents = loader.load()

# For large datasets, lazily load documents
documents = []

# for i, doc in enumerate(loader.lazy_load()):
#     if i >= 10:
#         break
#     documents.append(doc)

for i, document in enumerate(loader.lazy_load()):
    print("Document number:", i)
    print("Metadata:", document.metadata)
    print("Content preview:", document.page_content[:200])
    print("-" * 50)

    if i == 2:  # check first 3 outputs only
        break