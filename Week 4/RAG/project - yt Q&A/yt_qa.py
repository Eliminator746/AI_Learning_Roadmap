from dotenv import load_dotenv
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
import sys

env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(env_path)

MODEL = "gemini-3.6-flash"  # current model id, no change needed

model = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0,
)



from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

video_id = "OYvlznJ4IZQ"

# Loading
try:
    api = YouTubeTranscriptApi()

    transcript_list = api.fetch(video_id, languages=["en"])

    # Flatten it to plain text
    transcript = " ".join(snippet.text for snippet in transcript_list)

    print(transcript)
    sys.exit()

except TranscriptsDisabled:
    print("No captions available for this video.")


# Chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = text_splitter.create_documents([transcript])  # returns a list of LangChain Document objects


# Embedding
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory='my_chroma_db',
    collection_name='sample'
)


# ============================================================
# CREATE DETERMINISTIC IDs
# ============================================================

ids = [
    f"youtube_{video_id}_chunk_{index}"
    for index in range(len(chunks))
]


# ============================================================
# CHECK FOR EXISTING DOCUMENTS
# ============================================================

existing_documents = vector_store.get(
    ids=ids,
    include=[]
)
# # include=[] -> Don't return documents, embeddings, or metadata. I only want to know which IDs exist

# existing_ids = set(existing_documents["ids"])


# # ============================================================
# #  REMOVE DUPLICATE CHUNKS
# # ============================================================

# new_chunks = []
# new_ids = []

# for chunk, chunk_id in zip(chunks, ids):
#     if chunk_id not in existing_ids:
#         new_chunks.append(chunk)
#         new_ids.append(chunk_id)

if existing_documents["ids"]:
    vector_store.delete(ids=existing_documents["ids"])
added_ids = vector_store.add_documents(documents=chunks, ids=ids)

# ============================================================
#  ADD NEW DOCUMENTS TO CHROMA
# ============================================================

# FIX: Chroma's add_documents/upsert throws a ValueError if given an
# empty list ("Expected Embeddings to be non-empty list or numpy array,
# got []"). This happens whenever every chunk for this video_id was
# already added in a previous run (since persist_directory keeps data
# across runs). Guard against the empty case instead of always calling
# add_documents unconditionally.
# if new_chunks:
#     added_ids = vector_store.add_documents(documents=new_chunks, ids=new_ids)
#     print(f"Added {len(added_ids)} new chunk(s):", added_ids)
# else:
    
#     print(f"No new chunks to add — all {len(chunks)} chunk(s) for video '{video_id}' already exist in the store.")

# view documents
# print(vector_store.get(include=['embeddings', 'documents', 'metadatas']))

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """
)


question          = "What this video talk about?"
retrieved_docs    = retriever.invoke(question)
# print(retrieved_docs) # List of docs

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
final_prompt = prompt.invoke({"context": context_text, "question": question})

answer = model.invoke(final_prompt)
print("\nAnswer:", answer.text)  # FIX: was answer.content


# We are manually calling fn eveytime i.e we are invoking mulitiple times, so to automate we can build chain. WIth single invoke, all steps will be done
# Building a Chain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text


parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question":RunnablePassthrough()
})

parallel_chain.invoke('who is the GRANDFATHER of ISEKAI?')
print("Parallel chain: ", parallel_chain)

main_chain = parallel_chain | prompt | model | parser

answer = main_chain.invoke(question)
print("answer : ", answer)
