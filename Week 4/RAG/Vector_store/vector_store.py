"""
Updated version — mid-2026 LangChain.

Changes from your original:
1. Chroma import fixed:
   OLD (deprecated): from langchain_community.vectorstores import Chroma
   NEW:               from langchain_chroma import Chroma
   -> pip install langchain-chroma

2. NOTHING else needed changing:
   - "gemini-3.6-flash" is a current, valid model id (Google's latest
     Flash-tier model as of July 2026) - your original code already
     had this right.
   - persist_directory + no .persist() call is correct for modern
     Chroma (>=0.4.x auto-persists; .persist() was removed and will
     now raise AttributeError if called).

3. Reminder (not a code change): langchain-google-genai v4+ uses the
   consolidated google-genai SDK, which reads GEMINI_API_KEY from the
   environment (not the older GOOGLE_API_KEY). Make sure your .env has:
       GEMINI_API_KEY=...
       OPENAI_API_KEY=...     # still needed since you're using OpenAIEmbeddings
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma  # <-- FIXED: was langchain_community.vectorstores
from langchain_core.documents import Document
import sys

env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(env_path)

MODEL = "gemini-3.6-flash"  # current model id, no change needed

model = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0,
)

# Create LangChain documents for IPL players

doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
    metadata={"team": "Royal Challengers Bangalore"}
)
doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
    metadata={"team": "Mumbai Indians"}
)
doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
    metadata={"team": "Chennai Super Kings"}
)
doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
    metadata={"team": "Mumbai Indians"}
)
doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
    metadata={"team": "Chennai Super Kings"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory='my_chroma_db',
    collection_name='sample'
)

# add documents
# NOTE: Chroma auto-generates a UUID per document if you don't pass ids=[...].
# Capture the returned ids so you can reference them later (e.g. for update/delete)
# instead of hardcoding a UUID string, which is fragile.
added_ids = vector_store.add_documents(docs)
print("Added IDs:", added_ids)

# view documents
print(vector_store.get(include=['embeddings', 'documents', 'metadatas']))

sys.exit()

# search documents
vector_store.similarity_search(
    query='Who among these are a bowler?',
    k=2
)

# search with similarity score
vector_store.similarity_search_with_score(
    query='Who among these are a bowler?',
    k=2
)

# meta-data filtering
vector_store.similarity_search_with_score(
    query="",
    filter={"team": "Chennai Super Kings"}
)

# update documents
updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata={"team": "Royal Challengers Bangalore"}
)

# Use an id from added_ids (or from vector_store.get()) rather than a
# hardcoded string that won't exist in your own database.
vector_store.update_document(document_id=added_ids[0], document=updated_doc1)

# view documents
print(vector_store.get(include=['embeddings', 'documents', 'metadatas']))

# delete document
vector_store.delete(ids=[added_ids[0]])

# view documents
print(vector_store.get(include=['embeddings', 'documents', 'metadatas']))