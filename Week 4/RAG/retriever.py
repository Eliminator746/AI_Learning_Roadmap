"""
Updated LangChain retrievers demo — fixed for current LangChain (post-1.0, mid-2026).

Changes from the original / video code:
1. Chroma import fixed:
   OLD (deprecated, removed in 1.0): from langchain_community.vectorstores import Chroma
   NEW:                              from langchain_chroma import Chroma
   -> Chroma was split out of langchain_community into its own partner
      package (langchain-chroma). `pip install langchain-chroma` if you
      haven't already.

2. Added the missing import for WikipediaRetriever (the original snippet
   used it without importing it):
   from langchain_community.retrievers import WikipediaRetriever

3. Swapped `gpt-3.5-turbo` -> `gpt-4o-mini`. gpt-3.5-turbo is being phased
   out by OpenAI; gpt-4o-mini is the current cheap/fast default and is
   what OpenAI recommends for this kind of workload.

4. NEW (this update) — LangChain 1.0 import path changes:
   As of the LangChain 1.0 major release, several pre-1.0 building blocks
   -- including MultiQueryRetriever, ContextualCompressionRetriever, and
   LLMChainExtractor -- were moved out of the core `langchain` package
   into a new separate package: `langchain_classic`. The old
   `from langchain.retrievers...` imports now raise ImportError on
   langchain>=1.0.
   OLD (broken on langchain>=1.0):
       from langchain.retrievers.multi_query import MultiQueryRetriever
       from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
       from langchain.retrievers.document_compressors import LLMChainExtractor
   NEW:
       from langchain_classic.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
       from langchain_classic.retrievers.document_compressors import LLMChainExtractor
   -> pip install -U langchain-classic

5. FAISS import and .invoke() usage were already using current LangChain
   APIs -- no changes needed there.
"""

from langchain_community.retrievers import WikipediaRetriever
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma  # <-- FIXED: was langchain_community.vectorstores
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document

from langchain_classic.retrievers import MultiQueryRetriever, ContextualCompressionRetriever  # <-- FIXED: moved out of `langchain` in v1.0
from langchain_classic.retrievers.document_compressors import LLMChainExtractor  # <-- FIXED: moved out of `langchain` in v1.0

MODEL_NAME = "gpt-4o-mini"  # <-- FIXED: was gpt-3.5-turbo (being retired)

# ---------------------------------------------------------------------------
# 1. WikipediaRetriever
# ---------------------------------------------------------------------------
retriever = WikipediaRetriever(top_k_results=2, lang="en")

query = "the geopolitical history of india and pakistan from the perspective of a chinese"
docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")


# ---------------------------------------------------------------------------
# 2. Chroma vector store + retriever
# ---------------------------------------------------------------------------
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

embedding_model = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "What is Chroma used for?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)


# ---------------------------------------------------------------------------
# 3. MMR (Maximal Marginal Relevance) with FAISS
# ---------------------------------------------------------------------------
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

embedding_model = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(documents=docs, embedding=embedding_model)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5},
)

query = "What is langchain?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)


# ---------------------------------------------------------------------------
# 4. MultiQueryRetriever
# ---------------------------------------------------------------------------
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents=all_docs, embedding=embedding_model)

similarity_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=ChatOpenAI(model=MODEL_NAME),
)

query = "How to improve energy levels and maintain balance?"

similarity_results = similarity_retriever.invoke(query)
multiquery_results = multiquery_retriever.invoke(query)

for i, doc in enumerate(similarity_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

print("*" * 150)

for i, doc in enumerate(multiquery_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)


# ---------------------------------------------------------------------------
# 5. ContextualCompressionRetriever
# ---------------------------------------------------------------------------
docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"}),
]

embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding_model)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(model=MODEL_NAME)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor,
)

query = "What is photosynthesis?"
compressed_results = compression_retriever.invoke(query)

for i, doc in enumerate(compressed_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)