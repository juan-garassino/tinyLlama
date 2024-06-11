# from llama_index.llms.ollama import Ollama

# llm = Ollama(model="llama2", request_timeout=60.0)

# response = llm.complete("What is the capital of France?")

# print(response)

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext

# Creating a Chroma client
# EphemeralClient operates purely in-memory, PersistentClient will also save to disk
db = chromadb.PersistentClient(path="./chroma_db")

chroma_collection = db.create_collection("cortazar")

# construct vector store
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

documents = SimpleDirectoryReader("data").load_data()

# Settings.chunk_size = 512

# nomic embedding model
# Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="thuan9889/llama_embedding_model_v1"
)

# ollama
Settings.llm = Ollama(model="llama2", request_timeout=660.0)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, transformations=[SentenceSplitter(chunk_size=256)]
) #, embed_model=embed_model

query_engine = index.as_query_engine(similarity_top_k=3)

#query_engine = index.as_chat_engine()

response = query_engine.query("Is 'la maga' a caracter of one of the books? which one?")

print(response)

# response = query_engine.chat("Oh interesting, tell me more.")
# print(response)