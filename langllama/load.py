import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, Settings

from llama_index.llms.ollama import Ollama

# load from disk
db = chromadb.PersistentClient(path="./chroma_db")

#print(db)

chroma_collection = db.get_or_create_collection("cortazar")

#print(chroma_collection)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

embed_model = HuggingFaceEmbedding(
    model_name="thuan9889/llama_embedding_model_v1"
)

Settings.llm = Ollama(model="llama2", request_timeout=660.0)

index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)

# Query Data from the persisted index
query_engine = index.as_query_engine()

response = query_engine.query("what is the context about? tell me anything!")

print(response)
#display(Markdown(f"<b>{response}</b>"))