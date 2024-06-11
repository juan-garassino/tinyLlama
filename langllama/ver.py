import torch

from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    # model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf',
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=4096,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": -1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)


from llama_index import SimpleDirectoryReader
from llama_index import Document

documents = SimpleDirectoryReader(
    input_files = ["./documents/survey_on_llms.pdf"]
).load_data()

documents = Document(text = "\n\n".join([doc.text for doc in documents]))


import os
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index import VectorStoreIndex, ServiceContext, load_index_from_storage



def get_build_index(documents,llm,embed_model="local:BAAI/bge-small-en-v1.5",sentence_window_size=3,save_dir="./vector_store/index"):
  
  node_parser = SentenceWindowNodeParser(
      window_size = sentence_window_size,
      window_metadata_key = "window",
      original_text_metadata_key = "original_text"
  )

  sentence_context = ServiceContext.from_defaults(
      llm = llm,
      embed_model= embed_model,
      node_parser = node_parser,
  )

  if not os.path.exists(save_dir):
        # create and load the index
        index = VectorStoreIndex.from_documents(
            [documents], service_context=sentence_context
        )
        index.storage_context.persist(persist_dir=save_dir)
  else:
      # load the existing index
      index = load_index_from_storage(
          StorageContext.from_defaults(persist_dir=save_dir),
          service_context=sentence_context,
      )

  return index


# get the vector index
vector_index = get_build_index(documents=documents, llm=llm, embed_model="local:BAAI/bge-small-en-v1.5", sentence_window_size=3, save_dir="./vector_store/index")


from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank

def get_query_engine(sentence_index, similarity_top_k=6, rerank_top_n=2):
  postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
  rerank = SentenceTransformerRerank(
      top_n=rerank_top_n, model="BAAI/bge-reranker-base"
  )
  engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
  )

  return engine

query_engine = get_query_engine(sentence_index=vector_index, similarity_top_k=6, rerank_top_n=2)

while True:
  query=input()
  response = query_engine.query(query)
  display_response(response)
  print("\n")