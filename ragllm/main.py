from sourcing import download_pdf_if_not_exists, save_load_embeddings, open_and_read_document
from preprocess import process_and_chunk_texts
from indexing import generate_sentence_embeddings
#from retriving import print_top_results_and_scores, retrieve_relevant_resources
from utils import get_model_num_params, print_wrapped
from generating import ask_llm, ask_llm_streaming
from pretrained import initialize_model_and_tokenizer

import os
import numpy as np
import torch
from sentence_transformers import util, SentenceTransformer
import random

pdf_path = "human-nutrition-text.pdf"

url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"

#context = 'harry_potter_1'

context = 'human-nutrition-text'

epub_path = f"{context}.pdf"

model_name_or_path = "all-mpnet-base-v2"

device = "cuda"

model_id = "google/gemma-1.1-2b-it"

use_quantization_config = False #True

num_sentence_chunk_size = 10

min_token_length = 30

embeddings_df_save_path = f"{context}.csv"

embedding_model = SentenceTransformer(model_name_or_path=model_name_or_path, device=device)

# Nutrition-style questions generated with GPT4
gpt4_questions = [
    "What are the macronutrients, and what roles do they play in the human body?",
    "How do vitamins and minerals differ in their roles and importance for health?",
    "Describe the process of digestion and absorption of nutrients in the human body.",
    "What role does fibre play in digestion? Name five fibre containing foods.",
    "Explain the concept of energy balance and its importance in weight management."
]

# Manually created question list
manual_questions = [
    "How often should infants be breastfed?",
    "What are symptoms of pellagra?",
    "How does saliva help with digestion?",
    "What is the RDI for protein per day?",
    "water soluble vitamins"
]

query_list = gpt4_questions + manual_questions

# download_pdf_if_not_exists(pdf_path, url) # just if i do not have a source

if os.path.exists(embeddings_df_save_path):

    embeddings_df = None

    embeddings_df = save_load_embeddings(df=embeddings_df, file_name=context, overwrite=False)

    print("File loaded successfully.")

    # TODO this is not working use chroma

else:
    print(f"'{embeddings_df_save_path}' File does not exist.")

    pages_and_texts = open_and_read_document(file_path=epub_path)

    embeddings_df = process_and_chunk_texts(pages_and_texts, num_sentence_chunk_size, min_token_length)#.iloc[50:55]

    text_chunks = embeddings_df["sentence_chunk"].tolist()

    embeddings_dict = generate_sentence_embeddings(embedding_model, text_chunks)

    embeddings_df["embedding"] = embeddings_df["sentence_chunk"].map(embeddings_dict)

    save_load_embeddings(df=embeddings_df, file_name=context, overwrite=True)

embeddings = torch.tensor(np.array(embeddings_df["embedding"].tolist()), dtype=torch.float32).to(device)

pages_and_chunks_dict = embeddings_df.to_dict(orient="records")

tokenizer, llm_model = initialize_model_and_tokenizer(model_id, use_quantization_config, device)

query = random.choice(query_list)

print(query)

answer, context_items = ask_llm(query=query,
                            temperature=0.7,
                            max_new_tokens=512,
                            return_answer_only=False,
                            embedding_model=embedding_model,
                            pages_and_chunks=pages_and_chunks_dict,
                            llm_model=llm_model,
                            tokenizer=tokenizer,
                            embeddings=embeddings)

# # Call the function and get the generator
# streaming_generator = ask_llm_streaming(query,
#                                         temperature=0.7,
#                                         max_new_tokens=512,
#                                         format_answer_text=True,
#                                         return_answer_only=False,
#                                         tokenizer=tokenizer,
#                                         llm_model=llm_model,
#                                         embeddings=embeddings,
#                                         pages_and_chunks=pages_and_chunks_dict,
#                                         embedding_model=embedding_model)

# # Iterate over the generator and print each yielded value
# for value in streaming_generator:
#     print(value)

print(f"Query: {query}")
print(f"Answer:\n")
print_wrapped(answer)
print(f"Context items:")
context_items