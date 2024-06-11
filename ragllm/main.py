from sourcing import download_pdf_if_not_exists
from preprocess import open_and_read_pdf, process_and_chunk_texts, generate_sentence_embeddings, save_embeddings_to_file
import random
import pandas as pd
from sentence_transformers import SentenceTransformer

pdf_path = "human-nutrition-text.pdf"

url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"

download_pdf_if_not_exists(pdf_path, url)

pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)

num_sentence_chunk_size = 10

min_token_length = 30

df = process_and_chunk_texts(pages_and_texts, num_sentence_chunk_size, min_token_length)

# Generate embeddings for the filtered chunks
text_chunks = df["sentence_chunk"].tolist()

# Load the SentenceTransformer model
model_name_or_path = "all-mpnet-base-v2"

device = "cpu"

embedding_model = SentenceTransformer(model_name_or_path=model_name_or_path, device=device)

embeddings_dict = generate_sentence_embeddings(embedding_model, text_chunks)

# Add embeddings to the DataFrame
df["embedding"] = df["sentence_chunk"].map(embeddings_dict)

# Save the DataFrame to a CSV file
embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
save_embeddings_to_file(df, embeddings_df_save_path)
#

