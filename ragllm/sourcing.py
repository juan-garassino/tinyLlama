import requests
from tqdm.auto import tqdm
import fitz
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
#from chromadb.storage import VectorDatabase

def download_pdf_if_not_exists(pdf_path: str, url: str) -> None:
    """
    Downloads a PDF file from a given URL if it doesn't already exist in the specified path.

    Parameters:
    pdf_path (str): The local path where the PDF file will be saved.
    url (str): The URL of the PDF file to download.

    Returns:
    None
    """
    # Check if the PDF file already exists
    if not os.path.exists(pdf_path):
        print("File doesn't exist, downloading...")

        try:
            # Send a GET request to the URL
            response = requests.get(url)
            # Check if the request was successful
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx and 5xx)

            # Open a file in binary write mode and save the content to it
            with open(pdf_path, "wb") as file:
                file.write(response.content)
            print(f"The file has been downloaded and saved as {pdf_path}")
        
        except requests.exceptions.RequestException as e:
            # Handle any exceptions that occur during the request
            print(f"Failed to download the file. Error: {e}")
    else:
        print(f"File {pdf_path} already exists.")

# def save_load_embeddings(df: pd.DataFrame, file_path: str, overwrite: bool = False) -> pd.DataFrame:
#     """
#     Saves the DataFrame columns that are not 'embedding' to a CSV file, the 'embedding' column to a numpy array file,
#     and the entire DataFrame to a ChromaDB for Vector Databases.
#     Loads both files and the ChromaDB if they exist and overwrite is False.

#     Parameters:
#     df (pd.DataFrame): The DataFrame containing the embeddings.
#     file_path (str): The base path where the files will be saved or loaded from (without extension).
#     overwrite (bool): Whether to overwrite the existing files and ChromaDB if they exist.

#     Returns:
#     pd.DataFrame: The DataFrame containing the embeddings, either loaded from files and ChromaDB or the provided one.
#     """
#     csv_file_path = f"{file_path}.csv"
#     npy_file_path = f"{file_path}_embedding.npy"
#     chromadb_path = f"{file_path}.chromadb"

#     if os.path.exists(csv_file_path) and os.path.exists(npy_file_path) and os.path.exists(chromadb_path):
#         if overwrite:
#             print(f"Overwriting existing files at {csv_file_path}, {npy_file_path}, and {chromadb_path}.")
#             df.drop(columns=['embedding']).to_csv(csv_file_path, index=False)
#             np.save(npy_file_path, df['embedding'].values)
#             chromadb = VectorDatabase(chromadb_path)
#             chromadb.clear()
#             chromadb.bulk_insert(df['embedding'].values)
#             print(f"Files saved at {csv_file_path}, {npy_file_path}, and {chromadb_path}.")
#         else:
#             print(f"Loading existing files from {csv_file_path}, {npy_file_path}, and {chromadb_path}.")
#             df_non_embedding = pd.read_csv(csv_file_path)
#             embeddings = np.load(npy_file_path, allow_pickle=True)
#             df = df_non_embedding
#             df['embedding'] = embeddings
#             chromadb = VectorDatabase(chromadb_path)
#             print(f"Files loaded from {csv_file_path}, {npy_file_path}, and {chromadb_path}.")
#     else:
#         print(f"Files do not exist, saving new files at {csv_file_path}, {npy_file_path}, and {chromadb_path}.")
#         df.drop(columns=['embedding']).to_csv(csv_file_path, index=False)
#         np.save(npy_file_path, df['embedding'].values)
#         chromadb = VectorDatabase(chromadb_path)
#         chromadb.bulk_insert(df['embedding'].values)
#         print(f"Files saved at {csv_file_path}, {npy_file_path}, and {chromadb_path}.")
    
#     return df

def save_load_embeddings(df: pd.DataFrame, file_name: str, overwrite: bool = False) -> pd.DataFrame:
    """
    Saves the DataFrame columns that are not 'embedding' to a CSV file and the 'embedding' column to a numpy array file.
    Loads both files and combines them into a single DataFrame if the files exist and overwrite is False.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the embeddings.
    file_path (str): The base path where the files will be saved or loaded from (without extension).
    overwrite (bool): Whether to overwrite the existing files if they exist.

    Returns:
    pd.DataFrame: The DataFrame containing the embeddings, either loaded from files or the provided one.
    """
    csv_file_path = f"{file_name}.csv"
    npy_file_path = f"{file_name}_embedding.npy"

    if os.path.exists(csv_file_path) and os.path.exists(npy_file_path):
        if overwrite:
            print(f"Overwriting existing files at {csv_file_path} and {npy_file_path}.")
            df.drop(columns=['embedding']).to_csv(csv_file_path, index=False)
            np.save(npy_file_path, df['embedding'].values)
            print(f"Files saved at {csv_file_path} and {npy_file_path}.")
        else:
            print(f"Loading existing files from {csv_file_path} and {npy_file_path}.")
            df_non_embedding = pd.read_csv(csv_file_path)
            embeddings = np.load(npy_file_path, allow_pickle=True)
            df = df_non_embedding
            df['embedding'] = embeddings
            print(f"Files loaded from {csv_file_path} and {npy_file_path}.")
    else:
        print(f"Files do not exist, saving new files at {csv_file_path} and {npy_file_path}.")
        df.drop(columns=['embedding']).to_csv(csv_file_path, index=False)
        np.save(npy_file_path, df['embedding'].values)
        print(f"Files saved at {csv_file_path} and {npy_file_path}.")
    
    return df

# def save_load_embeddings(df: pd.DataFrame, file_path: str, overwrite: bool = False) -> pd.DataFrame:
#     """
#     Saves the embeddings to a CSV file if the file does not exist or if overwrite is True.
#     Loads the embeddings from the CSV file if it exists and overwrite is False.

#     Parameters:
#     df (pd.DataFrame): The DataFrame containing the embeddings.
#     file_path (str): The path where the CSV file will be saved or loaded from.
#     overwrite (bool): Whether to overwrite the existing file if it exists.

#     Returns:
#     pd.DataFrame: The DataFrame containing the embeddings, either loaded from file or the provided one.
#     """
#     if os.path.exists(file_path):
#         if overwrite:
#             print(f"Overwriting existing file at {file_path}.")
#             df.to_csv(file_path, index=False)
#             print(f"File saved at {file_path}.")
#         else:
#             print(f"Loading existing file from {file_path}.")
#             df = pd.read_csv(file_path)
#             print(f"File loaded from {file_path}.")
#     else:
#         print(f"File does not exist, saving new file at {file_path}.")
#         df.to_csv(file_path, index=False)
#         print(f"File saved at {file_path}.")
    
#     return df

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip() # note: this might be different for each doc (best to experiment)

    # Other potential text formatting functions can go here
    return cleaned_text

# def open_and_read_pdf(pdf_path: str) -> list[dict]:
#     """
#     Opens a PDF file, reads its text content page by page, and collects statistics.

#     Parameters:
#         pdf_path (str): The file path to the PDF document to be opened and read.

#     Returns:
#         list[dict]: A list of dictionaries, each containing the page number
#         (adjusted), character count, word count, sentence count, token count, and the extracted text
#         for each page.
#     """
#     doc = fitz.open(pdf_path)  # open a document
#     pages_and_texts = []
#     for page_number, page in tqdm(enumerate(doc)):  # iterate the document pages
#         text = page.get_text()  # get plain text encoded as UTF-8
#         text = text_formatter(text)
#         pages_and_texts.append({"page_number": page_number - 41,  # adjust page numbers since our PDF starts on page 42
#                                 "page_char_count": len(text),
#                                 "page_word_count": len(text.split(" ")),
#                                 "page_sentence_count_raw": len(text.split(". ")),
#                                 "page_token_count": len(text) / 4,  # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
#                                 "text": text})
#     return pages_and_texts



def text_formatter(text):
    # A placeholder function, replace with actual formatting as needed.
    return text.strip()

def open_and_read_document(file_path: str) -> list[dict]:
    """
    Opens a PDF, EPUB, or TXT file, reads its text content page by page, and collects statistics.

    Parameters:
        file_path (str): The file path to the document to be opened and read.

    Returns:
        list[dict]: A list of dictionaries, each containing the page number
        (adjusted for PDF), character count, word count, sentence count, token count, and the extracted text
        for each page.
    """

    # TODO make it open all the files in that folder
    
    file_ext = os.path.splitext(file_path)[1].lower()
    pages_and_texts = []

    if file_ext == '.pdf':
        doc = fitz.open(file_path)  # open a PDF document
        for page_number, page in tqdm(enumerate(doc)):  # iterate the document pages
            text = page.get_text()  # get plain text encoded as UTF-8
            text = text_formatter(text)
            pages_and_texts.append({"page_number": page_number - 41,  # adjust page numbers since our PDF starts on page 42
                                    "page_char_count": len(text),
                                    "page_word_count": len(text.split()),
                                    "page_sentence_count_raw": len(text.split(". ")),
                                    "page_token_count": len(text) / 4,  # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                                    "text": text})

    elif file_ext == '.epub':
        book = epub.read_epub(file_path)
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text()
                text = text_formatter(text)
                pages_and_texts.append({"page_number": len(pages_and_texts),  # use sequential page numbering for EPUB
                                        "page_char_count": len(text),
                                        "page_word_count": len(text.split()),
                                        "page_sentence_count_raw": len(text.split(". ")),
                                        "page_token_count": len(text) / 4,  # 1 token = ~4 chars
                                        "text": text})

    elif file_ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            text = text_formatter(text)
            pages_and_texts.append({"page_number": 0,  # single page for TXT
                                    "page_char_count": len(text),
                                    "page_word_count": len(text.split()),
                                    "page_sentence_count_raw": len(text.split(". ")),
                                    "page_token_count": len(text) / 4,  # 1 token = ~4 chars
                                    "text": text})

    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

    return pages_and_texts
