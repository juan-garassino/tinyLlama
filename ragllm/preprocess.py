from tqdm.auto import tqdm 
from spacy.lang.en import English
import re
import pandas as pd
from sourcing import save_load_embeddings
from indexing import generate_sentence_embeddings

def process_and_chunk_texts(pages_and_texts: list, num_sentence_chunk_size: int, min_token_length: int) -> pd.DataFrame:
    """
    Processes a list of dictionaries to extract sentences, split them into chunks,
    and create separate items for each chunk with additional statistics.

    Parameters:
    pages_and_texts (list): A list of dictionaries, each containing a "text" field.
    num_sentence_chunk_size (int): The number of sentences per chunk.
    min_token_length (int): The minimum token length for filtering chunks.

    Returns:
    pd.DataFrame: A DataFrame containing the processed chunks that meet the minimum token length.
    """
    # Initialize the spaCy English model
    nlp = English()
    nlp.add_pipe("sentencizer")

    # Function to split a list into sublists of a specified size
    def split_list(input_list: list, slice_size: int) -> list[list[str]]:
        """
        Splits the input_list into sublists of size slice_size (or as close as possible).

        For example, a list of 17 sentences would be split into two lists of [[10], [7]]
        """
        return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

    pages_and_chunks = []

    # Iterate over each item in the input list with a progress bar
    for item in tqdm(pages_and_texts, desc="Processing texts"):
        # Extract sentences from the text field and convert them to strings
        sentences = list(nlp(item["text"]).sents)
        item["sentences"] = [str(sentence) for sentence in sentences]

        # Split sentences into chunks of the specified size
        sentence_chunks = split_list(input_list=item["sentences"], slice_size=num_sentence_chunk_size)

        # Create separate items for each chunk with additional statistics
        for sentence_chunk in sentence_chunks:
            chunk_dict = {
                "page_number": item.get("page_number", None),
                "sentence_chunk": re.sub(r'\.([A-Z])', r'. \1', "".join(sentence_chunk).replace("  ", " ").strip()),
            }
            chunk_dict["chunk_char_count"] = len(chunk_dict["sentence_chunk"])
            chunk_dict["chunk_word_count"] = len(chunk_dict["sentence_chunk"].split())
            chunk_dict["chunk_token_count"] = len(chunk_dict["sentence_chunk"]) // 4  # Approximation: 1 token = ~4 characters

            # Validate page number
            if chunk_dict["page_number"] is not None and chunk_dict["page_number"] < 0:
                print(f"Warning: Negative page number detected: {chunk_dict['page_number']}. Setting it to None.")
                chunk_dict["page_number"] = None

            # Filter chunks by minimum token length
            if chunk_dict["chunk_token_count"] >= min_token_length:
                pages_and_chunks.append(chunk_dict)

    # Convert the list of chunks to a DataFrame
    df = pd.DataFrame(pages_and_chunks)

    return df

if __name__ == "__main__":
    # Usage example
    pages_and_texts = [
        {"page_number": 1, "text": "This is a sentence. This is another sentence. This is a third sentence. And yet another one. This is the fifth sentence. Here is the sixth. Seventh sentence here. Eighth sentence here. Ninth sentence. Tenth sentence. Eleventh sentence. Twelfth sentence."},
        {"page_number": 2, "text": "Here is one more sentence. And another one. Here's a third. Fourth one here. Fifth sentence here. Sixth one. Seventh sentence. Eighth. Ninth sentence. Tenth. Eleventh sentence. Twelfth sentence."}
    ]
    num_sentence_chunk_size = 10
    min_token_length = 30
    df = process_and_chunk_texts(pages_and_texts, num_sentence_chunk_size, min_token_length)

    # Generate embeddings for the filtered chunks
    text_chunks = df["sentence_chunk"].tolist()
    embeddings_dict = generate_sentence_embeddings(text_chunks)

    # Add embeddings to the DataFrame
    df["embedding"] = df["sentence_chunk"].map(embeddings_dict)

    # Save the DataFrame to a CSV file
    embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
    save_load_embeddings(df, embeddings_df_save_path)


    # Display random chunks with under 30 tokens in length
    for row in df.iterrows():
        print(f'Chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["sentence_chunk"]}')

