def generate_sentence_embeddings(embedding_model, sentences: list) -> dict:
    """
    Generates embeddings for a list of sentences using the SentenceTransformer model.

    Parameters:
    sentences (list): A list of sentences to be embedded.
    model_name_or_path (str): The name or path of the SentenceTransformer model to use.
    device (str): The device to load the model to (e.g., 'cpu' or 'cuda').

    Returns:
    dict: A dictionary where each sentence is mapped to its corresponding embedding.
    """

    # Encode/Embed the sentences
    embeddings = embedding_model.encode(sentences)

    # Create a dictionary mapping sentences to their embeddings
    embeddings_dict = dict(zip(sentences, embeddings))

    return embeddings_dict

if __name__ == "__main__":

    from sentence_transformers import SentenceTransformer

    # Usage example
    sentences = [
        "The Sentence Transformers library provides an easy and open-source way to create embeddings.",
        "Sentences can be embedded one by one or as a list of strings.",
        "Embeddings are one of the most powerful concepts in machine learning!",
        "Learn to use embeddings well and you'll be well on your way to being an AI engineer."
    ]

    model_name_or_path=None

    device=None

    embedding_model = SentenceTransformer(model_name_or_path=model_name_or_path, device=device)

    embeddings_dict = generate_sentence_embeddings(embedding_model, sentences)

    # See the embeddings
    for sentence, embedding in embeddings_dict.items():
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")