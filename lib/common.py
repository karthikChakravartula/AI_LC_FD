import re
from chromadb.utils import embedding_functions

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

def simple_tokenizer(text):
    text = text.lower()  # Make everything lowercase
    cleaned_text = re.sub(r"[^\w\s\-_]", " ", text)  # Remove punctuation like periods/commas
    return cleaned_text.split()  # Chop into individual words