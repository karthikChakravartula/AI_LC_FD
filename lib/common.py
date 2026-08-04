import re

def simple_tokenizer(text):
    text = text.lower()  # Make everything lowercase
    cleaned_text = re.sub(r"[^\w\s\-_]", " ", text)  # Remove punctuation like periods/commas
    return cleaned_text.split()  # Chop into individual words