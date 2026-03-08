import re

def load_cranfield_documents(filepath):
    documents = []

    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    docs = content.split(".I ")[1:]

    for doc in docs:
        parts = doc.split(".W")
        if len(parts) > 1:
            documents.append(parts[1].strip())

    return documents


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()