import re

import nltk
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

nltk.download("punkt")

encoder = SentenceTransformer("all-MiniLM-L6-v2")


def extract_data_from_pdf(pdf_path: str):
    """
    Extracts data from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        list: A list of dictionaries containing the extracted data. Each dictionary
              contains the following keys:
              - 'text': The extracted text from the page.
              - 'sentences': A list of sentences extracted from the page.
              - 'page_number': The page number of the extracted data.
    """
    documents = []
    reader = PdfReader(pdf_path)
    number_of_pages = len(reader.pages)
    for i in range(number_of_pages):
        page = reader.pages[i]
        text = page.extract_text()
        text = re.sub(r"\n+", " ", text)
        sentences = nltk.tokenize.sent_tokenize(text)
        # store more metatdata in documents if possible
        documents.append({"text": text, "sentences": sentences, "page_number": i + 1})
    return documents


def generate_embeddings(documents: str):
    """
    Generate embeddings for the given documents.

    Args:
        documents (str): A list of documents.

    Returns:
        vectors (list): A list of embeddings generated for the documents.
    """
    vectors = encoder.encode(
        [sent for doc in documents for sent in doc["sentences"]],
        batch_size=32,
        show_progress_bar=True,
    )
    return vectors
