import numpy as np

from src.utils import extract_data_from_pdf, generate_embeddings


def test_extract_data_from_pdf():
    # Prepare test data
    pdf_path = "data/netflix_cosine.pdf"
    expected_page_count = 9
    expected_text = "Is Cosine-Similarity of Embeddings Really About Similarity?"

    # Call the function
    documents = extract_data_from_pdf(pdf_path)

    # Assert the results
    assert len(documents) == expected_page_count
    assert expected_text in documents[0]["text"]
    assert expected_text in documents[0]["sentences"][0]
    assert len(documents[0]["sentences"]) > 1
    assert documents[0]["page_number"] == 1


def test_generate_embeddings():
    # Prepare test data
    documents = [
        {
            "text": "This is a sample document.",
            "sentences": ["This is a sample document."],
        },
        {
            "text": "Another example sentence.",
            "sentences": ["Another example sentence."],
        },
    ]
    expected_embedding_shape = (2, 384)

    # Call the function
    embeddings = generate_embeddings(documents)

    # Assert the results
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == expected_embedding_shape
