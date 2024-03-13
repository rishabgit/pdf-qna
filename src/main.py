from pypdf import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import os
import openai
from openai import OpenAI
from pydantic import Field, BaseModel, model_validator, ValidationInfo
from typing import List
import instructor
import re
from dotenv import load_dotenv

nltk.download("punkt")

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
client = instructor.patch(OpenAI())

encoder = SentenceTransformer("all-MiniLM-L6-v2")


def extract_data_from_pdf(pdf_path):
    documents = []
    reader = PdfReader(pdf_path)
    number_of_pages = len(reader.pages)
    for i in range(number_of_pages):
        page = reader.pages[i]
        text = page.extract_text()
        text = re.sub(r"\n+", " ", text)
        sentences = sent_tokenize(text)
        # store more metatdata in documents if possible
        documents.append({"text": text, "sentences": sentences, "page_number": i + 1})
    return documents


def generate_embeddings(documents):
    vectors = encoder.encode(
        [sent for doc in documents for sent in doc["sentences"]],
        batch_size=32,
        show_progress_bar=True,
    )
    return vectors


class Fact(BaseModel):
    fact: str = Field(...)
    substring_quote: List[str] = Field(...)

    @model_validator(mode="after")
    def validate_sources(self, info: ValidationInfo) -> "Fact":
        text_chunks = info.context.get("text_chunk", None)
        spans = list(self.get_spans(text_chunks))
        self.substring_quote = [
            text_chunks[span[0] : span[1]] for span in spans  # noqa: E203
        ]
        return self

    def get_spans(self, context):
        for quote in self.substring_quote:
            yield from self._get_span(quote, context)

    def _get_span(self, quote, context):
        for match in re.finditer(re.escape(quote), context):
            yield match.span()


class QuestionsAnswers(BaseModel):
    questions: List[str] = Field(description="The question generated from the context.")
    answers: List[Fact] = Field(description="The answer of the question.")

    @model_validator(mode="after")
    def validate_sources(self) -> "QuestionsAnswers":
        self.answers = [
            fact if len(fact.substring_quote) > 0 else None for fact in self.answers
        ]
        return self


def generate_qa_pairs(context):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a world class system that generates multiple distinctive question-answer pair based on the context.",  # noqa: E501
            },
            {
                "role": "user",
                "content": f"Use this context to generate the question-answer pairs: {context}",
            },
        ],
        model="gpt-3.5-turbo",
        response_model=QuestionsAnswers,
        max_tokens=200,
        n=1,
        validation_context={"text_chunk": context},
    )
    return response


documents = extract_data_from_pdf("netflix_cosine.pdf")
vectors = generate_embeddings(documents)
for doc in documents:
    response = generate_qa_pairs(doc["text"])  # send entire page text
    qa_pairs = []
    for q, a in zip(response.questions, response.answers):
        qa_pairs.append({"question": q, "answer": a.fact, "quotes": a.substring_quote})
    doc["qa_pairs"] = qa_pairs
