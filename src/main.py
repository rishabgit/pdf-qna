import argparse
import logging
import os
import re
from typing import List

import instructor
import openai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationInfo, model_validator
from tqdm import tqdm
from uptrain import CritiqueTone, EvalLLM, Evals, Settings

from src.utils import extract_data_from_pdf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = "gpt-3.5-turbo"

openai.api_key = OPENAI_API_KEY
client = instructor.patch(OpenAI())

settings = Settings(model=OPENAI_MODEL, openai_api_key=OPENAI_API_KEY)
eval_llm = EvalLLM(settings)


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
                "content": "You are a world class system that generates as many as possible distinctive question-answer pair based on the context.",  # noqa: E501
            },
            {
                "role": "user",
                "content": f"Use this context to generate the question-answer pairs: {context}",
            },
        ],
        model=OPENAI_MODEL,
        response_model=QuestionsAnswers,
        max_tokens=200,
        n=1,
        validation_context={"text_chunk": context},
    )
    logger.info(
        f"OAI {OPENAI_MODEL} usage for generation: {response._raw_response.usage}."
    )
    return response


def evals(question, answer, context):
    data = [
        {
            "question": question,
            "response": answer,
            "context": context,
        }
    ]
    persona = "Responds to questions with precise, factual answers, omitting any extraneous or informal content."
    res = eval_llm.evaluate(
        data=data,
        checks=[
            Evals.RESPONSE_COMPLETENESS_WRT_CONTEXT,
            CritiqueTone(llm_persona=persona),
            Evals.FACTUAL_ACCURACY,
            Evals.CRITIQUE_LANGUAGE,
        ],
    )[0]
    res.pop("question", None)
    res.pop("context", None)
    res.pop("response", None)
    return res


def run(pdf_path):
    """
    Extracts data from a PDF file and generates question-answer pairs for each page.

    This function processes each page of the PDF, extracting the text content and generating
    question-answer pairs using the OpenAI API. It performs evaluation on each question-answer
    pair, assessing the quality and accuracy of the generated responses.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a page in the PDF.
              Each dictionary contains the following keys:
              - 'text' (str): The extracted text content of the page.
              - 'sentences' (list): The extracted text split into sentences.
              - 'page_number' (int): Page number.
              - 'qa_pairs' (list): A list of question-answer pairs generated for the page.
                                   Each question-answer pair is represented as a dictionary
                                   with the following keys:
                                   - 'question' (str): The generated question.
                                   - 'answer' (str): The generated answer.
                                   - 'quotes' (list): The substring quotes from the page text
                                                      that support the answer, used for citation.
                                   - 'evals' (dict): The evaluation scores and reasoning for the
                                                     question-answer pair. Current evals:
                                                     - context utilization : Score measuring
                                                       how complete the answer is given the context.
                                                     - factual accuracy: Score indicating
                                                       whether the answer is factually correct based
                                                       on the context.
                                                     - language features: Score assessing
                                                       the quality and effectiveness of language in
                                                       the answer.
                                                     - tonality: Score evaluating whether
                                                       the answer matches the required persona's tone.
    """
    documents = extract_data_from_pdf(pdf_path)
    logger.info(f"Extracted data from PDF. Total pages: {len(documents)}.")

    for page_i, doc in enumerate(documents):
        response = generate_qa_pairs(doc["text"])  # send entire page text
        logger.info(
            f"Generated QnAs for page {page_i+1}.\nCount of Qs: "
            f"{len(response.questions)} & As: {len(response.answers)}."
        )
        qa_pairs = []
        logger.info("Running evals...")
        for q, a in tqdm(zip(response.questions, response.answers)):
            evals_res = evals(q, a.fact, doc["text"])
            qa_pairs.append(
                {
                    "question": q,
                    "answer": a.fact,
                    "quotes": a.substring_quote,
                    "evals": evals_res,
                }
            )
        doc["qa_pairs"] = qa_pairs
    return documents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate question-answer pairs from a PDF file."
    )
    parser.add_argument(
        "-path",
        "--pdf_path",
        type=str,
        default="data/The-Emperors-New-Clothes.pdf",
        help="Path to the PDF file",
    )
    args = parser.parse_args()

    documents = run(args.pdf_path)
    print(documents)
