import argparse
import os
import re
from typing import List

import instructor
import nltk
import openai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationInfo, model_validator
from uptrain import CritiqueTone, EvalLLM, Evals, Settings

from utils import extract_data_from_pdf

nltk.download("punkt")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate question-answer pairs from a PDF file."
    )
    parser.add_argument("-path", "--pdf_path", type=str, help="Path to the PDF file")
    args = parser.parse_args()

    documents = extract_data_from_pdf(args.pdf_path)

    for doc in documents:
        response = generate_qa_pairs(doc["text"])  # send entire page text
        qa_pairs = []
        for q, a in zip(response.questions, response.answers):
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
    print(documents)
