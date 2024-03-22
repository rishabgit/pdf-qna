import re
from typing import List

from pydantic import BaseModel, Field, ValidationInfo, model_validator


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
