
from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str = Field(min_length=2, max_length=5000)


class BatchSentimentRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)
