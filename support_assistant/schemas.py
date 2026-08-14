"""
Pydantic models for the /ask endpoint.

AskRequest  -> incoming request body: {"query": str}
AskResponse -> validated, structured output guaranteed on every response:
               answer (str), sources (list[str]), confidence (float 0-1)
"""

from typing import List

from pydantic import BaseModel, Field


# Stores the user's question sent to the API.
class AskRequest(BaseModel):
    query: str


# Defines and validates the structure of the API response.
class AskResponse(BaseModel):
    answer: str  # Stores the final answer given to the user.
    sources: List[str] = Field(default_factory=list)  # Stores the sources used for the answer.
    confidence: float = Field(ge=0.0, le=1.0)  # Stores confidence score between 0 and 1.