from __future__ import annotations

from pydantic import BaseModel, Field


class Credentials(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_tag: str


class TextPredictionRequest(BaseModel):
    designation: str = Field(..., min_length=1)
    description: str = ""
    top_k: int = Field(default=5, ge=1, le=27)


class BatchTextPredictionRequest(BaseModel):
    items: list[TextPredictionRequest] = Field(..., min_length=1)


class PredictionCandidate(BaseModel):
    rakuten_code: int
    probability: float


class TextPredictionResponse(BaseModel):
    predicted_rakuten_code: int
    top_k_predictions: list[PredictionCandidate]
    probabilities: dict[str, float]