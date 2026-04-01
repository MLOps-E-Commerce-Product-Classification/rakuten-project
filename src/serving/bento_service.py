from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import bentoml
import jwt
import pandas as pd
from bentoml.exceptions import NotFound
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.serving.mlflow_bento import resolve_bento_model_reference
from src.serving.schemas import (
    BatchTextPredictionRequest,
    Credentials,
    HealthResponse,
    TextPredictionRequest,
    TextPredictionResponse,
)

JWT_SECRET_KEY = "rakuten_text_service_secret_key_2026"
JWT_ALGORITHM = "HS256"
USERS = {
    "user123": "password123",
    "user456": "password456",
}


class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        protected_paths = {"/predict", "/predict_batch"}
        if request.url.path in protected_paths:
            authorization_header = request.headers.get("Authorization")
            if not authorization_header:
                return JSONResponse(status_code=401, content={"detail": "Missing authentication token"})
            try:
                scheme, token = authorization_header.split(" ", maxsplit=1)
                if scheme.lower() != "bearer":
                    raise ValueError("Unsupported authorization scheme")
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token has expired"})
            except (ValueError, jwt.InvalidTokenError):
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})
            request.state.user = payload.get("sub")
        return await call_next(request)


def create_jwt_token(user_id: str, expires_in_hours: int = 1) -> str:
    expiration = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)
    payload = {"sub": user_id, "exp": expiration}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def _resolved_model_reference() -> dict[str, Any]:
    return resolve_bento_model_reference()


def _check_model_ready() -> bool:
    model_reference = _resolved_model_reference()
    try:
        bentoml.models.get(model_reference["model_tag"])
    except NotFound:
        return False
    return True


def _coerce_prediction_rows(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, pd.DataFrame):
        return result.to_dict(orient="records")
    if isinstance(result, dict):
        return [result]
    if hasattr(result, "to_dict"):
        converted = result.to_dict()
        if isinstance(converted, list):
            return converted
    return list(result)


@bentoml.service(name="rakuten_text_model_service")
class TextModelService:
    def __init__(self) -> None:
        self._loaded = False
        self.model_reference = _resolved_model_reference()

    def _load(self) -> None:
        if self._loaded:
            return

        self.model_reference = _resolved_model_reference()
        self.model_ref = bentoml.models.get(self.model_reference["model_tag"])
        self.model = bentoml.mlflow.load_model(self.model_ref)
        self._loaded = True

    def is_ready(self) -> bool:
        return _check_model_ready()

    def _predict_rows(self, items: list[TextPredictionRequest]) -> list[dict[str, Any]]:
        self._load()
        input_frame = pd.DataFrame(
            [
                {
                    "designation": item.designation,
                    "description": item.description,
                    "top_k": item.top_k,
                }
                for item in items
            ]
        )
        rows = _coerce_prediction_rows(self.model.predict(input_frame))
        return [dict(row) for row in rows]

    @bentoml.api
    def predict_text(self, input_data: TextPredictionRequest) -> dict[str, Any]:
        return self._predict_rows([input_data])[0]

    @bentoml.api
    def predict_texts(self, items: list[TextPredictionRequest]) -> list[dict[str, Any]]:
        return self._predict_rows(items)


@bentoml.service(name="rakuten_text_service")
class TextBentoService:
    model_service = bentoml.depends(TextModelService)

    @bentoml.api(route="/health")
    def health(self) -> HealthResponse:
        ready_value = self.model_service.is_ready()
        ready = ready_value if isinstance(ready_value, bool) else _check_model_ready()
        model_reference = _resolved_model_reference()
        return HealthResponse(
            status="ok" if ready else "degraded",
            model_ready=ready,
            model_tag=model_reference["model_tag"],
            mlflow_model_name=model_reference.get("mlflow_model_name"),
            mlflow_alias=model_reference.get("mlflow_alias"),
            mlflow_version=model_reference.get("mlflow_version"),
            mlflow_run_id=model_reference.get("mlflow_run_id"),
            validation_status=model_reference.get("validation_status"),
        )

    @bentoml.api(route="/login")
    def login(self, credentials: Credentials) -> dict[str, str]:
        if USERS.get(credentials.username) == credentials.password:
            token = create_jwt_token(credentials.username)
            return {"token": token}
        return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})

    @bentoml.api(route="/predict")
    def predict(self, input_data: TextPredictionRequest) -> TextPredictionResponse:
        result = self.model_service.predict_text(input_data)
        return TextPredictionResponse.model_validate(result)

    @bentoml.api(route="/predict_batch")
    def predict_batch(self, input_data: BatchTextPredictionRequest) -> list[TextPredictionResponse]:
        results = self.model_service.predict_texts(input_data.items)
        return [TextPredictionResponse.model_validate(result) for result in results]


TextBentoService.add_asgi_middleware(JWTAuthMiddleware)