from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import bentoml
import jwt
import torch
from bentoml.exceptions import NotFound
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.inference.run_text_inference import predict_single_text
from src.serving.schemas import (
    BatchTextPredictionRequest,
    Credentials,
    HealthResponse,
    TextPredictionRequest,
    TextPredictionResponse,
)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_TAG = "rakuten_text_classifier:latest"
DEFAULT_PREPROCESSING_CONFIG_PATH = BASE_DIR / "configs/text_preprocessing_config.yaml"
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
                return JSONResponse(
                    status_code=401, content={"detail": "Missing authentication token"}
                )
            try:
                scheme, token = authorization_header.split(" ", maxsplit=1)
                if scheme.lower() != "bearer":
                    raise ValueError("Unsupported authorization scheme")
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse(
                    status_code=401, content={"detail": "Token has expired"}
                )
            except (ValueError, jwt.InvalidTokenError):
                return JSONResponse(
                    status_code=401, content={"detail": "Invalid token"}
                )
            request.state.user = payload.get("sub")
        return await call_next(request)


def create_jwt_token(user_id: str, expires_in_hours: int = 1) -> str:
    expiration = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)
    payload = {"sub": user_id, "exp": expiration}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def _check_model_ready() -> bool:
    if not DEFAULT_PREPROCESSING_CONFIG_PATH.exists():
        return False
    try:
        bentoml.models.get(DEFAULT_MODEL_TAG)
    except NotFound:
        return False
    return True


def _load_registered_pytorch_model(model_ref: bentoml.Model, device: torch.device):
    previous_env = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    try:
        bert_cls = None
        try:
            from transformers.models.bert.modeling_bert import (
                BertForSequenceClassification,
            )

            bert_cls = BertForSequenceClassification
        except (ImportError, AttributeError, ModuleNotFoundError):
            bert_cls = None

        if bert_cls is not None and hasattr(torch.serialization, "safe_globals"):
            with torch.serialization.safe_globals([bert_cls]):
                model = bentoml.pytorch.load_model(model_ref)
        else:
            model = bentoml.pytorch.load_model(model_ref)
    finally:
        if previous_env is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous_env

    model = model.to(device)
    model.eval()
    return model


@bentoml.service(name="rakuten_text_model_service")
class TextModelService:
    def __init__(self) -> None:
        self.preprocessing_config_path = DEFAULT_PREPROCESSING_CONFIG_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return

        self.model_ref = bentoml.models.get(DEFAULT_MODEL_TAG)
        self.model = _load_registered_pytorch_model(self.model_ref, self.device)

        custom_objects = dict(self.model_ref.custom_objects or {})
        self.tokenizer = custom_objects["tokenizer"]
        self.idx_to_label = custom_objects["idx_to_label"]
        self._loaded = True

    def is_ready(self) -> bool:
        return _check_model_ready()

    def _predict_one(self, input_data: TextPredictionRequest) -> dict[str, Any]:
        self._load()
        return predict_single_text(
            model=self.model,
            text_input={
                "designation": input_data.designation,
                "description": input_data.description,
            },
            tokenizer=self.tokenizer,
            preprocessing_config_path=self.preprocessing_config_path,
            device=self.device,
            idx_to_label=self.idx_to_label,
            top_k=input_data.top_k,
        )

    @bentoml.api
    def predict_text(self, input_data: TextPredictionRequest) -> dict[str, Any]:
        return self._predict_one(input_data)

    @bentoml.api
    def predict_texts(self, items: list[TextPredictionRequest]) -> list[dict[str, Any]]:
        self._load()
        return [self._predict_one(item) for item in items]


@bentoml.service(name="rakuten_text_service")
class TextBentoService:
    model_service = bentoml.depends(TextModelService)

    @bentoml.api(route="/health")
    def health(self) -> HealthResponse:
        ready_value = self.model_service.is_ready()
        ready = ready_value if isinstance(ready_value, bool) else _check_model_ready()
        return HealthResponse(
            status="ok" if ready else "degraded",
            model_ready=ready,
            model_tag=DEFAULT_MODEL_TAG,
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
    def predict_batch(
        self, input_data: BatchTextPredictionRequest
    ) -> list[TextPredictionResponse]:
        results = self.model_service.predict_texts(input_data.items)
        return [TextPredictionResponse.model_validate(result) for result in results]


TextBentoService.add_asgi_middleware(JWTAuthMiddleware)
