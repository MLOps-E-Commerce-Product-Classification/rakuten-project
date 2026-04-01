import pytest

pytest.importorskip("bentoml")
pytest.importorskip("jwt")
pytest.importorskip("torch")

from src.serving.bento_service import TextBentoService, create_jwt_token
from src.serving.schemas import BatchTextPredictionRequest, Credentials, TextPredictionRequest


class StubModelService:
    def __init__(self, ready: bool = True):
        self.ready = ready

    def is_ready(self):
        return self.ready

    def predict_text(self, payload):
        return {
            "predicted_rakuten_code": 10,
            "top_k_predictions": [{"rakuten_code": 10, "probability": 0.91}],
            "probabilities": {"10": 0.91},
        }

    def predict_texts(self, payloads):
        return [
            {
                "predicted_rakuten_code": 10,
                "top_k_predictions": [{"rakuten_code": 10, "probability": 0.91}],
                "probabilities": {"10": 0.91},
            }
            for _ in payloads
        ]


def test_health_reports_degraded_when_model_store_is_not_ready():
    service = TextBentoService()
    service.model_service = StubModelService(ready=False)

    response = service.health()

    assert response.status == "degraded"
    assert response.model_ready is False
    assert response.model_tag == "rakuten_text_classifier:latest"


def test_login_returns_jwt_for_valid_credentials():
    service = TextBentoService()

    response = service.login(Credentials(username="user123", password="password123"))

    assert isinstance(response, dict)
    assert "token" in response
    assert isinstance(response["token"], str)


def test_login_rejects_invalid_credentials():
    service = TextBentoService()

    response = service.login(Credentials(username="user123", password="wrong"))

    assert response.status_code == 401


def test_predict_delegates_to_model_service():
    service = TextBentoService()
    service.model_service = StubModelService()

    response = service.predict(TextPredictionRequest(designation="robe femme", description="bleu", top_k=1))

    assert response.predicted_rakuten_code == 10
    assert response.top_k_predictions[0].rakuten_code == 10
    assert response.top_k_predictions[0].probability == 0.91
    assert response.probabilities["10"] == 0.91


def test_predict_batch_delegates_to_model_service():
    service = TextBentoService()
    service.model_service = StubModelService()

    request = BatchTextPredictionRequest(
        items=[
            TextPredictionRequest(designation="robe femme", description="bleu", top_k=1),
            TextPredictionRequest(designation="jeu video", description="ps4", top_k=1),
        ]
    )
    response = service.predict_batch(request)

    assert len(response) == 2
    assert response[0].predicted_rakuten_code == 10
    assert response[1].probabilities["10"] == 0.91


def test_create_jwt_token_returns_string():
    token = create_jwt_token("user123")
    assert isinstance(token, str)
