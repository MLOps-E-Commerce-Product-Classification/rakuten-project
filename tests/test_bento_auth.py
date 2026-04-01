import asyncio
import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse

pytest.importorskip("bentoml")
pytest.importorskip("jwt")

from src.serving.bento_service import JWTAuthMiddleware, create_jwt_token


def _request(path: str, authorization: str | None = None) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if authorization is not None:
        headers.append((b"authorization", authorization.encode("utf-8")))
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": path,
        "headers": headers,
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 3000),
        "scheme": "http",
    }

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


def _dispatch(request: Request):
    middleware = JWTAuthMiddleware(app=lambda scope, receive, send: None)

    async def call_next(req: Request) -> JSONResponse:
        assert hasattr(req.state, "user") or req.url.path not in {"/predict", "/predict_batch"}
        return JSONResponse({"ok": True, "user": getattr(req.state, "user", None)})

    return asyncio.run(middleware.dispatch(request, call_next))


def test_jwt_middleware_rejects_missing_token():
    response = _dispatch(_request("/predict"))

    assert response.status_code == 401
    assert response.body == b'{"detail":"Missing authentication token"}'


def test_jwt_middleware_rejects_invalid_token():
    response = _dispatch(_request("/predict", authorization="Bearer invalid-token"))

    assert response.status_code == 401
    assert response.body == b'{"detail":"Invalid token"}'


def test_jwt_middleware_rejects_expired_token():
    expired_token = create_jwt_token("user123", expires_in_hours=-1)

    response = _dispatch(_request("/predict", authorization=f"Bearer {expired_token}"))

    assert response.status_code == 401
    assert response.body == b'{"detail":"Token has expired"}'


def test_jwt_middleware_accepts_valid_token():
    token = create_jwt_token("user123")

    response = _dispatch(_request("/predict", authorization=f"Bearer {token}"))

    assert response.status_code == 200
    assert response.body == b'{"ok":true,"user":"user123"}'


def test_jwt_middleware_skips_unprotected_routes():
    response = _dispatch(_request("/health"))

    assert response.status_code == 200
    assert response.body == b'{"ok":true,"user":null}'
