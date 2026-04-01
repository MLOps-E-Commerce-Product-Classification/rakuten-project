import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

bentoml = pytest.importorskip("bentoml")
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("jwt")


def _model_registered() -> bool:
    try:
        bentoml.models.get("rakuten_text_classifier:latest")
    except Exception:
        return False
    return True


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_ready(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (URLError, ConnectionError, HTTPError) as exc:
            last_error = exc
            time.sleep(0.5)
    raise AssertionError(f"Bento readiness endpoint did not become ready: {last_error}")


def _login_token(port: int) -> str:
    request = Request(
        url=f"http://127.0.0.1:{port}/login",
        data=json.dumps({"credentials": {"username": "user123", "password": "password123"}}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=15) as response:
        body = json.loads(response.read().decode("utf-8"))
    assert response.status == 200
    return body["token"]


def test_bento_http_health_endpoint():
    cli = shutil.which("bentoml")
    if cli is None:
        pytest.skip("bentoml CLI is not installed in the test environment")

    port = _free_port()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    process = subprocess.Popen(
        [cli, "serve", "src.serving.bento_service:TextBentoService", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )

    try:
        _wait_for_ready(f"http://127.0.0.1:{port}/readyz")

        request = Request(
            url=f"http://127.0.0.1:{port}/health",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=10) as response:
            body = json.loads(response.read().decode("utf-8"))

        assert response.status == 200
        assert body["status"] in {"ok", "degraded"}
        assert body["model_tag"] == "rakuten_text_classifier:latest"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_bento_http_predict_endpoint_when_model_available():
    cli = shutil.which("bentoml")
    if cli is None:
        pytest.skip("bentoml CLI is not installed in the test environment")
    if not Path("models/text_backbone/config.json").exists():
        pytest.skip("Local Hugging Face backbone assets are not available")
    if not Path("models/best_text_model.pt").exists():
        pytest.skip("Model weights are not available")
    if not _model_registered():
        pytest.skip("BentoML model rakuten_text_classifier:latest is not registered")

    port = _free_port()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    process = subprocess.Popen(
        [cli, "serve", "src.serving.bento_service:TextBentoService", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )

    try:
        _wait_for_ready(f"http://127.0.0.1:{port}/readyz")
        token = _login_token(port)
        request = Request(
            url=f"http://127.0.0.1:{port}/predict",
            data=json.dumps({"input_data": {"designation": "robe femme", "description": "bleu", "top_k": 1}}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            method="POST",
        )
        with urlopen(request, timeout=15) as response:
            body = json.loads(response.read().decode("utf-8"))
        assert response.status == 200
        assert "predicted_rakuten_code" in body
        assert "top_k_predictions" in body
        assert "probabilities" in body
        assert all(isinstance(key, str) for key in body["probabilities"].keys())
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_bento_http_predict_batch_endpoint_when_model_available():
    cli = shutil.which("bentoml")
    if cli is None:
        pytest.skip("bentoml CLI is not installed in the test environment")
    if not Path("models/text_backbone/config.json").exists():
        pytest.skip("Local Hugging Face backbone assets are not available")
    if not Path("models/best_text_model.pt").exists():
        pytest.skip("Model weights are not available")
    if not _model_registered():
        pytest.skip("BentoML model rakuten_text_classifier:latest is not registered")

    port = _free_port()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    process = subprocess.Popen(
        [cli, "serve", "src.serving.bento_service:TextBentoService", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )

    try:
        _wait_for_ready(f"http://127.0.0.1:{port}/readyz")
        token = _login_token(port)
        request = Request(
            url=f"http://127.0.0.1:{port}/predict_batch",
            data=json.dumps({
                "input_data": {
                    "items": [
                        {"designation": "robe femme", "description": "bleu", "top_k": 1},
                        {"designation": "jeu video", "description": "ps4", "top_k": 2},
                    ]
                }
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            method="POST",
        )
        with urlopen(request, timeout=45) as response:
            body = json.loads(response.read().decode("utf-8"))
        assert response.status == 200
        assert isinstance(body, list)
        assert len(body) == 2
        assert len(body[0]["top_k_predictions"]) == 1
        assert len(body[1]["top_k_predictions"]) == 2
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
