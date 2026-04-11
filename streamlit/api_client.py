"""API client for communicating with the Rakuten backend."""

import time
import requests
from requests.auth import HTTPBasicAuth
from settings_manager import load_config


class RakutenAPIClient:
    """Client for the Rakuten prediction API."""

    def __init__(self):
        self._jwt_token: str | None = None
        self._token_time: float = 0.0
        self._token_ttl: float = 3500  # refresh before 1h expiry

    def _cfg(self) -> dict:
        return load_config().get("api", {})

    def _base_url(self) -> str:
        return self._cfg().get("base_url", "").rstrip("/")

    def _timeout(self) -> int:
        return self._cfg().get("timeout_seconds", 30)

    def _nginx_auth(self) -> HTTPBasicAuth:
        cfg = self._cfg()
        return HTTPBasicAuth(cfg.get("nginx_user", ""), cfg.get("nginx_pass", ""))

    def _ensure_token(self) -> str:
        """Get a valid JWT token, refreshing if needed."""
        if self._jwt_token and (time.time() - self._token_time) < self._token_ttl:
            return self._jwt_token
        self._jwt_token = self.login()
        self._token_time = time.time()
        return self._jwt_token

    def _auth_headers(self) -> dict:
        token = self._ensure_token()
        headers = {"Authorization": f"Bearer {token}"}
        api_key = self._cfg().get("api_key", "").strip()
        if api_key:
            headers["X-API-Key"] = api_key
        else:
            import streamlit as st
            st.warning("API Key ist nicht konfiguriert. Bitte in den Admin-Einstellungen hinterlegen.")
        return headers

    def health_check(self) -> dict:
        """Check backend health."""
        url = f"{self._base_url()}/health"
        try:
            resp = requests.post(
                url,
                auth=self._nginx_auth(),
                timeout=self._timeout(),
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            return {"status": "error", "detail": str(e)}

    def login(self) -> str:
        """Authenticate and return JWT token."""
        cfg = self._cfg()
        url = f"{self._base_url()}/login"
        payload = {
            "credentials": {
                "username": cfg.get("bento_user", ""),
                "password": cfg.get("bento_pass", ""),
            }
        }
        try:
            resp = requests.post(
                url,
                json=payload,
                auth=self._nginx_auth(),
                timeout=self._timeout(),
            )
            resp.raise_for_status()
            data = resp.json()
            token = data.get("token", "")
            if not token:
                raise ValueError("Kein Token in der Antwort erhalten.")
            return token
        except requests.HTTPError as e:
            raise ConnectionError(f"Login fehlgeschlagen (HTTP {e.response.status_code}): {e.response.text}") from e
        except requests.RequestException as e:
            raise ConnectionError(f"Verbindungsfehler beim Login: {e}") from e

    def predict_single(self, designation: str, description: str = "", top_k: int = 5) -> dict:
        """Run a single prediction."""
        url = f"{self._base_url()}/predict"
        payload = {
            "input_data": {
                "designation": designation,
                "description": description,
                "top_k": top_k,
            }
        }
        try:
            resp = requests.post(
                url,
                json=payload,
                headers=self._auth_headers(),
                auth=self._nginx_auth(),
                timeout=self._timeout(),
            )
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            if e.response.status_code == 401:
                # Token expired, retry once
                self._jwt_token = None
                resp = requests.post(
                    url,
                    json=payload,
                    headers=self._auth_headers(),
                    auth=self._nginx_auth(),
                    timeout=self._timeout(),
                )
                resp.raise_for_status()
                return resp.json()
            raise ConnectionError(f"Vorhersage fehlgeschlagen (HTTP {e.response.status_code}): {e.response.text}") from e
        except requests.RequestException as e:
            raise ConnectionError(f"Verbindungsfehler bei Vorhersage: {e}") from e

    def predict_batch(self, items: list[dict]) -> list[dict]:
        """Run batch prediction. Items: list of {designation, description, top_k}."""
        url = f"{self._base_url()}/predict_batch"
        payload = {"input_data": {"items": items}}
        try:
            resp = requests.post(
                url,
                json=payload,
                headers=self._auth_headers(),
                auth=self._nginx_auth(),
                timeout=max(self._timeout(), 120),
            )
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            if e.response.status_code == 401:
                self._jwt_token = None
                resp = requests.post(
                    url,
                    json=payload,
                    headers=self._auth_headers(),
                    auth=self._nginx_auth(),
                    timeout=max(self._timeout(), 120),
                )
                resp.raise_for_status()
                return resp.json()
            raise ConnectionError(f"Batch-Vorhersage fehlgeschlagen (HTTP {e.response.status_code}): {e.response.text}") from e
        except requests.RequestException as e:
            raise ConnectionError(f"Verbindungsfehler bei Batch-Vorhersage: {e}") from e


def get_client() -> RakutenAPIClient:
    """Get or create a cached API client instance."""
    import streamlit as st
    if "api_client" not in st.session_state:
        st.session_state["api_client"] = RakutenAPIClient()
    return st.session_state["api_client"]
