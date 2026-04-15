"""Monitoring page with Prometheus metrics and Grafana iframe."""

import requests
import streamlit as st
from settings_manager import load_config


def render():
    """Render the monitoring page."""
    st.header("Monitoring")
    cfg = load_config()
    mon_cfg = cfg.get("monitoring", {})
    api_cfg = cfg.get("api", {})

    prometheus_url = mon_cfg.get("prometheus_url", "")
    grafana_url = mon_cfg.get("grafana_url", "")

    # Prometheus Metrics
    st.subheader("Prometheus Metriken")
    if prometheus_url:
        try:
            resp = requests.get(
                prometheus_url,
                auth=(api_cfg.get("nginx_user", ""), api_cfg.get("nginx_pass", "")),
                timeout=10,
            )
            if resp.status_code == 200:
                metrics_text = resp.text
                # Show first 5000 chars in a code block
                if len(metrics_text) > 5000:
                    st.code(metrics_text[:5000] + "\n\n... (gekuerzt)")
                else:
                    st.code(metrics_text)
            else:
                st.warning(f"Prometheus nicht erreichbar (HTTP {resp.status_code}).")
        except requests.RequestException as e:
            st.warning(f"Prometheus nicht erreichbar: {e}")
    else:
        st.info("Keine Prometheus-URL konfiguriert.")

    # Grafana Dashboard
    st.subheader("Grafana Dashboard")
    if grafana_url:
        try:
            st.components.v1.iframe(grafana_url, height=600, scrolling=True)
        except Exception as e:
            st.warning(f"Grafana konnte nicht geladen werden: {e}")
    else:
        st.info("Keine Grafana-URL konfiguriert. URL kann in den Admin-Einstellungen hinterlegt werden.")

    # Health Check
    st.subheader("Backend-Status")
    if st.button("Health-Check ausfuehren"):
        from api_client import get_client
        client = get_client()
        result = client.health_check()
        if result.get("status") == "error":
            st.error(f"Backend nicht erreichbar: {result.get('detail', 'Unbekannter Fehler')}")
        else:
            st.success("Backend erreichbar.")
            st.json(result)
