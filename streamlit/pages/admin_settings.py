"""Admin Settings page for editing config.yaml."""

import bcrypt
import streamlit as st
from auth import require_role, get_current_user
from settings_manager import load_config, save_config


def render():
    """Render the admin settings page."""
    st.header("Einstellungen")
    if not require_role("admin"):
        return

    cfg = load_config(force=True)

    with st.form("settings_form"):
        # API Settings
        st.subheader("API-Einstellungen")
        api = cfg.get("api", {})
        base_url = st.text_input("Base URL", value=api.get("base_url", ""))
        api_key = st.text_input("API Key", value=api.get("api_key", ""))
        nginx_user = st.text_input("Nginx User", value=api.get("nginx_user", ""))
        nginx_pass = st.text_input("Nginx Passwort", value=api.get("nginx_pass", ""), type="password")
        bento_user = st.text_input("Bento User", value=api.get("bento_user", ""))
        bento_pass = st.text_input("Bento Passwort", value=api.get("bento_pass", ""), type="password")
        timeout = st.number_input("Timeout (Sekunden)", value=api.get("timeout_seconds", 30), min_value=5, max_value=300)

        # Prediction Settings
        st.subheader("Vorhersage-Einstellungen")
        pred = cfg.get("prediction", {})
        default_top_k = st.number_input("Default Top-K", value=pred.get("default_top_k", 5), min_value=1, max_value=27)
        max_top_k = st.number_input("Max Top-K", value=pred.get("max_top_k", 27), min_value=1, max_value=27)
        batch_limit = st.number_input("Batch-Limit", value=pred.get("batch_limit", 100), min_value=1, max_value=1000)

        # Monitoring
        st.subheader("Monitoring")
        mon = cfg.get("monitoring", {})
        grafana_url = st.text_input("Grafana URL", value=mon.get("grafana_url", ""))
        prometheus_url = st.text_input("Prometheus URL", value=mon.get("prometheus_url", ""))
        refresh_interval = st.number_input("Refresh-Interval (Sekunden)", value=mon.get("refresh_interval_seconds", 30), min_value=5)

        # App Settings
        st.subheader("App-Einstellungen")
        app = cfg.get("app", {})
        app_title = st.text_input("App-Titel", value=app.get("title", ""))
        session_timeout = st.number_input("Session-Timeout (Minuten)", value=app.get("session_timeout_minutes", 60), min_value=5)
        max_csv_mb = st.number_input("Max CSV-Upload (MB)", value=app.get("max_csv_upload_mb", 10), min_value=1, max_value=100)

        # Paths
        st.subheader("Pfade")
        paths = cfg.get("paths", {})
        corr_path = st.text_input("Corrections CSV", value=paths.get("corrections_csv", ""))
        demo_path = st.text_input("Demo Selections CSV", value=paths.get("demo_selections_csv", ""))
        logs_path = st.text_input("Logs-Pfad", value=paths.get("logs", ""))

        submitted = st.form_submit_button("Einstellungen speichern")

    if submitted:
        cfg["api"] = {
            "base_url": base_url,
            "api_key": api_key,
            "nginx_user": nginx_user,
            "nginx_pass": nginx_pass,
            "bento_user": bento_user,
            "bento_pass": bento_pass,
            "timeout_seconds": int(timeout),
        }
        cfg["prediction"] = {
            "default_top_k": int(default_top_k),
            "max_top_k": int(max_top_k),
            "batch_limit": int(batch_limit),
        }
        cfg["monitoring"] = {
            "grafana_url": grafana_url,
            "prometheus_url": prometheus_url,
            "refresh_interval_seconds": int(refresh_interval),
        }
        cfg["app"] = {
            "title": app_title,
            "session_timeout_minutes": int(session_timeout),
            "max_csv_upload_mb": int(max_csv_mb),
        }
        cfg["paths"] = {
            "corrections_csv": corr_path,
            "demo_selections_csv": demo_path,
            "logs": logs_path,
        }
        save_config(cfg)
        st.success("Einstellungen gespeichert.")

    # User Management
    st.divider()
    st.subheader("Benutzerverwaltung")
    users = cfg.get("users", {})

    # Display existing users
    for uname, udata in users.items():
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.text(f"{uname} ({udata.get('role', 'user')})")
        with col2:
            disabled = udata.get("disabled", False)
            status = "Deaktiviert" if disabled else "Aktiv"
            st.text(status)
        with col3:
            if uname != "admin":  # Don't allow disabling admin
                if disabled:
                    if st.button(f"Aktivieren", key=f"enable_{uname}"):
                        cfg["users"][uname]["disabled"] = False
                        save_config(cfg)
                        st.rerun()
                else:
                    if st.button(f"Deaktivieren", key=f"disable_{uname}"):
                        cfg["users"][uname]["disabled"] = True
                        save_config(cfg)
                        st.rerun()

    # Change password
    st.subheader("Passwort aendern")
    with st.form("change_pw_form"):
        pw_user = st.selectbox("Benutzer", list(users.keys()))
        new_pw = st.text_input("Neues Passwort", type="password")
        pw_submit = st.form_submit_button("Passwort aendern")

    if pw_submit and new_pw:
        hashed = bcrypt.hashpw(new_pw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        cfg["users"][pw_user]["password_hash"] = hashed
        save_config(cfg)
        st.success(f"Passwort fuer {pw_user} geaendert.")

    # Add new user
    st.subheader("Neuen Benutzer anlegen")
    with st.form("add_user_form"):
        new_uname = st.text_input("Benutzername")
        new_upw = st.text_input("Passwort", type="password", key="new_user_pw")
        new_role = st.selectbox("Rolle", ["user", "admin"])
        add_submit = st.form_submit_button("Benutzer anlegen")

    if add_submit:
        if not new_uname or not new_upw:
            st.error("Benutzername und Passwort sind Pflichtfelder.")
        elif new_uname in cfg.get("users", {}):
            st.error(f"Benutzer '{new_uname}' existiert bereits.")
        else:
            hashed = bcrypt.hashpw(new_upw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            if "users" not in cfg:
                cfg["users"] = {}
            cfg["users"][new_uname] = {
                "password_hash": hashed,
                "role": new_role,
            }
            save_config(cfg)
            st.success(f"Benutzer '{new_uname}' angelegt.")
