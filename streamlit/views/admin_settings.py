"""Admin Settings page for editing config.yaml."""

import bcrypt
import streamlit as st
from auth import require_role
from settings_manager import load_config, save_config


def render():
    """Render the admin settings page."""
    st.header("Settings")
    if not require_role("admin"):
        return

    cfg = load_config(force=True)

    with st.form("settings_form"):
        st.subheader("API Settings")
        api = cfg.get("api", {})
        base_url = st.text_input("Base URL", value=api.get("base_url", ""))
        api_key = st.text_input("API Key", value=api.get("api_key", ""))
        nginx_user = st.text_input("Nginx User", value=api.get("nginx_user", ""))
        nginx_pass = st.text_input(
            "Nginx Password", value=api.get("nginx_pass", ""), type="password"
        )
        bento_user = st.text_input("Bento User", value=api.get("bento_user", ""))
        bento_pass = st.text_input(
            "Bento Password", value=api.get("bento_pass", ""), type="password"
        )
        timeout = st.number_input(
            "Timeout (seconds)",
            value=api.get("timeout_seconds", 30),
            min_value=5,
            max_value=300,
        )

        st.subheader("Prediction Settings")
        pred = cfg.get("prediction", {})
        default_top_k = st.number_input(
            "Default Top-K",
            value=pred.get("default_top_k", 5),
            min_value=1,
            max_value=27,
        )
        max_top_k = st.number_input(
            "Max Top-K", value=pred.get("max_top_k", 27), min_value=1, max_value=27
        )
        batch_limit = st.number_input(
            "Batch Limit",
            value=pred.get("batch_limit", 100),
            min_value=1,
            max_value=1000,
        )

        st.subheader("Monitoring")
        mon = cfg.get("monitoring", {})
        grafana_url = st.text_input("Grafana URL", value=mon.get("grafana_url", ""))
        prometheus_url = st.text_input(
            "Prometheus URL", value=mon.get("prometheus_url", "")
        )
        refresh_interval = st.number_input(
            "Refresh Interval (seconds)",
            value=mon.get("refresh_interval_seconds", 30),
            min_value=5,
        )

        st.subheader("App Settings")
        app = cfg.get("app", {})
        app_title = st.text_input("App Title", value=app.get("title", ""))
        session_timeout = st.number_input(
            "Session Timeout (minutes)",
            value=app.get("session_timeout_minutes", 60),
            min_value=5,
        )
        max_csv_mb = st.number_input(
            "Max CSV Upload (MB)",
            value=app.get("max_csv_upload_mb", 10),
            min_value=1,
            max_value=100,
        )

        st.subheader("Paths")
        paths = cfg.get("paths", {})
        corr_path = st.text_input(
            "Corrections CSV", value=paths.get("corrections_csv", "")
        )
        demo_path = st.text_input(
            "Demo Selections CSV", value=paths.get("demo_selections_csv", "")
        )
        logs_path = st.text_input("Logs Path", value=paths.get("logs", ""))

        submitted = st.form_submit_button("Save Settings")

    if submitted:
        cfg = load_config(force=True)
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
        st.success("Settings saved.")

    st.divider()
    st.subheader("User Management")
    users = cfg.get("users", {})

    for uname, udata in users.items():
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.text(f"{uname} ({udata.get('role', 'user')})")
        with col2:
            disabled = udata.get("disabled", False)
            st.text("Disabled" if disabled else "Active")
        with col3:
            if uname != "admin":
                if disabled:
                    if st.button("Enable", key=f"enable_{uname}"):
                        cfg["users"][uname]["disabled"] = False
                        save_config(cfg)
                        st.rerun()
                else:
                    if st.button("Disable", key=f"disable_{uname}"):
                        cfg["users"][uname]["disabled"] = True
                        save_config(cfg)
                        st.rerun()

    st.subheader("Change Password")
    with st.form("change_pw_form"):
        pw_user = st.selectbox("User", list(users.keys()))
        new_pw = st.text_input("New Password", type="password")
        pw_submit = st.form_submit_button("Change Password")

    if pw_submit and new_pw:
        hashed = bcrypt.hashpw(new_pw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        cfg["users"][pw_user]["password_hash"] = hashed
        save_config(cfg)
        st.success(f"Password for {pw_user} changed.")

    st.subheader("Add New User")
    with st.form("add_user_form"):
        new_uname = st.text_input("Username")
        new_upw = st.text_input("Password", type="password", key="new_user_pw")
        new_role = st.selectbox("Role", ["user", "admin"])
        add_submit = st.form_submit_button("Create User")

    if add_submit:
        if not new_uname or not new_upw:
            st.error("Username and password are required.")
        elif new_uname in cfg.get("users", {}):
            st.error(f"User '{new_uname}' already exists.")
        else:
            hashed = bcrypt.hashpw(new_upw.encode("utf-8"), bcrypt.gensalt()).decode(
                "utf-8"
            )
            if "users" not in cfg:
                cfg["users"] = {}
            cfg["users"][new_uname] = {
                "password_hash": hashed,
                "role": new_role,
            }
            save_config(cfg)
            st.success(f"User '{new_uname}' created.")
