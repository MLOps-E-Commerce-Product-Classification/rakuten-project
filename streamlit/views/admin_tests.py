"""Admin Unit Tests page."""

import os
import subprocess
import streamlit as st
from auth import require_role


def _resolve_tests_dir() -> str:
    """Resolve tests directory: works both in Docker (/app/tests) and locally."""
    docker_path = "/app/tests"
    if os.path.isdir(docker_path):
        return docker_path
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        candidate = os.path.join(here, "tests")
        if os.path.isdir(candidate):
            return candidate
        here = os.path.dirname(here)
    return docker_path


TESTS_DIR = _resolve_tests_dir()


def _find_test_files(tests_dir: str) -> list[str]:
    """Find all test files in the tests directory."""
    if not os.path.isdir(tests_dir):
        return []
    test_files = []
    for root, _, files in os.walk(tests_dir):
        for f in files:
            if f.startswith("test_") and f.endswith(".py"):
                rel_path = os.path.relpath(os.path.join(root, f), tests_dir)
                test_files.append(rel_path)
    return sorted(test_files)


def render():
    """Render the admin unit tests page."""
    st.header("Unit Tests")
    if not require_role("admin"):
        return

    if not os.path.isdir(TESTS_DIR):
        st.warning(f"Tests directory '{TESTS_DIR}' not found. "
                   "Make sure the directory is available in the Docker container.")
        return

    test_files = _find_test_files(TESTS_DIR)

    if not test_files:
        st.info("No test files found.")
        return

    st.subheader("Available Tests")

    selected_tests = []
    for tf in test_files:
        if st.checkbox(tf, value=True, key=f"test_cb_{tf}"):
            selected_tests.append(tf)

    col1, col2 = st.columns(2)
    with col1:
        run_button = st.button("Run Tests", key="run_tests")
    with col2:
        st.text(f"{len(selected_tests)} of {len(test_files)} selected")

    if run_button:
        if not selected_tests:
            st.warning("No tests selected.")
            return

        test_paths = [os.path.join(TESTS_DIR, t) for t in selected_tests]

        st.subheader("Test Output")
        output_area = st.empty()

        cmd = ["python", "-m", "pytest", "-v", "--tb=short"] + test_paths

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd="/app",
            )

            full_output = result.stdout
            if result.stderr:
                full_output += "\n--- STDERR ---\n" + result.stderr

            output_area.code(full_output)

            if result.returncode == 0:
                st.success("All tests passed.")
            else:
                st.error(f"Tests failed (exit code: {result.returncode}).")

        except subprocess.TimeoutExpired:
            st.error("Test run aborted (timeout: 300 seconds).")
        except Exception as e:
            st.error(f"Error running tests: {e}")
