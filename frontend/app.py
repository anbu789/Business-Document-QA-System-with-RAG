"""
frontend/app.py
----------------
Streamlit chat interface for the Business Document QA System.

Access is gated behind a login (streamlit-authenticator) using a small,
hardcoded list of usernames in auth_config.yaml — appropriate for "let a
handful of named people in", not a general-purpose signup system.

Layout once logged in:
  - Left:  list of the 4 source documents. Click one to preview its content
           (table for CSVs, extracted text for the PDF) in a modal, with a
           download button.
  - Right: the chat interface, talking to the FastAPI backend over HTTP.

This file contains NO retrieval or generation logic — it's a thin UI layer.
All RAG pipeline work happens server-side in api/main.py.
"""

import os
import yaml
import requests
import pandas as pd
import pdfplumber
import streamlit as st
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader

API_URL = os.getenv("API_URL", "http://localhost:8000")
DATA_DIR = os.getenv("DATA_DIR", "data")
AUTH_CONFIG_PATH = os.getenv("AUTH_CONFIG_PATH", "auth_config.yaml")

st.set_page_config(page_title="Business Document QA", page_icon="📄", layout="wide")


# ── Authentication gate ────────────────────────────────────────────────────
# Loads the hardcoded credentials list and shows a login form. Nothing below
# this block renders until authentication_status is True.

def load_auth_config():
    """
    Load auth credentials — from Streamlit's secrets manager when deployed
    on Streamlit Community Cloud, or from the local auth_config.yaml file
    when running locally.

    This matters because Streamlit Community Cloud requires a public
    GitHub repo, and auth_config.yaml (real usernames + password hashes)
    is intentionally gitignored and never committed. On Community Cloud,
    the same credentials are pasted into the app's Secrets manager instead
    — never into the repo itself.
    """
    if "credentials" in st.secrets:
        return {
            "credentials": {
                "usernames": {
                    username: dict(details)
                    for username, details in st.secrets["credentials"]["usernames"].items()
                }
            },
            "cookie": dict(st.secrets["cookie"]),
        }

    if os.path.exists(AUTH_CONFIG_PATH):
        with open(AUTH_CONFIG_PATH) as f:
            return yaml.load(f, Loader=SafeLoader)

    st.error(
        f"No auth config found. Locally: copy `auth_config.yaml` into the "
        f"project root. On Streamlit Community Cloud: add your credentials "
        f"under the app's Secrets settings."
    )
    st.stop()


auth_config = load_auth_config()

authenticator = stauth.Authenticate(
    auth_config["credentials"],
    auth_config["cookie"]["name"],
    auth_config["cookie"]["key"],
    auth_config["cookie"]["expiry_days"],
)

authenticator.login(location="main")

if st.session_state.get("authentication_status") is False:
    st.error("Username or password is incorrect.")
    st.stop()
elif st.session_state.get("authentication_status") is None:
    st.warning("Please enter your username and password.")
    st.stop()

# ── Everything below only runs for an authenticated user ──────────────────

# ── Document registry ─────────────────────────────────────────────────────
# Maps to the 4 sample files under data/ (see PHASE2_COMPLETE.md).
DOCUMENTS = [
    {
        "filename": "service_contract_nexus_solutions.pdf",
        "label": "📄 Service Contract — Nexus Solutions",
        "type": "pdf",
    },
    {
        "filename": "invoices.csv",
        "label": "🧾 Invoices",
        "type": "csv",
    },
    {
        "filename": "employees.csv",
        "label": "👥 Employees",
        "type": "csv",
    },
    {
        "filename": "sales_report_q1_q2_2024.csv",
        "label": "📈 Sales Report (Q1–Q2 2024)",
        "type": "csv",
    },
]


# ── Document preview modal ────────────────────────────────────────────────

def show_document(doc: dict):
    """Open a modal previewing the given document's content."""
    path = os.path.join(DATA_DIR, doc["filename"])

    @st.dialog(doc["label"], width="large")
    def _dialog():
        if not os.path.exists(path):
            st.error(
                f"Could not find `{path}`. Make sure Streamlit is running "
                f"from the project root, so the `{DATA_DIR}/` folder resolves."
            )
            return

        if doc["type"] == "csv":
            df = pd.read_csv(path)
            st.caption(f"{len(df)} rows")
            st.dataframe(df, use_container_width=True)

        elif doc["type"] == "pdf":
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    st.markdown(f"**Page {i}**")
                    text = page.extract_text() or "*(no extractable text on this page)*"
                    st.text(text)
                    if i < len(pdf.pages):
                        st.divider()

        with open(path, "rb") as f:
            st.download_button(
                label="Download original file",
                data=f.read(),
                file_name=doc["filename"],
                use_container_width=True,
            )

    _dialog()


# ── Layout: document panel (left) + chat (right) ──────────────────────────

doc_col, chat_col = st.columns([1, 2.5], gap="large")

with doc_col:
    st.subheader("📁 Source Documents")
    st.caption("Click a document to preview its content.")
    for doc in DOCUMENTS:
        if st.button(doc["label"], use_container_width=True, key=f"doc_{doc['filename']}"):
            show_document(doc)

    st.divider()
    st.subheader("About")
    st.markdown(
        "This assistant answers questions grounded in these 4 documents "
        "using hybrid retrieval + reranking + a scope guard + Gemini."
    )
    st.caption(
        "Questions you type are sent to Google's Gemini API to generate "
        "answers. No data is stored beyond your current session."
    )
    st.markdown(f"**API:** `{API_URL}`")
    st.markdown(f"**Logged in as:** `{st.session_state.get('username')}`")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    authenticator.logout("Log out", location="main")

with chat_col:
    st.title("📄 Business Document QA")
    st.caption(
        "Ask questions about invoices, service contracts, employee records, "
        "and sales reports."
    )

    # ── Session state: chat history ────────────────────────────────────
    # Streamlit reruns the entire script top-to-bottom on every interaction.
    # Without session_state, chat history would reset on every message.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    def render_meta(meta: dict):
        """Render pipeline details (scope decision, rerank score, sources)
        for one assistant message, tucked into a collapsed expander."""
        with st.expander("Pipeline details"):
            badge = "✅ In scope" if meta["in_scope"] else "🚫 Out of scope"
            st.markdown(f"**{badge}** — `{meta['scope_reason']}`")
            if meta.get("rerank_score") is not None:
                st.markdown(f"**Top rerank score:** `{meta['rerank_score']:.4f}`")
            if meta.get("sources_used"):
                st.markdown("**Sources cited:**")
                for src in meta["sources_used"]:
                    st.markdown(f"- {src}")

    # ── Render existing chat history ───────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                render_meta(msg["meta"])

    # ── Handle new input ────────────────────────────────────────────────
    if question := st.chat_input("Ask a question about the business documents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query",
                        json={"question": question},
                        timeout=60,
                    )
                    response.raise_for_status()
                    data = response.json()

                    st.markdown(data["answer"])
                    meta = {
                        "in_scope": data["in_scope"],
                        "scope_reason": data["scope_reason"],
                        "rerank_score": data.get("rerank_score"),
                        "sources_used": data.get("sources_used", []),
                    }
                    render_meta(meta)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "meta": meta,
                    })

                except requests.exceptions.ConnectionError:
                    error_msg = (
                        f"Could not reach the API at `{API_URL}`. "
                        "Make sure the FastAPI backend is running "
                        "(`uvicorn api.main:app --reload`)."
                    )
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

                except requests.exceptions.HTTPError as e:
                    error_msg = f"API error: {e.response.status_code} — {e.response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

                except requests.exceptions.Timeout:
                    error_msg = "The request timed out. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
