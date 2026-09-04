import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = os.getenv("API_PORT", "8000")
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"

# Page configuration
st.set_page_config(
    page_title="NTP RAG Chat Assistant",
    page_icon="🤖",
    layout="wide"
)

# Helper function to check API Health
def get_api_health():
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if res.status_code == 200:
            return True, res.json()
    except Exception:
        pass
    return False, None

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your AI assistant for the **National Transformation Program (NTP) Annual Report 2025**. Ask me anything about the report's goals, initiatives, and achievements!",
            "sources": []
        }
    ]

# Sidebar Controls
with st.sidebar:
    st.title("⚙️ Settings")

    is_online, health_data = get_api_health()
    if is_online:
        st.success(f"🟢 Backend Online (`{API_HOST}:{API_PORT}`)")
    else:
        st.error(f"🔴 Backend Offline (`{API_HOST}:{API_PORT}`)")

    st.subheader("🔑 OpenAI API Key")
    user_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Optional: Enter your OpenAI API key here. If left blank, the server default key from .env will be used."
    )

    st.subheader("Model Configuration")
    selected_model = st.selectbox(
        "OpenAI Model",
        options=["gpt-4o-mini", "gpt-4o"],
        index=0
    )

    top_k = st.slider(
        "Context Chunks (Top-K)",
        min_value=1,
        max_value=8,
        value=3,
        help="Number of relevant document chunks retrieved from FAISS."
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Controls creativity. Lower values produce more deterministic and grounded answers."
    )

    st.divider()
    st.subheader("📊 Knowledge Base")

    if health_data:
        st.metric("Total Chunks", health_data.get("total_chunks", 0))
        st.write(f"**Document:** `{health_data.get('document', 'ntp_report.pdf')}`")
        st.write(f"**Embeddings:** `{health_data.get('embedding_model', 'text-embedding-3-small')}`")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Chat history cleared. How can I help you with the NTP Annual Report 2025?",
                    "sources": []
                }
            ]
            st.rerun()

    with col2:
        if st.button("🔄 Reindex", use_container_width=True):
            with st.spinner("Rebuilding FAISS index..."):
                try:
                    res = requests.post(f"{API_BASE_URL}/reindex", timeout=10)
                    if res.status_code == 200:
                        st.success("Reindex initiated!")
                    else:
                        st.error("Failed to reindex.")
                except Exception as e:
                    st.error(f"Error: {e}")

# Main Chat Interface
st.title("🏛️ National Transformation Program Assistant")
st.caption("Retrieval-Augmented Generation (RAG) assistant powered by FastAPI and OpenAI")

# Suggested questions buttons if conversation just started
if len(st.session_state.messages) <= 1:
    st.write("##### 💡 Suggested Questions:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🎯 What is the primary aim of NTP?", use_container_width=True):
            st.session_state.pending_query = "What is the primary aim of the National Transformation Program?"
            st.rerun()
    with col2:
        if st.button("🚀 What are the key digital achievements?", use_container_width=True):
            st.session_state.pending_query = "What are the key digital transformation achievements in the report?"
            st.rerun()
    with col3:
        if st.button("🌱 How does NTP support sustainability?", use_container_width=True):
            st.session_state.pending_query = "How does NTP support environmental sustainability?"
            st.rerun()

pending_query = getattr(st.session_state, "pending_query", None)
if pending_query:
    st.session_state.pending_query = None

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Retrieved Context & Sources ({len(msg['sources'])} chunks)"):
                for i, src in enumerate(msg["sources"], 1):
                    score_pct = f"{src.get('score', 0):.2%}" if src.get('score') is not None else "N/A"
                    st.markdown(f"**Source {i}** — *Page {src.get('page', 'N/A')}* (Relevance: `{score_pct}`)")
                    st.info(src.get("text", ""))

# Handle User Input
user_input = st.chat_input("Ask a question about the NTP Annual Report 2025...") or pending_query

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating answer..."):
            history_payload = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
                if m["role"] in ["user", "assistant"]
            ]

            payload = {
                "message": user_input,
                "history": history_payload,
                "top_k": top_k,
                "model": selected_model,
                "temperature": temperature,
                "api_key": user_api_key.strip() if user_api_key else None
            }

            try:
                response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    answer_text = data["answer"]
                    sources = data.get("sources", [])

                    st.markdown(answer_text)

                    if sources:
                        with st.expander(f"📚 Retrieved Context & Sources ({len(sources)} chunks)", expanded=True):
                            for i, src in enumerate(sources, 1):
                                score_pct = f"{src.get('score', 0):.2%}" if src.get('score') is not None else "N/A"
                                st.markdown(f"**Source {i}** — *Page {src.get('page', 'N/A')}* (Relevance: `{score_pct}`)")
                                st.info(src.get("text", ""))

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources
                    })
                else:
                    try:
                        err_msg = response.json().get("detail", response.text)
                    except Exception:
                        err_msg = response.text
                    st.error(f"⚠️ {err_msg}")
            except requests.exceptions.ConnectionError:
                st.error(f"⚠️ Could not connect to FastAPI backend at `{API_BASE_URL}`. Make sure `python api.py` is running.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
