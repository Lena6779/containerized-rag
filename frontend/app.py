# Streamlit chat interface for the RAG API
# Everything goes through the backend; it never calls Ollama or ChromaDB directly
import os

import requests
import streamlit as st

# Set to http://backend:8000 in docker-compose.yml
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Assistant", page_icon="📚")
st.title("📚 RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources):
    """Show each source's file, chunk, distance, and text in an expander."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, source in enumerate(sources, start=1):
            st.markdown(
                f"**[{i}] {source['source']}**, chunk {source['chunk_index']} (distance: {source['distance']})"
            )
            st.write(source["text"])


# --- Sidebar: status, stats, ingest, clear chat ---
with st.sidebar:
    st.header("System Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        stats = requests.get(f"{API_URL}/stats", timeout=5).json()
        st.success("Backend: connected")
        if health["ollama"] == "connected":
            st.success("Ollama: connected")
        else:
            st.warning("Ollama: unavailable")
        col1, col2 = st.columns(2)
        col1.metric("Documents", stats["document_count"])
        col2.metric("Chunks", stats["chunk_count"])
    except requests.exceptions.RequestException:
        st.error(f"Can't reach backend at {API_URL}")

    st.divider()
    # Long timeout because the first run downloads the embedding model
    if st.button("🔄 Re-index Documents", use_container_width=True):
        with st.spinner("Re-indexing documents..."):
            try:
                response = requests.post(f"{API_URL}/ingest", timeout=300)
                response.raise_for_status()
                data = response.json()
                st.success(
                    f"Indexed {data['chunks']} chunks from {len(data['files'])} files"
                )
            except requests.exceptions.RequestException as error:
                st.error(f"Ingest failed: {error}")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Main: Chat history and new questions ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("confidence"):
            st.caption(f"confidence: {message['confidence']}")
        show_sources(message.get("sources"))

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Long timeout because the LLM can be slow on CPU
                response = requests.post(
                    f"{API_URL}/ask", json={"question": question}, timeout=180
                )
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as error:
                st.error(f"Request failed: {error}")
                st.stop()

        st.write(data["answer"])
        st.caption(f"confidence: {data['confidence']}")
        show_sources(data["sources"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": data["answer"],
            "sources": data["sources"],
            "confidence": data["confidence"],
        }
    )
