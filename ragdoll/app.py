# The UI Layer
import streamlit as st
import os
from ragdoll import RAGdoll

st.set_page_config(page_title="RAGdoll", layout="wide")
st.title("🐶 RAGdoll: Your Personal File Assistant")

# Initialize Engine
if "rag" not in st.session_state:
    st.session_state.rag = RAGdoll()

# --- Sidebar ---
st.sidebar.header("📁 Knowledge Base")
folder_input = st.sidebar.text_area("Folder Paths (One per line):", "/Users/username/Documents")
auto_sync = st.sidebar.toggle("Enable Auto-Sync (Watchdog)", value=False)

paths = [p.strip() for p in folder_input.split("\n") if p.strip()]

# Watchdog Logic
if auto_sync:
    if st.session_state.rag.observer is None or not st.session_state.rag.observer.is_alive():
        st.session_state.rag.start_watchdog(paths)
    st.sidebar.caption("🟢 Live Monitoring Active")
else:
    if st.session_state.rag.observer:
        st.session_state.rag.observer.stop()
        st.session_state.rag.observer = None
    st.sidebar.caption("🔴 Live Monitoring Paused")

# Manual Sync Button
if st.sidebar.button("Manual Sync"):
    prog_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    def ui_callback(p, t):
        prog_bar.progress(p)
        status_text.text(t)

    with st.spinner("Processing documents..."):
        res = st.session_state.rag.ingest_docs(paths, progress_callback=ui_callback)
        st.sidebar.success(res)

    # Give the user a moment to see 100% before clearing
    import time
    time.sleep(1)
    prog_bar.empty()
    status_text.empty()

# --- Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask RAGdoll..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            resp = st.session_state.rag.get_chat_response(prompt)
            st.markdown(resp["answer"])

            # Use get() to prevent crashes if source_documents is missing
            sources = resp.get("source_documents", [])
            if sources:
                with st.expander("📚 Sources"):
                    for doc in sources:
                        # Clean display of source path
                        file_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                        st.write(f"- {file_name}")

    st.session_state.messages.append({"role": "assistant", "content": resp["answer"]})
