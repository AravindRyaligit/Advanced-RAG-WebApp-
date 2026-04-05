import os

import streamlit as st
import requests

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Agent - PDF Q&A",
    page_icon="📄",
    layout="wide",
)

# ── Session state init ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Sidebar: Upload & Document Management ────────────────────────────────────
with st.sidebar:
    st.header("Document Management")

    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF to index it for Q&A.",
    )

    if uploaded_file is not None:
        if st.button("Index Document", use_container_width=True):
            with st.spinner("Processing PDF..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(
                            f"Indexed **{data['filename']}**\n\n"
                            f"Doc ID: `{data['doc_id']}`\n\n"
                            f"Chunks: {data['chunk_count']}"
                        )
                    else:
                        st.error(f"Upload failed: {resp.json().get('detail', resp.text)}")
                except requests.ConnectionError:
                    st.error("Cannot connect to backend. Is the FastAPI server running on port 8000?")

    st.divider()
    st.subheader("Indexed Documents")

    if st.button("Refresh List", use_container_width=True):
        st.session_state.pop("doc_list_cache", None)

    try:
        if "doc_list_cache" not in st.session_state:
            resp = requests.get(f"{API_BASE}/documents", timeout=10)
            if resp.status_code == 200:
                st.session_state.doc_list_cache = resp.json().get("documents", [])
            else:
                st.session_state.doc_list_cache = []

        docs = st.session_state.doc_list_cache

        if docs:
            for doc in docs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(
                        f"**{doc['source']}**  \n"
                        f"`{doc['doc_id']}` | {doc['chunk_count']} chunks"
                    )
                with col2:
                    if st.button("Delete", key=f"del_{doc['doc_id']}"):
                        try:
                            del_resp = requests.delete(
                                f"{API_BASE}/documents/{doc['doc_id']}", timeout=10
                            )
                            if del_resp.status_code == 200:
                                st.success("Deleted!")
                                st.session_state.pop("doc_list_cache", None)
                                st.rerun()
                            else:
                                st.error("Delete failed.")
                        except requests.ConnectionError:
                            st.error("Backend not reachable.")
        else:
            st.info("No documents indexed yet. Upload a PDF to get started.")

    except requests.ConnectionError:
        st.warning("Backend not reachable. Start the FastAPI server first.")

    st.divider()
    st.caption("RAG Agent | Groq Llama 70B + BGE Embeddings + Reranker")


# ── Main: Chat Interface ────────────────────────────────────────────────────
st.title("RAG Agent - PDF Q&A")
st.markdown("Ask questions about your uploaded PDF documents.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("View Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**[{i}] {src['source']}** (Page {src['page']})\n\n"
                        f">{src['content']}..."
                    )

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/query",
                    json={"question": prompt},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander("View Sources"):
                            for i, src in enumerate(sources, 1):
                                st.markdown(
                                    f"**[{i}] {src['source']}** (Page {src['page']})\n\n"
                                    f">{src['content']}..."
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                else:
                    error_msg = resp.json().get("detail", "Unknown error occurred.")
                    st.error(f"Error: {error_msg}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {error_msg}",
                    })
            except requests.ConnectionError:
                st.error("Cannot connect to backend. Is the FastAPI server running?")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Error: Cannot connect to the backend server.",
                })
