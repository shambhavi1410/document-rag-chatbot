import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from loaders import SUPPORTED, load_file
from rag import answer, build_index, get_embeddings, label, retrieve

load_dotenv()
st.set_page_config(page_title="Document Chatbot", page_icon="📄")


@st.cache_resource(show_spinner="Loading embedding model...")
def embeddings():
    return get_embeddings()


st.title("Chat with your documents")
st.caption("Upload PDF, DOCX, PPTX, PNG or JPG files, build the knowledge base, then ask questions.")

with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload files", type=SUPPORTED, accept_multiple_files=True)
    if st.button("Build knowledge base", disabled=not files):
        docs, failed = [], []
        with st.spinner("Reading, chunking and embedding..."):
            for f in files:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix)
                tmp.write(f.getbuffer())
                tmp.close()
                try:
                    docs += load_file(tmp.name, f.name)
                except Exception as e:
                    failed.append(f"{f.name}: {e}")
                finally:
                    os.unlink(tmp.name)
            if docs:
                st.session_state.index, n = build_index(docs, embeddings())
                st.session_state.stats = (len(files) - len(failed), n)
                st.session_state.messages = []
        for msg in failed:
            st.error(msg)
        if not docs:
            st.error("No readable text found in the uploaded files.")
    if "stats" in st.session_state:
        st.success(f"{st.session_state.stats[0]} file(s), {st.session_state.stats[1]} chunks indexed")

st.session_state.setdefault("messages", [])
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(f"**{s[0]}**\n\n{s[1]}")

if q := st.chat_input("Ask about your documents", disabled="index" not in st.session_state):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            hits = retrieve(st.session_state.index, q)
            reply = answer(q, hits)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply,
                                      "sources": [(label(d), d.page_content[:300] + "...") for d in hits]})
    st.rerun()
if "index" not in st.session_state:
    st.info("Upload files in the sidebar and click Build knowledge base to start.")
