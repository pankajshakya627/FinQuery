#!/usr/bin/env python3
"""
Streamlit Web Interface for Minimal RAG
"""
import streamlit as st
from rag_module.rag import MinimalRAG
import tempfile
import os


@st.cache_resource
def get_rag_engine():
    return MinimalRAG()


def main():
    st.set_page_config(page_title="Minimal RAG", page_icon="🔍", layout="wide")
    st.title("🔍 Minimal RAG System")

    rag = get_rag_engine()

    tab1, tab2 = st.tabs(["📄 Upload Documents", "❓ Ask Questions"])

    with tab1:
        st.subheader("Upload Documents")
        uploaded_file = st.file_uploader("Upload PDF file", type="pdf")

        if uploaded_file is not None:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    # Save uploaded file to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Use LangChain loader + Document Aware Semantic Chunking
                        chunks = rag.load_and_chunk_document(tmp_path)
                        st.success(f"✅ Document processed! Added {chunks} semantic chunks.")
                    finally:
                        os.unlink(tmp_path)

        st.info("Uses Document Aware Semantic Chunking. Supports: PDF, TXT files.")

    with tab2:
        st.subheader("Query Documents")
        question = st.text_input("Enter your question:")

        if question:
            if st.button("Get Answer", type="primary"):
                with st.spinner("Generating answer..."):
                    answer = rag.query(question)
                    st.subheader("Answer:")
                    st.write(answer)

    with st.sidebar:
        st.subheader("⚙️ Configuration")
        st.info(f"Active Provider: **{rag.settings.llm_provider.upper()}**")
        st.info(f"Model: **{rag.settings.llm_model}**")
        st.info(f"Collection: **{rag.settings.chroma_collection_name}**")
        st.info(f"Chunking: Table-aware + Hybrid RRF")
        
        if st.button("🗑️ Clear Cache & Restart"):
            st.cache_resource.clear()
            st.rerun()


if __name__ == "__main__":
    main()
