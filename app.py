import streamlit as st
from llama_cpp import Llama
import fitz  # PyMuPDF
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from collections import Counter
import re

# --- Load LLM ---
MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
llm = Llama(model_path=MODEL_PATH, n_ctx=2048)

# --- Load Sentence Transformer ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Streamlit Setup ---
st.set_page_config(page_title="PDF QA Chatbot", layout="wide")
st.title(" PDF ChatBot (RAG with TinyLlama)")

# --- Upload PDF ---
uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_pdf:
    with st.spinner(" Reading PDF..."):
        doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
        text_chunks = []
        for page in doc:
            text_chunks.append(page.get_text())
        full_text = " ".join(text_chunks)

    # --- Show extracted keywords ---
    def extract_keywords(text, top_n=10):
        words = re.findall(r'\b\w+\b', text.lower())
        stopwords = set([
            "the", "and", "for", "are", "that", "this", "with", "you", "your",
            "from", "was", "have", "not", "but", "all", "can", "has", "had", 
            "pdf", "they", "will", "their", "which", "been", "use", "using", "also"
        ])
        filtered_words = [word for word in words if word not in stopwords and len(word) > 3]
        common = Counter(filtered_words).most_common(top_n)
        return [f"{word} ({count})" for word, count in common]

    st.subheader(" Top Keywords in PDF")
    keywords = extract_keywords(full_text)
    st.write(", ".join(keywords))

    # --- Split into chunks ---
    def chunk_text(text, max_tokens=200):
        words = text.split()
        return [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

    chunks = chunk_text(full_text)

    # --- Embedding + FAISS indexing ---
    with st.spinner(" Embedding and indexing..."):
        embeddings = embedder.encode(chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

    st.success(" PDF processed and ready!")

    # --- Chat interface ---
    user_question = st.text_input(" Ask a question from the PDF:")
    if user_question:
        with st.spinner(" Finding best context..."):
            query_embedding = embedder.encode([user_question])
            D, I = index.search(np.array(query_embedding), k=3)
            context = "\n\n".join([chunks[i] for i in I[0]])

        # --- Prompt for TinyLlama ---
        prompt = f"""
You are a helpful assistant. Use the context below to answer the user's question.

### Context:
{context}

### Question:
{user_question}

### Answer:
"""

        with st.spinner(" Generating answer..."):
            response = llm(prompt, max_tokens=200, stop=["###"])
            answer = response["choices"][0]["text"].strip()
            st.markdown("**Answer:**")
            st.write(answer)
