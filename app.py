import streamlit as st
import faiss
import os
import numpy as np
from io import BytesIO
import requests

from docx import Document
from dotenv import load_dotenv
load_dotenv()
from PyPDF2 import PdfReader

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Use a free model
HF_MODEL = "google/flan-t5-large"

# Set environment variable
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


def process_input(input_type, input_data):
    if input_type == "Link":
        loader = WebBaseLoader(input_data)
        documents = loader.load()
        raw_text = "\n".join([doc.page_content for doc in documents])
    elif input_type == "PDF":
        pdf_reader = PdfReader(BytesIO(input_data.read()))
        raw_text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    elif input_type == "Text":
        raw_text = input_data
    elif input_type == "DOCX":
        doc = Document(BytesIO(input_data.read()))
        raw_text = "\n".join([para.text for para in doc.paragraphs])
    elif input_type == "TXT":
        raw_text = input_data.read().decode('utf-8')
    else:
        raise ValueError("Unsupported input type")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vectorstore = FAISS.from_texts(texts, embedding=embeddings)

    return vectorstore

def answer_question(vectorstore, query):
    try:
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        payload = {
            "inputs": f"Context: {context}\n\nQuestion: {query}",
            "parameters": {"temperature": 0.7, "max_new_tokens": 300}
        }

        headers = {
            "Authorization": f"Bearer {hf_token}"
        }

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and "generated_text" in result[0]:
                return result[0]["generated_text"]
            else:
                return str(result)
        else:
            return f"Error: {response.status_code} - {response.text}"

    except Exception as e:
        return f" Exception occurred: {str(e)}"

def main():
    st.set_page_config(page_title=" RAG Q&A with HuggingFace", layout="centered")
    st.title(" RAG based Assistant") 

    input_type = st.selectbox("Select Input Type", ["Link", "PDF", "Text", "DOCX", "TXT"])
    input_data = None

    if input_type == "Link":
        url = st.text_input("Enter a URL")
        input_data = url
    elif input_type == "Text":
        input_data = st.text_area("Enter text here")
    else:
        input_data = st.file_uploader("Upload file", type=["pdf", "txt", "docx"])

    if st.button("Process Document"):
        if input_data:
            with st.spinner("Processing..."):
                try:
                    vectorstore = process_input(input_type, input_data)
                    st.session_state.vectorstore = vectorstore
                    st.success("✅ Document processed and indexed!")
                except Exception as e:
                    st.error(f"Failed to process: {e}")
        else:
            st.warning("Please upload or enter content.")

    if "vectorstore" in st.session_state:
        query = st.text_input("Ask a question:")
        if st.button("Submit Question"):
            with st.spinner("Thinking..."):
                answer = answer_question(st.session_state.vectorstore, query)
                st.markdown(f"### Answer:\n{answer}")

if __name__ == "__main__":
    main()
