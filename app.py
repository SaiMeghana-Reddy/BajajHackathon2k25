# app.py
import streamlit as st
import requests
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-8b-8192"  # can use llama3-70b if needed

# ----------------------------
# PDF Text Extraction
# ----------------------------
def extract_pdf_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text.strip()

# ----------------------------
# Chunk text into smaller pieces
# ----------------------------
def chunk_text(text, chunk_size=800):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# ----------------------------
# Build FAISS index
# ----------------------------
def build_faiss_index(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings, model

# ----------------------------
# Retrieve top N relevant chunks
# ----------------------------
def retrieve_chunks(query, index, chunks, model, top_n=3):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec), top_n)
    return [chunks[i] for i in indices[0]]

# ----------------------------
# Query Groq API
# ----------------------------
def query_groq(context, user_query):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
Answer the user's query based only on the following context from the policy:

{context}

User Query: {user_query}

If the answer is not found in the context, respond with "Not found in the policy document".
    """

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant answering insurance policy questions strictly from provided context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📄 PDF Policy Q&A with Groq + Retrieval")

uploaded_file = st.file_uploader("Upload your policy PDF", type=["pdf"])
query = st.text_area("Enter your query")

if st.button("Get Answer"):
    if not GROQ_API_KEY:
        st.error("API key not found. Please set GROQ_API_KEY in your .env file.")
    elif uploaded_file and query.strip():
        with st.spinner("Processing..."):
            # 1. Extract PDF text
            pdf_text = extract_pdf_text(uploaded_file)

            # 2. Chunk PDF
            chunks = chunk_text(pdf_text)

            # 3. Build FAISS index
            index, embeddings, embed_model = build_faiss_index(chunks)

            # 4. Retrieve relevant chunks
            top_chunks = retrieve_chunks(query, index, chunks, embed_model, top_n=3)
            context = "\n\n".join(top_chunks)

            # 5. Query Groq
            answer = query_groq(context, query)

            # 6. Display
            st.subheader("Answer:")
            st.success(answer)
    else:
        st.warning("Please upload a PDF and enter a query.")
