import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_ollama import OllamaLLM

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

st.title("📚 RAG Chatbot with PDFs & Ollama LLM")

# ---------------------------
# Step 1: Load PDF documents
# ---------------------------
if "vectorstore_loaded" not in st.session_state:
    documents = []
    data_folder = "data"  # folder containing PDFs
    for file in os.listdir(data_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(f"{data_folder}/{file}")
            documents.extend(loader.load())
    
    st.write(f"Documents Loaded: {len(documents)}")

    # ---------------------------
    # Step 2: Split into chunks
    # ---------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    st.write(f"Chunks Created: {len(chunks)}")

    # ---------------------------
    # Step 3: Create embeddings & vectorstore
    # ---------------------------
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local("vectorstore")
    st.write("Vectorstore created and saved locally!")

    st.session_state.vectorstore_loaded = True

# ---------------------------
# Step 4: Load vectorstore
# ---------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vectorstore = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------------------
# Step 5: Initialize Ollama LLM
# ---------------------------
llm = OllamaLLM(model="llama2")  

# ---------------------------
# Step 6: Create RetrievalQA
# ---------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# ---------------------------
# Step 7: Ask questions
# ---------------------------
query = st.text_input("Ask a question about your PDFs:")

if query:
    with st.spinner("Generating answer..."):
        # Use .invoke() instead of deprecated __call__
        result = qa_chain.invoke({"query": query})
        st.write("**Answer:**", result["result"])