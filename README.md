This is a Streamlit web app that creates a RAG (Retrieval-Augmented Generation) chatbot using PDFs and an Ollama LLM. Here's a concise breakdown:

Load PDFs – Reads all PDF files in a data folder.

Split Text – Breaks PDF content into smaller chunks for better processing.

Create Embeddings & Vectorstore – Converts chunks into vector embeddings using HuggingFaceEmbeddings and stores them in FAISS for fast similarity search.

Load Vectorstore – Loads the saved vector database for retrieval.

Initialize LLM – Uses OllamaLLM (Llama2 model) to generate answers.

RetrievalQA – Combines the LLM with the vectorstore retriever so it can answer questions using the PDF content.

Ask Questions – User inputs a query in Streamlit, and the bot retrieves relevant chunks from PDFs and generates an answer.

Effectively: It’s a chatbot that can answer questions based on the content of document.
