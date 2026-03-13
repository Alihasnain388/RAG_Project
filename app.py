from fastapi import FastAPI
from rag_pipeline import qa_chain

app = FastAPI()

@app.get("/ask")

def ask_question(query: str):

    result = qa_chain({"query": query})

    return {
        "question": query,
        "answer": result["result"]
    }