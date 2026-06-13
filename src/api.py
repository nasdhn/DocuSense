from fastapi import FastAPI
from pydantic import BaseModel
from src.retrieval import retrieve
from src.generate import generate

app = FastAPI()

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(body: AskRequest):
    chunks = retrieve(body.question, 3)
    result = generate(body.question, chunks['documents'][0])
    return {"answer": result}


