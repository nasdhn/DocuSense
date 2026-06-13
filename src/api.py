from fastapi import FastAPI
from pydantic import BaseModel
from src.retrieval import retrieve
from src.generate import generate

app = FastAPI()
history = []

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(body: AskRequest):
    global history 

    chunks = retrieve(body.question, 3)
    result = generate(body.question, chunks['documents'][0], history)
    history.append({"role": "user",      "content": body.question})
    history.append({"role": "assistant", "content": result})
    
    return {"answer": result}


