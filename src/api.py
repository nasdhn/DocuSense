from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from src.ingest import ingest_file
from src.retrieval import retrieve
from src.generate import generate
from fastapi.responses import HTMLResponse
from fastapi import HTTPException
from fastapi import UploadFile, File
import shutil
from sentence_transformers import SentenceTransformer

app = FastAPI()
history = []
model = SentenceTransformer("all-MiniLM-L6-v2")

class AskRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return HTMLResponse(content=open("index.html").read())

@app.post("/ask")
def ask(body: AskRequest):
    global history 
    chunks = retrieve(body.question, 3)
    print("CHUNKS RÉCUPÉRÉS:", chunks['documents'][0]) 
    result = generate(body.question, chunks['documents'][0], history)
    history.append({"role": "user",      "content": body.question})
    history.append({"role": "assistant", "content": result})
    return {"answer": result}

@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    if Path(f"data/{file.filename}").exists():
        raise HTTPException(status_code=400, detail="Un fichier avec ce nom existe déjà.")
    else:
        with open(f"data/{file.filename}", "wb") as f:
            shutil.copyfileobj(file.file, f)
    ingest_file(f"data/{file.filename}", model)
    return("File uploaded successfully.")

