import fitz
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path

def parse_pdf(path):
    result = []
    doc = fitz.open(path)

    for page in doc:
        text = page.get_text()
        result.append(text)

    print(result[0])
    print("-----------")
    print("Nombre de string dasn result : ", len(result))
    return result


def chunk_text(pages, chunk_size=500, overlap=50):
    result = []
    total_text = "".join(pages)
    i = 0
    while (i < len(total_text)):
        chunk = total_text[i: i + chunk_size]
        result.append(chunk)
        i = i + max(1, len(chunk) - overlap)

    return result

def store_embeddings(chunks, embeddings, filename):
    ids = [f"{filename}_{i}" for i in range(len(chunks))]

    client = chromadb.PersistentClient(path="db/")
    collection = client.get_or_create_collection("documents")
    collection.add(ids=ids, embeddings=embeddings, documents=chunks)


def ingest_folder(folder_path):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    files = Path(folder_path).glob("*.pdf")
    for file in files:
        page = parse_pdf(file)
        chunk =chunk_text(page)
        embedding = model.encode(chunk)
        store_embeddings(chunk, embedding, file)


if __name__ == "__main__":
    ingest_folder("data/")
