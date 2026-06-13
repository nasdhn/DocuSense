from sentence_transformers import SentenceTransformer
import chromadb

def retrieve(question, n_results=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vector_question = model.encode(question)
 
    client = chromadb.PersistentClient(path="db/")
    collection = client.get_or_create_collection("documents")
    result = collection.query(
        query_embeddings=[vector_question],
        n_results=n_results
    )

    return result

if __name__ == "__main__":
    result = retrieve("Quelle est la formation faite par Dahman Nasim", 3)
    print(result['documents'][0])