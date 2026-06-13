import fitz
from sentence_transformers import SentenceTransformer

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


def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(chunks)


if __name__ == "__main__":
   pages =  parse_pdf("data/CV_Nasim_DAHMAN_T2i.pdf")
   chunks = chunk_text(pages)
   print("-----------")
   print(len(chunks))
   print("-----------")
   print(chunks[1])

   embeddings = embed_chunks(chunks)
   print(embeddings)