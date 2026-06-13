from groq import Groq
import os
from dotenv import load_dotenv
from retrieval import retrieve

load_dotenv()

def generate(question, chunks):
    question_and_chunks = question + "\n\nContext:\n" + "\n".join(chunks)
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "Tu es un assistant. Réponds uniquement en te basant sur les extraits fournis."},
        {"role": "user", "content": question_and_chunks}
        ]
    )

    print(response.choices[0].message.content)
    return(response.choices[0].message.content)


if __name__ == "__main__":
    question = "Dans quel domaine le cv fourni peut il etre utilisé ? "
    chunks = retrieve(question, 3)
    generate(question, chunks['documents'][0])
