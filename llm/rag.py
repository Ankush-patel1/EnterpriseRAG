import os
from google import genai
from dotenv import load_dotenv
from retrieval.rag import retrieve

load_dotenv()

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
_MODEL = "gemini-2.5-flash"



def generate_context(chunks):
    """Formats retrieved chunks into a context string with source headers."""
    parts = []
    for chunk in chunks:
        header = f"[{chunk['filename']}, p.{chunk['page']}]"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def ask_gemini(question, context):
    """Sends the question and context to Gemini and returns the answer."""
    prompt = f"""You are an enterprise knowledge assistant.
Answer ONLY using the provided context.
Cite the source of every factual claim inline like this: [filename, p.N]
If the answer is not in the context, say: "I could not find this information in the provided documents."

Context:
{context}

Question:
{question}

Answer:"""

    response = _client.models.generate_content(model=_MODEL, contents=prompt)
    return response.text


def ask(question):
    """Retrieves relevant chunks, builds context, and returns an answer with sources."""
    chunks = retrieve(question)

    if not chunks:
        return {
            "answer": "I could not find any relevant information in the provided documents.",
            "sources": []
        }

    context = generate_context(chunks)
    answer = ask_gemini(question, context)

    sources = [
        {"filename": c["filename"], "page": c["page"], "text": c["text"]}
        for c in chunks
    ]

    return {"answer": answer, "sources": sources}

