from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from llm.rag import ask

router = APIRouter()


class QuestionRequest(BaseModel):
    question: str


@router.get("/")
def root():
    return {"message": "Enterprise RAG Backend Running"}


@router.get("/health")
def health():
    return {"status": "healthy"}


@router.post("/ask")
def ask_question(body: QuestionRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        return ask(body.question)
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            raise HTTPException(
                status_code=429,
                detail="Gemini API quota exceeded. Please wait a minute and try again."
            )
        raise HTTPException(status_code=500, detail=f"Internal error: {err_msg[:200]}")

