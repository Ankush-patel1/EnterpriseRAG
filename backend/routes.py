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
    return ask(body.question)
