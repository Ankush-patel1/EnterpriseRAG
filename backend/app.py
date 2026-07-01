import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import router

app = FastAPI(title="Enterprise RAG API")

# CORS_ORIGINS — comma-separated list of allowed origins.
# Set this env var in the Render dashboard to your frontend URL,
# e.g. "https://enterpriserag-frontend.onrender.com"
# Falls back to localhost for local development.
_raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(router)
