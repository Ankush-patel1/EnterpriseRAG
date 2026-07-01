# EnterpriseRAG

An enterprise-grade Retrieval-Augmented Generation (RAG) system that answers questions about internal PDF documents using hybrid retrieval (FAISS + BM25 + cross-encoder reranking) and Google Gemini.

## Architecture

```
frontend/   React + Vite (Static Site)
backend/    FastAPI REST API
llm/        Gemini 2.5 Flash integration
retrieval/  Hybrid retrieval pipeline (FAISS · BM25 · CrossEncoder)
evaluation/ RAGAS evaluation harness
documents/  Drop PDF files here
```

---

## Local Development

### Prerequisites
- Python 3.11+
- Node.js 18+
- A Gemini API key ([get one here](https://aistudio.google.com/app/apikey))

### 1. Backend

```bash
# Clone and enter the project
cd EnterpriseRAG

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Copy and fill in your API key
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_actual_key

# Add PDF files to documents/
# (required — the app returns empty results without them)

# Run the API server
python -m backend.main
# → API available at http://localhost:8000
# → Swagger UI at http://localhost:8000/docs
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
# → App available at http://localhost:5173
```

---

## Deploying to Render

This project ships with a `render.yaml` [Blueprint](https://render.com/docs/blueprint-spec) that provisions **both** services in one click.

### Step-by-step

1. **Push to GitHub** — make sure `Dockerfile` and `render.yaml` are committed.

2. **Add your PDFs** — either commit sample PDFs to `documents/` (they are gitignored by default; remove that rule if you want them tracked), or mount a [Render Disk](https://render.com/docs/disks) and upload files there.

3. **Create a Blueprint on Render**
   - Go to [dashboard.render.com](https://dashboard.render.com) → **New → Blueprint**
   - Connect your GitHub repo
   - Render will detect `render.yaml` and create both services automatically

4. **Set Environment Variables** in the Render dashboard for each service:

   | Service | Key | Value |
   |---------|-----|-------|
   | `enterpriserag-backend` | `GEMINI_API_KEY` | Your Gemini API key |
   | `enterpriserag-backend` | `CORS_ORIGINS` | Your frontend URL, e.g. `https://enterpriserag-frontend.onrender.com` |
   | `enterpriserag-frontend` | `VITE_API_URL` | Your backend URL, e.g. `https://enterpriserag-backend.onrender.com` |

   > **Note**: Set the backend `CORS_ORIGINS` *after* the frontend service URL is known. You can redeploy the backend service afterwards.

5. **Deploy** — click Deploy on both services. The backend will build the Docker image and start Gunicorn.

### First-request warm-up

On the first request after deploy (or cold start on the free plan), the backend will:
- Download the embedding model (`all-MiniLM-L6-v2`) and reranker
- Parse all PDFs and build the FAISS + BM25 index

This takes **30–90 seconds**. The `/health` endpoint responds immediately and won't block Render's health check.

---

## Evaluation

Run the RAGAS evaluation harness against your golden dataset:

```bash
python -m evaluation.eval
# Results saved to evaluation/results.csv
```

Edit `evaluation/golden_dataset.json` to add your own question/answer pairs.

---

## Safety Features

Every `/ask` request is checked for:
- **PII** — emails, phone numbers, SSNs, credit card numbers
- **Prompt injection** — known jailbreak/override phrases
- **Daily token budget** — 50,000 tokens per IP per day (in-memory; resets on restart)
