# EnterpriseRAG

An enterprise-grade Retrieval-Augmented Generation (RAG) system that answers questions about internal PDF documents using hybrid retrieval (FAISS + BM25 + cross-encoder reranking) and Google Gemini.

## Architecture

```
frontend/   React + Vite (Static Site)
backend/    FastAPI REST API
llm/        Gemini 2.5 Flash integration
retrieval/  Hybrid retrieval pipeline (FAISS · BM25 · CrossEncoder)
evaluation/ RAGAS evaluation harness
documents/  PDF files for retrieval
```

---

## Local Development

### Prerequisites
- Python 3.11+
- Node.js 18+
- A Gemini API key ([get one here](https://aistudio.google.com/app/apikey))

### 1. Backend

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Copy and fill in your API key
cp .env.example .env
# Edit .env → set GEMINI_API_KEY=your_actual_key

# Run the API server
python -m backend.main
# → API at http://localhost:8000
# → Swagger UI at http://localhost:8000/docs
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
# → App at http://localhost:5173
```

---

## Deploying to Railway

This project deploys as **two separate Railway services** from the same GitHub repo.

### Prerequisites
- A [Railway](https://railway.com) account
- Your repo pushed to GitHub (`Ankush-patel1/EnterpriseRAG`)

---

### Step 1 — Create a new Railway project

1. Go to **[railway.com](https://railway.com)** → **New Project**
2. Choose **Deploy from GitHub repo**
3. Select `Ankush-patel1/EnterpriseRAG`
4. Railway detects `railway.toml` and creates the **backend** service automatically

---

### Step 2 — Set backend environment variables

In the Railway dashboard, click the backend service → **Variables** tab:

| Key | Value |
|-----|-------|
| `GEMINI_API_KEY` | Your Gemini API key |
| `CORS_ORIGINS` | Your frontend Railway URL *(set after Step 4)* |

Click **Deploy** to trigger the first build.

> **Build time**: ~5–10 minutes on first deploy (downloading ML models and Python deps).

---

### Step 3 — Add the frontend service

1. In your Railway project, click **+ New Service**
2. Choose **GitHub Repo** → select `Ankush-patel1/EnterpriseRAG` again
3. Click the service → **Settings** → set **Root Directory** to `frontend`
4. Railway auto-detects Node.js and runs `npm ci && npm run build`
5. Under **Settings → Networking** → click **Generate Domain** to get a public URL

---

### Step 4 — Set frontend environment variable

In the frontend service → **Variables** tab:

| Key | Value |
|-----|-------|
| `VITE_API_URL` | Your backend Railway URL (e.g. `https://enterpriserag-backend.up.railway.app`) |

Save → Railway rebuilds the frontend with the correct API URL baked in.

---

### Step 5 — Update backend CORS

Now that you have the frontend URL, go back to the **backend service → Variables**:

| Key | Value |
|-----|-------|
| `CORS_ORIGINS` | Your frontend Railway URL (e.g. `https://enterpriserag-frontend.up.railway.app`) |

Save → backend redeploys with CORS unlocked for your frontend.

---

### Step 6 — Verify

```bash
# Health check
curl https://your-backend.up.railway.app/health
# → {"status":"healthy"}
```

Open your frontend URL in the browser and ask a question. The **first query is slow** (~30–90s) — the backend is loading the embedding model and indexing all PDFs. Subsequent queries are fast (~2–5s).

---

## Evaluation

```bash
python -m evaluation.eval
# Results saved to evaluation/results.csv
```

Edit `evaluation/golden_dataset.json` to add your own Q&A pairs.

---

## Safety Features

Every `/ask` request is checked for:
- **PII** — emails, phone numbers, SSNs, credit card numbers
- **Prompt injection** — known jailbreak/override phrases
- **Daily token budget** — 50,000 tokens per IP per day
