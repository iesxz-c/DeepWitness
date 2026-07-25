# Agentic AI CCTV Investigation System

An AI-powered CCTV investigation platform that detects weapons and suspicious activity in video footage, builds event timelines, and generates investigation reports using multi-provider LLM agents.

## Architecture

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│ Frontend │────▶│   Backend    │────▶│  CV Pipeline  │
│ React/Vite│    │   FastAPI    │     │  YOLO (dual)  │
└──────────┘     └──────┬───────┘     └──────────────┘
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
       ┌─────────┐ ┌─────────┐ ┌─────────┐
       │ Agents  │ │ Schemas │ │ Notebooks│
       │ LLM     │ │ Pydantic│ │ Analysis │
       │ Fallback│ │         │ │          │
       └─────────┘ └─────────┘ └─────────┘
```

## Structure

| Directory | Description |
|-----------|-------------|
| `backend/` | FastAPI REST API — event CRUD, video upload, AI query, report generation |
| `frontend/` | React + Vite + Tailwind dashboard for visualizing events and querying the system |
| `cv_pipeline/` | Dual YOLOv8 weapon/knife detection pipeline with model weights |
| `agents/` | AI agents — timeline builder, evidence bundler, query investigator, report generator, LLM client with 4-tier fallback |
| `schemas/` | Shared Pydantic models (`Event`) |
| `notebooks/` | Jupyter notebooks for model training and analysis |
| `scripts/` | Test scripts for endpoints, fallback chain, and full pipeline |
| `csv/` | Data files |

## Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **npm** (comes with Node.js)

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
GROQ_API_KEY=your_groq_key
```

The `.env` file is gitignored. Never commit API keys.

### LLM Fallback Chain

The system uses a 4-tier fallback chain for AI queries — if one provider fails, it automatically falls back to the next:

| Tier | Provider | Model | Notes |
|------|----------|-------|-------|
| 1 | OpenRouter (free) | `openrouter/free` | Primary, no cost |
| 2 | Groq | `openai/gpt-oss-20b` | Fast inference |
| 3 | OpenRouter (paid) | `openai/gpt-4o-mini` | Uses same key as tier 1 |
| 4 | Google Gemini | `gemini-2.5-flash` | Last resort |

## Getting Started

### 1. Clone

```bash
git clone https://github.com/iesxz-c/fnl.git
cd fnl
```

### 2. Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r ../requirements.txt
pip install fastapi uvicorn[standard] opencv-python ultralytics torch google-generativeai openai

uvicorn main:app --reload --port 8000
```

API docs available at **http://localhost:8000/docs**

### 3. Frontend

```bash
cd frontend
npm install --legacy-peer-deps
npm run dev
```

Frontend runs at **http://localhost:5173**

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/events` | List events (optional filters: `camera`, `event_type`, `start_time`, `end_time`) |
| `GET` | `/timeline` | Get merged event timeline |
| `POST` | `/query` | Ask a natural language question about events (`{"question": "..."}`) |
| `GET` | `/report` | Generate a full investigation report (Markdown + structured) |
| `POST` | `/videos` | Upload a video for YOLO inference (`file` + optional `camera` form field) |
| `GET` | `/videos` | List uploaded and processed videos |

### Example: Upload & process a video

```bash
curl -X POST http://localhost:8000/videos \
  -F "file=@cv_pipeline/test_clips/sample.mp4" \
  -F "camera=cam-entrance"
```

## CV Pipeline

Uses two YOLOv8 models for detection:

- **`weapon_detect_v1_best.pt`** — detects guns and heavy weapons
- **`knife_detect_v2_best.pt`** — detects knives (dedicated retrain for better recall)

Model weights are stored in `cv_pipeline/weights/`.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/test_endpoints.py` | Smoke tests all API endpoints |
| `scripts/test_full_pipeline.py` | End-to-end pipeline test |
| `scripts/test_fallback_chain.py` | Tests the 4-tier LLM fallback |
| `scripts/test_backend_fallback.py` | Backend-specific fallback tests |
| `scripts/test_query_agent_fallback.py` | Query agent fallback tests |
| `scripts/test_report_agent_fallback.py` | Report agent fallback tests |
| `scripts/test_groq_reliability.py` | Groq provider reliability tests |
