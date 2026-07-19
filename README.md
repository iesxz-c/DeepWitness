# CCTV Investigation System

A monorepo for an AI-powered CCTV investigation platform.

## Structure

| Directory | Description |
|-----------|-------------|
| `backend/` | FastAPI backend service |
| `frontend/` | React + Vite frontend |
| `cv_pipeline/` | Computer vision pipeline |
| `agents/` | AI agent modules |
| `notebooks/` | Jupyter notebooks for analysis |
| `schemas/` | Shared data schemas |

## Getting Started

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```
