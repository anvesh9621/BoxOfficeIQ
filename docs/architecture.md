# BoxOfficeIQ — Technical Architecture

> **Version:** 1.0  
> **Last Updated:** 2024

---

## Table of Contents

1. [System Components Diagram](#system-components-diagram)
2. [Data Flow](#data-flow)
3. [Component Interactions](#component-interactions)
4. [Deployment Architecture](#deployment-architecture)
5. [Security Considerations](#security-considerations)

---

## System Components Diagram

```
╔══════════════════════════════════════════════════════════════════════════╗
║                     BoxOfficeIQ — Component Map                         ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                     DATA LAYER                                   │    ║
║  │                                                                  │    ║
║  │  ┌───────────────────┐    ┌────────────────────────────────┐    │    ║
║  │  │  Raw Data Sources │    │        PostgreSQL DB            │    │    ║
║  │  │                   │    │                                │    │    ║
║  │  │  · cinema_att.csv │    │  Tables:                       │    │    ║
║  │  │  · movies.csv     │───▶│  · cinema_attendance           │    │    ║
║  │  │  · reviews.csv    │    │  · theatre_info                │    │    ║
║  │  │  · theatre_info   │    │  · movies                      │    │    ║
║  │  └───────────────────┘    │  · movie_reviews               │    │    ║
║  │                           │  · forecasts                   │    │    ║
║  │  ┌───────────────────┐    │  · model_metrics               │    │    ║
║  │  │  ML Pipeline      │    └────────────────────────────────┘    │    ║
║  │  │                   │                                           │    ║
║  │  │  download_data.py │                                           │    ║
║  │  │  validate_data.py │                                           │    ║
║  │  │  preprocess.py    │                                           │    ║
║  │  │  feature_eng.py   │                                           │    ║
║  │  └───────────────────┘                                           │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
║                                                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                   ML / AI LAYER                                  │    ║
║  │                                                                  │    ║
║  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │    ║
║  │  │   XGBoost    │  │  LightGBM    │  │        LSTM          │  │    ║
║  │  │  Regressor   │  │  Regressor   │  │  (PyTorch, 28-day    │  │    ║
║  │  │              │  │              │  │   lookback)          │  │    ║
║  │  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │    ║
║  │         └─────────────────┴─────────────────────┘               │    ║
║  │                           │ Ensemble / Best Model                │    ║
║  │                           ▼                                      │    ║
║  │               ┌───────────────────────┐                          │    ║
║  │               │   SHAP Explainer      │                          │    ║
║  │               │   (TreeExplainer)     │                          │    ║
║  │               └───────────────────────┘                          │    ║
║  │                                                                   │    ║
║  │  ┌──────────────────────────────────────────────────────────┐   │    ║
║  │  │  DistilBERT Sentiment Engine                              │   │    ║
║  │  │  · Fine-tuned on IMDb reviews                            │   │    ║
║  │  │  · Positive / Negative classification                     │   │    ║
║  │  │  · Confidence scores (softmax)                           │   │    ║
║  │  └──────────────────────────────────────────────────────────┘   │    ║
║  │                                                                   │    ║
║  │  ┌──────────────────────────────────────────────────────────┐   │    ║
║  │  │  MLflow Tracking Server                                   │   │    ║
║  │  │  · Experiment runs, params, metrics, artifacts           │   │    ║
║  │  │  · Model Registry (Staging / Production stages)          │   │    ║
║  │  └──────────────────────────────────────────────────────────┘   │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
║                                                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                   RAG / NLP LAYER                                │    ║
║  │                                                                  │    ║
║  │  ┌───────────────────────┐    ┌─────────────────────────────┐  │    ║
║  │  │  Document Corpus      │    │  FAISS Vector Index          │  │    ║
║  │  │  (docs/*.md files)    │───▶│  (IndexFlatL2, 384-dim)     │  │    ║
║  │  └───────────────────────┘    └────────────────┬────────────┘  │    ║
║  │                                                 │               │    ║
║  │  ┌───────────────────────┐                      ▼               │    ║
║  │  │  SQL Data Tool        │    ┌─────────────────────────────┐  │    ║
║  │  │  (pandas + intent     │    │  OpenAI GPT-3.5-turbo        │  │    ║
║  │  │   routing)            │───▶│  (LLM response generation)  │  │    ║
║  │  └───────────────────────┘    └─────────────────────────────┘  │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
║                                                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                   API LAYER                                      │    ║
║  │                                                                  │    ║
║  │  FastAPI (Uvicorn)  — http://localhost:8000                      │    ║
║  │  Routers: attendance · theatres · movies · forecasts ·          │    ║
║  │           sentiment · metrics · simulator · assistant            │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
║                                                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                  PRESENTATION LAYER                              │    ║
║  │                                                                  │    ║
║  │  HTML5 + CSS3 + Vanilla JavaScript + Chart.js                   │    ║
║  │  Pages: Overview · Forecasting · Analytics · Heatmap ·          │    ║
║  │         Sentiment · Model Metrics · Simulator · AI Assistant     │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Data Flow

The end-to-end data flow through BoxOfficeIQ has 7 stages:

### Stage 1 — Raw Data Ingestion

```
ml/data/raw/
  ├── cinema_attendance.csv   (synthetic, 2022–2024, 15 theatres)
  ├── movies.csv              (50 movies with metadata)
  └── sample_reviews.csv      (500 labelled reviews)
```

Script: `ml/pipeline/download_data.py`  
- Generates synthetic data with NumPy (seed 42) if real data is unavailable.  
- Applies realistic attendance patterns: theatre-type base, weekend multiplier, seasonal peaks.

### Stage 2 — Validation

Script: `ml/pipeline/validate_data.py`  
- Schema checks (expected columns, correct dtypes).  
- Range checks (`audience_count >= 0`, `ticket_price > 0`).  
- Duplicate detection (theatre_id + date uniqueness).  
- Missing-value report with column-level statistics.

### Stage 3 — Preprocessing

Script: `ml/pipeline/preprocess.py`  
- Coerce `date` column to `datetime64`.  
- Fill or drop rows with missing `audience_count`.  
- Remove exact duplicate rows.  
- Output: `ml/data/processed/attendance_clean.csv`

### Stage 4 — Feature Engineering

Script: `ml/pipeline/feature_engineering.py`  
- Temporal features: `day_of_week`, `is_weekend`, `month`, `week_of_year`, `quarter`, `day_of_month`.  
- Holiday indicator: `is_holiday` (Indian public holidays 2022–2024).  
- Lag features: `lag_1`, `lag_7`, `lag_14` (computed per theatre to avoid leakage).  
- Rolling statistics: `rolling_mean_7`, `rolling_mean_14`, `rolling_std_7` (per theatre, using only past data).  
- Output: `ml/data/processed/features.csv`, `ml/data/processed/feature_names.json`

### Stage 5 — Model Training

Scripts: `ml/training/train_xgboost.py`, `train_lightgbm.py`, `train_lstm.py`  
- **Time-based split**: train ≤ 2024-07-01, validation 2024-07-02 to 2024-12-31.  
- Each model logged to MLflow with params, metrics, and serialized artifact.  
- Prediction intervals calibrated on validation residuals.  
- SHAP values computed and cached per theatre.

### Stage 6 — Predictions & API

```
Feature vector  →  Loaded Model  →  Forecast DataFrame
                →  SHAP Explainer →  SHAP values JSON
                →  PostgreSQL INSERT into forecasts table
```

FastAPI endpoints load models at startup from `models/` directory.  
Predictions are served in real time or pre-computed nightly.

### Stage 7 — Frontend Consumption

```
Browser (JS fetch)  →  FastAPI endpoint  →  JSON response  →  Chart.js render
```

Each dashboard page fetches data from its corresponding API route and renders charts, tables, and explanations.

---

## Component Interactions

```
┌────────────┐    HTTP/JSON     ┌────────────────────────────────┐
│  Frontend  │◀────────────────▶│       FastAPI Backend           │
│  (HTML/JS) │                  │                                 │
└────────────┘                  │  Routers (fastapi.APIRouter):   │
                                │  · attendance.py                │
                                │  · theatres.py                  │
                                │  · movies.py                    │
                                │  · forecasts.py                 │
                                │  · sentiment.py                 │
                                │  · metrics.py                   │
                                │  · simulator.py                 │
                                │  · assistant.py                 │
                                │                                 │
                                │  Services:                      │
                                │  · ml_service.py (model I/O)   │
                                │  · rag_service.py (FAISS+LLM)  │
                                │  · db_service.py (SQLAlchemy)  │
                                └──────────────┬──────────────────┘
                                               │
                       ┌───────────────────────┼──────────────────────┐
                       │                       │                      │
                       ▼                       ▼                      ▼
              ┌─────────────────┐   ┌──────────────────┐   ┌────────────────┐
              │   PostgreSQL    │   │  ML Model Files  │   │  RAG System    │
              │   (SQLAlchemy)  │   │  (models/*.pkl   │   │  (FAISS index  │
              │                 │   │   models/*.pt)   │   │   + OpenAI)    │
              └─────────────────┘   └──────────────────┘   └────────────────┘
```

**Key interaction patterns:**

| Interaction | Mechanism |
|---|---|
| Frontend ↔ Backend | REST HTTP (JSON), CORS enabled for localhost |
| Backend ↔ PostgreSQL | SQLAlchemy ORM with connection pooling |
| Backend ↔ ML Models | `joblib.load()` / `torch.load()` at startup, cached in memory |
| Backend ↔ RAG | `rag_service.query(user_question)` returns `{answer, sources}` |
| Backend ↔ MLflow | `mlflow.sklearn.load_model()` for Production-stage models |
| RAG ↔ FAISS | `index.search(query_embedding, k=3)` for nearest-chunk retrieval |
| RAG ↔ OpenAI | `openai.ChatCompletion.create(messages=[...])` with retrieved context |

---

## Deployment Architecture

### Development (Local)

```
localhost:5500  →  Frontend (python -m http.server)
localhost:8000  →  FastAPI (uvicorn backend.app.main:app --reload)
localhost:5432  →  PostgreSQL
localhost:5000  →  MLflow UI (mlflow ui)
```

### Docker Compose

```yaml
services:
  postgres:    port 5432, volume: pgdata
  backend:     port 8000, depends_on: postgres, env_file: .env
```

Frontend is served as static files (no container needed in development).

### Production (Recommended)

```
                          ┌──────────────────┐
   User Browser ─────────▶  Nginx / Caddy    │
                          │  (Reverse Proxy)  │
                          └──────┬────────────┘
                                 │
                    ┌────────────┴──────────────┐
                    │                           │
                    ▼                           ▼
           ┌────────────────┐        ┌────────────────────┐
           │  Static Files  │        │  FastAPI (Gunicorn  │
           │  (frontend/)   │        │  + Uvicorn workers) │
           └────────────────┘        └────────┬───────────┘
                                              │
                              ┌───────────────┴──────────────┐
                              │                              │
                              ▼                              ▼
                     ┌────────────────┐          ┌─────────────────┐
                     │  PostgreSQL    │          │  MLflow Server  │
                     │  (managed RDS  │          │  (EC2 / Cloud)  │
                     │   or Cloud SQL)│          └─────────────────┘
                     └────────────────┘
```

**Recommended production stack:**
- Hosting: AWS EC2 / GCP Cloud Run / Azure App Service
- Database: AWS RDS PostgreSQL or GCP Cloud SQL
- Reverse proxy: Nginx with SSL termination (Let's Encrypt)
- Process manager: Gunicorn with Uvicorn workers
- Environment secrets: AWS Secrets Manager / GCP Secret Manager

---

## Security Considerations

### 1. Read-Only Database Access for AI Assistant

The RAG/AI assistant SQL tool uses a **separate read-only PostgreSQL role**:

```sql
CREATE ROLE boxofficeiq_readonly LOGIN PASSWORD '...';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO boxofficeiq_readonly;
```

This prevents the assistant from accidentally or maliciously modifying data.

### 2. Parameterized Queries

All database interactions use **SQLAlchemy ORM** or **parameterized text queries**:

```python
# Correct — parameterized
result = db.execute(text("SELECT * FROM cinema_attendance WHERE theatre_id = :tid"),
                    {"tid": theatre_id})

# Never done — string interpolation
# result = db.execute(f"SELECT * FROM ... WHERE id = {user_input}")  ← SQL injection
```

### 3. Environment Variables

All secrets (database credentials, OpenAI API key, TMDB API key) are stored in `.env` and loaded via `pydantic-settings`. The `.env` file is listed in `.gitignore` and is **never committed** to version control.

### 4. CORS Policy

CORS is restricted to known frontend origins:

```python
origins = ["http://localhost:3000", "http://localhost:5500", "null"]
```

In production, replace with the specific production domain.

### 5. No Arbitrary SQL Injection via AI Tool

The AI assistant SQL tool uses **keyword intent routing**: it recognises phrases like "total attendance", "top theatre", "average bookings" and maps them to **pre-approved pandas operations on cached DataFrames**, never to raw SQL string execution.

### 6. Input Validation

All API request bodies and query parameters are validated via **Pydantic models**. Invalid inputs return a `422 Unprocessable Entity` with descriptive error messages — no stack traces are leaked to the client.

### 7. Rate Limiting (Future)

For production deployments, rate limiting is recommended using `slowapi` (a FastAPI-compatible limiter), particularly for the `/api/assistant/chat` endpoint that calls OpenAI.
