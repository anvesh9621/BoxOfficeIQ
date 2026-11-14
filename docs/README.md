# BoxOfficeIQ — Intelligent Cinema Demand Forecasting & Decision Support System

> **AI-powered theatre attendance forecasting, sentiment analysis, and decision support for cinema operators.**

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Overview](#solution-overview)
3. [Architecture Diagram](#architecture-diagram)
4. [Tech Stack](#tech-stack)
5. [Quick Start](#quick-start)
6. [Feature List](#feature-list)
7. [API Endpoints Summary](#api-endpoints-summary)
8. [Dataset Description](#dataset-description)
9. [Screenshots](#screenshots)
10. [License](#license)

---

## Problem Statement

Cinema operators face significant uncertainty in daily attendance, making staffing, inventory, and marketing decisions difficult. Traditional methods rely on intuition and lagging indicators. BoxOfficeIQ addresses this by providing:

- **Accurate short-term attendance forecasts** (1–14 days ahead) at the individual theatre level.
- **Explainable predictions** via SHAP values, so managers understand *why* attendance is expected to be high or low.
- **Sentiment signals** from audience reviews to flag movies that may outperform or underperform box office expectations.
- **A conversational AI assistant** that answers operational questions grounded in real data.

---

## Solution Overview

BoxOfficeIQ is a full-stack machine learning platform consisting of:

| Layer | Description |
|---|---|
| **Data Pipeline** | Automated ingestion, validation, preprocessing, and feature engineering |
| **ML Models** | Ensemble of XGBoost, LightGBM, and LSTM models with calibrated prediction intervals |
| **Sentiment Engine** | Fine-tuned DistilBERT for positive/negative movie review classification |
| **REST API** | FastAPI backend serving predictions, metrics, SHAP explanations, and aggregated data |
| **Frontend Dashboard** | Vanilla JS + Chart.js 8-page interactive dashboard for cinema managers |
| **AI Assistant (RAG)** | Retrieval-Augmented Generation system combining FAISS vector search with an LLM for natural-language Q&A |
| **MLflow** | Experiment tracking, model registry, and artifact storage |
| **PostgreSQL** | Persistent store for attendance history, forecasts, model metrics, and reviews |

---

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                        BoxOfficeIQ Platform                        │
│                                                                    │
│  ┌───────────────┐    ┌──────────────────┐    ┌────────────────┐  │
│  │  Data Sources │    │   ML Pipeline    │    │  MLflow Server │  │
│  │               │    │                  │    │                │  │
│  │ · Cinema CSV  │───▶│ 1. Download/Gen  │───▶│ · Experiments  │  │
│  │ · Movie data  │    │ 2. Validate      │    │ · Model Reg.   │  │
│  │ · Reviews     │    │ 3. Preprocess    │    │ · Artifacts    │  │
│  └───────────────┘    │ 4. Feature Eng.  │    └────────────────┘  │
│                       │ 5. Train Models  │             │           │
│                       │    ·XGBoost      │             ▼           │
│                       │    ·LightGBM     │    ┌────────────────┐  │
│                       │    ·LSTM         │    │  Model Store   │  │
│                       │ 6. Evaluate      │───▶│  (joblib/pt)   │  │
│                       │ 7. SHAP Values   │    └───────┬────────┘  │
│                       └──────────────────┘            │           │
│                                                        ▼           │
│  ┌───────────────┐    ┌──────────────────┐    ┌────────────────┐  │
│  │   PostgreSQL  │◀───│  FastAPI Backend │◀───│  Loaded Models │  │
│  │               │    │                  │    └────────────────┘  │
│  │ · attendance  │───▶│ /api/forecasts   │                        │
│  │ · forecasts   │    │ /api/theatres    │    ┌────────────────┐  │
│  │ · metrics     │    │ /api/movies      │    │  RAG System    │  │
│  │ · reviews     │    │ /api/sentiment   │◀───│  · FAISS Index │  │
│  └───────────────┘    │ /api/assistant   │    │  · MiniLM-L6   │  │
│                       │ /api/metrics     │    │  · GPT-3.5     │  │
│                       └────────┬─────────┘    └────────────────┘  │
│                                │                                   │
│                                ▼                                   │
│                       ┌──────────────────┐                        │
│                       │  Frontend (JS)   │                        │
│                       │                  │                        │
│                       │ 1. Overview      │                        │
│                       │ 2. Forecasting   │                        │
│                       │ 3. Analytics     │                        │
│                       │ 4. Heatmap       │                        │
│                       │ 5. Sentiment     │                        │
│                       │ 6. Model Metrics │                        │
│                       │ 7. Simulator     │                        │
│                       │ 8. AI Assistant  │                        │
│                       └──────────────────┘                        │
└────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Category | Technology | Version / Notes |
|---|---|---|
| **Language** | Python | 3.10+ |
| **Web Framework** | FastAPI | 0.100+ |
| **ASGI Server** | Uvicorn | Standard extras |
| **Database** | PostgreSQL | 14+ |
| **ORM** | SQLAlchemy | 2.x |
| **Migrations** | Alembic | — |
| **ML – Gradient Boosting** | XGBoost, LightGBM, CatBoost | Latest stable |
| **ML – Deep Learning** | PyTorch, Transformers (HuggingFace) | — |
| **NLP – Sentiment** | DistilBERT (fine-tuned on IMDb) | HuggingFace |
| **Explainability** | SHAP | TreeExplainer + GradientExplainer |
| **Experiment Tracking** | MLflow | SQLite backend (dev) |
| **Vector Store** | FAISS (faiss-cpu) | IndexFlatL2 |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | 384-dim |
| **LLM** | OpenAI gpt-3.5-turbo | Fallback: rule-based |
| **Frontend** | HTML5 + CSS3 + Vanilla JS | — |
| **Charts** | Chart.js | 4.x |
| **Containerisation** | Docker + Docker Compose | — |
| **Data Processing** | pandas, NumPy, SciPy | — |
| **Visualisation** | Plotly, Matplotlib, Seaborn | — |

---

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ running locally (or via Docker)
- Node.js (optional, for a local HTTP server)
- Git

### 1 — Clone the Repository

```bash
git clone https://github.com/your-org/boxofficeiq.git
cd boxofficeiq
```

### 2 — Configure Environment

```bash
cp .env.example .env
# Edit .env and fill in DATABASE_URL, OPENAI_API_KEY, TMDB_API_KEY, etc.
```

### 3 — Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
cd ..
```

### 4 — Generate Synthetic Data & Build Database

```bash
make db-init          # Creates PostgreSQL schema
python ml/pipeline/download_data.py     # Generates synthetic CSV data
python ml/pipeline/validate_data.py    # Validates schema & ranges
python ml/pipeline/preprocess.py       # Cleans & saves processed data
python ml/pipeline/feature_engineering.py  # Engineers features
```

### 5 — Train Models

```bash
make train
# This runs: XGBoost, LightGBM, LSTM training
# Artifacts saved to models/ and tracked in MLflow
```

### 6 — Run the Backend

```bash
make run-backend
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### 7 — Open the Frontend

Open `frontend/index.html` in a browser, or use a local server:

```bash
cd frontend
python -m http.server 5500
# Visit http://localhost:5500
```

### Docker Compose (Alternative)

```bash
docker-compose up --build
```

---

## Feature List

BoxOfficeIQ provides **8 dashboard pages**, each serving a distinct decision-support function:

| # | Page | Description |
|---|---|---|
| **1** | **Overview** | KPI cards (total attendance, avg daily, top theatre, top movie), trend sparklines, attendance heatmap by day of week |
| **2** | **Forecasting** | 14-day attendance forecast per theatre, Chart.js line chart with confidence bands, MAE / RMSE metrics, SHAP feature importance bar chart |
| **3** | **Analytics** | Historical attendance trends, revenue estimates, comparisons across theatres and time periods |
| **4** | **Heatmap** | Calendar-style heatmap showing daily attendance density across all theatres; colour-coded intensity |
| **5** | **Sentiment** | DistilBERT-powered movie review sentiment (positive/negative), score distributions per movie, sentiment timeline |
| **6** | **Model Metrics** | Live model performance dashboard — MAE, RMSE, WAPE for each model; validation vs test set comparison |
| **7** | **Simulator** | "What-if" scenario tool — adjust holiday flag, weather, marketing spend, and see forecasted attendance change |
| **8** | **AI Assistant** | Natural-language Q&A powered by RAG (FAISS + GPT-3.5-turbo) + SQL data tool; asks and answers questions grounded in real data |

---

## API Endpoints Summary

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/theatres` | List all theatres |
| `GET` | `/api/movies` | List all movies |
| `GET` | `/api/attendance` | Attendance records with filters |
| `GET` | `/api/attendance/summary` | Aggregated KPIs (total, avg, top theatre) |
| `GET` | `/api/forecasts/{theatre_id}` | Forecast for a theatre |
| `POST` | `/api/forecasts/predict` | On-demand prediction for custom inputs |
| `GET` | `/api/sentiment/{movie_id}` | Sentiment scores for a movie |
| `POST` | `/api/sentiment/analyze` | Analyze raw review text |
| `GET` | `/api/metrics` | Model performance metrics |
| `GET` | `/api/shap/{theatre_id}` | SHAP values for latest forecast |
| `POST` | `/api/simulator` | Scenario simulation |
| `POST` | `/api/assistant/chat` | AI assistant chat message |

Full reference: see [`docs/api-reference.md`](./api-reference.md).

---

## Dataset Description

BoxOfficeIQ uses a **synthetic but realistic** dataset generated by `ml/pipeline/download_data.py`.

| File | Rows | Description |
|---|---|---|
| `ml/data/raw/cinema_attendance.csv` | ~16,425 | Daily attendance per theatre, 2022-01-01 to 2024-12-31, 15 theatres |
| `ml/data/raw/movies.csv` | 50 | Movie metadata: title, genre, release date, runtime, popularity, ratings |
| `ml/data/raw/sample_reviews.csv` | 500 | Balanced positive/negative movie reviews for DistilBERT training |
| `ml/data/processed/attendance_clean.csv` | ~16,000 | Cleaned attendance after deduplication & dtype coercion |
| `ml/data/processed/features.csv` | ~16,000 | Feature matrix with all engineered features |

**Key columns in `cinema_attendance.csv`:**

| Column | Type | Description |
|---|---|---|
| `date` | DATE | Calendar date (YYYY-MM-DD) |
| `theatre_id` | VARCHAR | Unique theatre identifier (T001–T015) |
| `movie_id` | VARCHAR | Movie shown that day |
| `audience_count` | INTEGER | Total daily attendance |
| `online_bookings` | INTEGER | Bookings made via app/web |
| `offline_bookings` | INTEGER | Walk-in / box-office bookings |
| `ticket_price` | FLOAT | Average ticket price (INR) |
| `show_count` | INTEGER | Number of screenings that day |

---

## Screenshots

> **Note:** Screenshots are captured from the live dashboard. Place PNG files in `docs/screenshots/`.

| Page | Screenshot |
|---|---|
| Overview | `docs/screenshots/overview.png` *(placeholder)* |
| Forecasting | `docs/screenshots/forecasting.png` *(placeholder)* |
| Sentiment | `docs/screenshots/sentiment.png` *(placeholder)* |
| AI Assistant | `docs/screenshots/assistant.png` *(placeholder)* |

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 BoxOfficeIQ Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
