"""
BoxOfficeIQ - FastAPI Main Application
Cinematic demand forecasting and decision support system.
Integrates ML inference, NLP sentiment, explainability, scenario simulation, and hybrid RAG.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure root package resolution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(CURRENT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR.parent))

from app.config import Settings
from app.routers import (
    historical,
    theatres,
    forecast,
    model_metrics,
    sentiment,
    explainability,
    simulation,
    assistant,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("boxofficeiq")

settings = Settings()

app = FastAPI(
    title="BoxOfficeIQ API",
    description="Intelligent Cinema Demand Forecasting & Decision Support API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for all origins in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers under /api prefix
app.include_router(historical.router, prefix="/api")
app.include_router(theatres.router, prefix="/api")
app.include_router(forecast.router, prefix="/api")
app.include_router(model_metrics.router, prefix="/api")
app.include_router(sentiment.router, prefix="/api")
app.include_router(explainability.router, prefix="/api")
app.include_router(simulation.router, prefix="/api")
app.include_router(assistant.router, prefix="/api")

START_TIME = time.time()


@app.on_event("startup")
async def startup_event():
    logger.info("Initializing BoxOfficeIQ backend services...")
    # Pre-warm analyzers and caches in background
    try:
        sentiment.get_analyzer()
        logger.info("Sentiment analyzer verified.")
    except Exception as e:
        logger.warning(f"Sentiment analyzer init warning: {e}")

    try:
        explainability.get_explainer()
        logger.info("SHAP explainer verified.")
    except Exception as e:
        logger.warning(f"SHAP explainer init warning: {e}")

    logger.info("BoxOfficeIQ startup sequence completed successfully.")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "Internal server error occurred.",
            "detail": str(exc) if settings.debug else "Please contact system administrator.",
            "path": request.url.path,
        },
    )


@app.get("/")
def root() -> Dict[str, str]:
    """Root platform health ping."""
    return {"status": "ok", "app": "BoxOfficeIQ", "version": "1.0.0"}


@app.get("/health")
def health() -> Dict[str, Any]:
    """Detailed system diagnostic status."""
    processed_dir = PROJECT_DIR / "ml" / "data" / "processed"
    models_dir = PROJECT_DIR / "ml" / "models"
    
    attendance_csv = processed_dir / "attendance_clean.csv"
    best_model_pkl = models_dir / "best_model.pkl"
    sentiment_dir = models_dir / "sentiment_model"

    return {
        "status": "healthy",
        "app": "BoxOfficeIQ",
        "version": "1.0.0",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "environment": settings.environment,
        "database": "csv_fallback_ready",
        "artifacts": {
            "attendance_clean_csv": attendance_csv.exists(),
            "best_model_pkl": best_model_pkl.exists(),
            "sentiment_model": sentiment_dir.exists(),
        },
    }
