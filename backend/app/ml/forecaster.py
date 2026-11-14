"""
BoxOfficeIQ - Model Forecaster
Inference wrapper for the trained best model.
"""

import json
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
ML_DATA_DIR = ROOT / "ml" / "data" / "processed"
ML_MODELS_DIR = ROOT / "ml" / "models"

BEST_MODEL_PKL = ML_MODELS_DIR / "best_model.pkl"
FEATURE_NAMES_JSON = ML_DATA_DIR / "feature_names.json"
SHAP_SUMMARY_JSON = ML_DATA_DIR / "shap_summary.json"

# Indian public holidays (sample set, 2022-2025)
INDIAN_HOLIDAYS = {
    "2022-01-26", "2022-08-15", "2022-10-02", "2022-11-01", "2022-12-25",
    "2023-01-26", "2023-08-15", "2023-10-02", "2023-10-24", "2023-12-25",
    "2024-01-26", "2024-08-15", "2024-10-02", "2024-10-13", "2024-12-25",
    "2025-01-26", "2025-08-15", "2025-10-02", "2025-10-21", "2025-12-25",
}


class ModelForecaster:
    """
    Wraps the trained best model for production inference.
    
    Usage:
        forecaster = ModelForecaster()
        result = forecaster.predict("theatre_1", "2025-01-07", horizon_days=7)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_names_path: Optional[str] = None,
        residual_std: Optional[Dict[str, float]] = None,
    ):
        self.model = None
        self.model_name = "unknown"
        self.feature_names: List[str] = []
        self._loaded = False
        self._residual_std: Dict[str, float] = residual_std or {}
        self._global_std: float = 50.0  # fallback

        mp = Path(model_path) if model_path else BEST_MODEL_PKL
        fp = Path(feature_names_path) if feature_names_path else FEATURE_NAMES_JSON

        self._load_model(mp, fp)

    def _load_model(self, model_path: Path, feature_names_path: Path):
        try:
            payload = joblib.load(model_path)
            self.model = payload["model"]
            self.model_name = payload.get("model_name", "unknown")
            self.feature_names = payload.get("feature_names", [])
            self._loaded = True
            logger.info(f"ModelForecaster loaded: {self.model_name} ({len(self.feature_names)} features)")
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            self._loaded = False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._loaded = False

        if not self.feature_names and feature_names_path.exists():
            try:
                with open(feature_names_path) as f:
                    fn_data = json.load(f)
                if isinstance(fn_data, dict):
                    self.feature_names = fn_data.get("feature_columns", list(fn_data.keys()))
                else:
                    self.feature_names = fn_data
            except Exception as e:
                logger.warning(f"Could not load feature names: {e}")

    def _prepare_features(
        self,
        theatre_id: str,
        date: datetime,
        recent_audience: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Build a feature vector for the given theatre and date.
        
        recent_audience: list of recent audience counts (most recent last),
                         used for lag features. If None, zeros are used.
        """
        d = pd.Timestamp(date)
        row: Dict[str, Any] = {}

        lag_values = recent_audience or []

        for feat in self.feature_names:
            if feat == "day_of_week":
                row[feat] = d.dayofweek
            elif feat == "is_weekend":
                row[feat] = int(d.dayofweek >= 5)
            elif feat == "month":
                row[feat] = d.month
            elif feat == "day_of_month":
                row[feat] = d.day
            elif feat == "week_of_year":
                row[feat] = d.isocalendar().week
            elif feat == "quarter":
                row[feat] = d.quarter
            elif feat == "is_holiday":
                row[feat] = int(d.strftime("%Y-%m-%d") in INDIAN_HOLIDAYS)
            elif feat == "lag_1":
                row[feat] = float(lag_values[-1]) if len(lag_values) >= 1 else 0.0
            elif feat == "lag_7":
                row[feat] = float(lag_values[-7]) if len(lag_values) >= 7 else 0.0
            elif feat == "lag_14":
                row[feat] = float(lag_values[-14]) if len(lag_values) >= 14 else 0.0
            elif feat == "rolling_mean_7":
                vals = lag_values[-7:] if len(lag_values) >= 7 else lag_values
                row[feat] = float(np.mean(vals)) if vals else 0.0
            elif feat == "rolling_mean_14":
                vals = lag_values[-14:] if len(lag_values) >= 14 else lag_values
                row[feat] = float(np.mean(vals)) if vals else 0.0
            elif feat == "rolling_std_7":
                vals = lag_values[-7:] if len(lag_values) >= 7 else lag_values
                row[feat] = float(np.std(vals)) if len(vals) > 1 else 0.0
            else:
                row[feat] = 0.0

        return pd.DataFrame([row])[self.feature_names]

    def _get_theatre_std(self, theatre_id: str) -> float:
        return self._residual_std.get(theatre_id, self._global_std)

    def predict(
        self,
        theatre_id: str,
        start_date: str,
        horizon_days: int = 7,
        features_df: Optional[pd.DataFrame] = None,
        recent_audience: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Generate predictions for a theatre over horizon_days.
        
        Args:
            theatre_id: theatre identifier string
            start_date: ISO date string "YYYY-MM-DD" for first prediction day
            horizon_days: number of days to forecast
            features_df: optional pre-built feature DataFrame (must have all feature cols);
                         if provided, used directly (must have horizon_days rows)
            recent_audience: list of recent historical audience counts for lag features
        
        Returns:
            {
              "theatre_id": str,
              "model_name": str,
              "confidence_note": str,
              "predictions": [
                {
                  "date": "YYYY-MM-DD",
                  "predicted_audience": float,
                  "lower_bound": float,
                  "upper_bound": float
                }, ...
              ]
            }
        """
        if not self._loaded:
            return {
                "error": "Model not loaded. Please run training first.",
                "theatre_id": theatre_id,
                "model_name": self.model_name,
                "predictions": [],
            }

        start = pd.Timestamp(start_date)
        t_std = self._get_theatre_std(theatre_id)
        margin = 1.96 * t_std

        predictions = []
        sim_audience = list(recent_audience or [])

        for i in range(horizon_days):
            pred_date = start + timedelta(days=i)

            if features_df is not None and i < len(features_df):
                X = features_df.iloc[[i]][self.feature_names]
            else:
                X = self._prepare_features(theatre_id, pred_date, sim_audience)

            # Handle baseline model (HistoricalMeanBaseline)
            model_cls = type(self.model).__name__
            if model_cls == "HistoricalMeanBaseline":
                pred_val = float(
                    self.model.predict_for_meta(pd.DataFrame({"theatre_id": [theatre_id]}))
                )[0]
            elif model_cls == "Pipeline":
                pred_val = float(self.model.predict(X)[0])
            else:
                pred_val = float(self.model.predict(X)[0])

            pred_val = max(0.0, pred_val)
            sim_audience.append(pred_val)

            predictions.append({
                "date": pred_date.strftime("%Y-%m-%d"),
                "predicted_audience": round(pred_val, 2),
                "lower_bound": round(max(0.0, pred_val - margin), 2),
                "upper_bound": round(pred_val + margin, 2),
            })

        return {
            "theatre_id": theatre_id,
            "model_name": self.model_name,
            "confidence_note": (
                f"Prediction intervals are ±1.96σ (95% CI) based on "
                f"per-theatre validation residuals (σ≈{t_std:.1f})."
            ),
            "predictions": predictions,
        }
