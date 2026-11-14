"""
BoxOfficeIQ - SHAP Explainer Wrapper
Loads pre-computed SHAP artifacts and provides explanation utilities.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
ML_DATA_DIR = ROOT / "ml" / "data" / "processed"

SHAP_VALUES_NPY = ML_DATA_DIR / "shap_values.npy"
SHAP_SUMMARY_JSON = ML_DATA_DIR / "shap_summary.json"
FEATURE_NAMES_JSON = ML_DATA_DIR / "feature_names.json"


class ShapExplainer:
    """
    Wrapper around pre-computed SHAP artefacts.
    
    Usage:
        explainer = ShapExplainer()
        global_imp = explainer.global_importance()
        explanation = explainer.explain_prediction(feature_vector)
        waterfall = explainer.waterfall_data(feature_vector)
    """

    def __init__(
        self,
        shap_values_path: Optional[str] = None,
        shap_summary_path: Optional[str] = None,
        feature_names_path: Optional[str] = None,
    ):
        self.shap_values: Optional[np.ndarray] = None
        self.shap_summary: List[Dict] = []
        self.feature_names: List[str] = []
        self._loaded = False

        svp = Path(shap_values_path) if shap_values_path else SHAP_VALUES_NPY
        ssp = Path(shap_summary_path) if shap_summary_path else SHAP_SUMMARY_JSON
        fnp = Path(feature_names_path) if feature_names_path else FEATURE_NAMES_JSON

        self._load(svp, ssp, fnp)

    def _load(self, svp: Path, ssp: Path, fnp: Path):
        try:
            if fnp.exists():
                with open(fnp) as f:
                    fn_data = json.load(f)
                if isinstance(fn_data, dict):
                    self.feature_names = fn_data.get("feature_columns", list(fn_data.keys()))
                else:
                    self.feature_names = fn_data
                logger.info(f"Loaded {len(self.feature_names)} feature names")

            if ssp.exists():
                with open(ssp) as f:
                    self.shap_summary = json.load(f)
                logger.info(f"Loaded SHAP summary: {len(self.shap_summary)} entries")

            if svp.exists():
                data = np.load(svp, allow_pickle=True)
                if data.size > 0:
                    self.shap_values = data
                    logger.info(f"Loaded SHAP values: shape={data.shape}")
                else:
                    logger.warning("SHAP values file is empty; global SHAP array unavailable")

            self._loaded = bool(self.feature_names)
        except Exception as e:
            logger.error(f"Failed to load SHAP artefacts: {e}")
            self._loaded = False

    # ── Public API ────────────────────────────────────────────────────────────

    def global_importance(self) -> List[Dict[str, Any]]:
        """
        Return feature importances sorted descending by mean absolute SHAP value.
        
        Returns:
            [{"feature": str, "importance": float}, ...]
        """
        if self.shap_summary:
            return [
                {"feature": item["feature"], "importance": item["mean_abs_shap"]}
                for item in self.shap_summary
            ]

        # Fall back to computing from shap_values if summary is unavailable
        if self.shap_values is not None and self.shap_values.ndim == 2:
            mean_abs = np.abs(self.shap_values).mean(axis=0)
            order = np.argsort(mean_abs)[::-1]
            return [
                {"feature": self.feature_names[i], "importance": round(float(mean_abs[i]), 6)}
                for i in order
            ]

        return []

    def explain_prediction(
        self, feature_vector: Union[List[float], np.ndarray, Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Explain a single prediction using SHAP values.
        
        feature_vector: array-like of length n_features, or dict {feature_name: value}.
        
        Returns:
            [{"feature": str, "value": float, "shap_value": float, "direction": "positive"|"negative"}, ...]
        """
        fv = self._coerce_feature_vector(feature_vector)
        if fv is None:
            return []

        if self.shap_values is None or self.shap_values.ndim != 2:
            # Approximate SHAP via global importance weighted by feature value deviation
            return self._approximate_explanation(fv)

        # Use mean SHAP value profile from training (closest-proxy explanation)
        # For a proper per-instance explanation, a live TreeExplainer is needed.
        # We return mean-weighted SHAP as the best available offline approximation.
        mean_shap = self.shap_values.mean(axis=0)
        result = []
        for i, fname in enumerate(self.feature_names):
            sv = float(mean_shap[i]) * float(fv[i]) / (np.abs(mean_shap).max() + 1e-8)
            result.append({
                "feature": fname,
                "value": round(float(fv[i]), 4),
                "shap_value": round(sv, 6),
                "direction": "positive" if sv >= 0 else "negative",
            })

        result.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return result

    def waterfall_data(
        self, feature_vector: Union[List[float], np.ndarray, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Produce data suitable for rendering a SHAP waterfall chart.
        
        Returns:
            {
              "base_value": float,
              "output_value": float,
              "features": [{"feature": str, "value": float, "shap_value": float}, ...],
              "positive_contributions": float,
              "negative_contributions": float,
            }
        """
        explanation = self.explain_prediction(feature_vector)
        if not explanation:
            return {}

        base_value = 0.0
        if self.shap_values is not None and self.shap_values.ndim == 2:
            # Approximate base value as mean prediction on training data
            base_value = float(np.mean(np.abs(self.shap_values).sum(axis=1)))

        pos_sum = sum(e["shap_value"] for e in explanation if e["shap_value"] > 0)
        neg_sum = sum(e["shap_value"] for e in explanation if e["shap_value"] < 0)
        output_value = base_value + pos_sum + neg_sum

        return {
            "base_value": round(base_value, 4),
            "output_value": round(output_value, 4),
            "features": [
                {
                    "feature": e["feature"],
                    "value": e["value"],
                    "shap_value": e["shap_value"],
                    "direction": e["direction"],
                }
                for e in explanation[:20]  # top 20 features for waterfall
            ],
            "positive_contributions": round(pos_sum, 4),
            "negative_contributions": round(neg_sum, 4),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _coerce_feature_vector(
        self, fv: Union[List[float], np.ndarray, Dict[str, float]]
    ) -> Optional[np.ndarray]:
        if isinstance(fv, dict):
            arr = np.array([fv.get(f, 0.0) for f in self.feature_names], dtype=np.float32)
        else:
            arr = np.array(fv, dtype=np.float32)

        if arr.shape[0] != len(self.feature_names):
            logger.warning(
                f"Feature vector length {arr.shape[0]} != expected {len(self.feature_names)}"
            )
            # Pad or trim to match
            if arr.shape[0] < len(self.feature_names):
                arr = np.pad(arr, (0, len(self.feature_names) - arr.shape[0]))
            else:
                arr = arr[: len(self.feature_names)]
        return arr

    def _approximate_explanation(self, fv: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback: use global importance as proxy SHAP values."""
        global_imp = {item["feature"]: item["importance"] for item in self.global_importance()}
        result = []
        for i, fname in enumerate(self.feature_names):
            imp = global_imp.get(fname, 0.0)
            sv = imp * np.sign(fv[i])
            result.append({
                "feature": fname,
                "value": round(float(fv[i]), 4),
                "shap_value": round(float(sv), 6),
                "direction": "positive" if sv >= 0 else "negative",
            })
        result.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return result
