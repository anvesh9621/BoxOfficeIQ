"""
BoxOfficeIQ - ML Training Script
Trains multiple models for cinema audience count forecasting.
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

from pathlib import Path
from datetime import datetime, timedelta

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

import shap

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "ml" / "data" / "processed"
MODELS_DIR = ROOT / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_CSV = DATA_DIR / "features.csv"
FEATURE_NAMES_JSON = DATA_DIR / "feature_names.json"
MODEL_COMPARISON_CSV = DATA_DIR / "model_comparison.csv"
FEATURE_IMPORTANCE_JSON = DATA_DIR / "feature_importance.json"
SHAP_VALUES_NPY = DATA_DIR / "shap_values.npy"
SHAP_FEATURE_NAMES_JSON = DATA_DIR / "shap_feature_names.json"
SHAP_SUMMARY_JSON = DATA_DIR / "shap_summary.json"
VALIDATION_PREDICTIONS_CSV = DATA_DIR / "validation_predictions.csv"
FORECASTS_7DAY_CSV = DATA_DIR / "forecasts_7day.csv"
BEST_MODEL_PKL = MODELS_DIR / "best_model.pkl"

TRAIN_CUTOFF = "2024-07-01"
VAL_END = "2024-12-31"
EXPERIMENT_NAME = "BoxOfficeIQ_ML"
MLFLOW_URI = str(ROOT / "mlflow.db")

# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "WAPE": round(wape, 4), "R2": round(r2, 4)}


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data():
    logger.info(f"Loading features from {FEATURES_CSV}")
    df = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    df = df.sort_values(["theatre_id", "date"]).reset_index(drop=True)

    with open(FEATURE_NAMES_JSON) as f:
        fn_data = json.load(f)
    if isinstance(fn_data, dict):
        feature_names = fn_data.get("feature_columns", list(fn_data.keys()))
    else:
        feature_names = fn_data

    # Drop initial rows per theatre where lag features are NaN
    init_len = len(df)
    df = df.dropna(subset=feature_names + ["audience_count"]).reset_index(drop=True)
    logger.info(f"Dropped {init_len - len(df)} NaN lag rows. Clean dataset shape: {df.shape}, features: {len(feature_names)}")
    return df, feature_names


def split_data(df, feature_names):
    train_mask = df["date"] < TRAIN_CUTOFF
    val_mask = (df["date"] >= TRAIN_CUTOFF) & (df["date"] <= VAL_END)

    X_train = df.loc[train_mask, feature_names].copy()
    y_train = df.loc[train_mask, "audience_count"].copy()
    X_val = df.loc[val_mask, feature_names].copy()
    y_val = df.loc[val_mask, "audience_count"].copy()
    meta_val = df.loc[val_mask, ["date", "theatre_id"]].copy()

    logger.info(f"Train: {len(X_train)} rows, Val: {len(X_val)} rows")
    return X_train, y_train, X_val, y_val, meta_val, train_mask, val_mask


# ─── Baseline ─────────────────────────────────────────────────────────────────

class HistoricalMeanBaseline:
    """Naive baseline: predict the per-theatre training mean."""

    def __init__(self):
        self.theatre_means_ = {}
        self.global_mean_ = 0.0
        self.name = "Baseline (Historical Mean)"

    def fit(self, df_train):
        self.theatre_means_ = df_train.groupby("theatre_id")["audience_count"].mean().to_dict()
        self.global_mean_ = df_train["audience_count"].mean()
        return self

    def predict_for_meta(self, meta_val):
        preds = meta_val["theatre_id"].map(self.theatre_means_).fillna(self.global_mean_)
        return preds.values


# ─── Training ─────────────────────────────────────────────────────────────────

def train_models(df, feature_names):
    X_train, y_train, X_val, y_val, meta_val, train_mask, val_mask = split_data(df, feature_names)
    df_train = df.loc[train_mask].copy()
    df_val = df.loc[val_mask].copy()

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_URI}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    all_metrics = []
    all_val_preds = []
    trained_models = {}

    # ── 1. Baseline ───────────────────────────────────────────────────────────
    logger.info("Training Baseline...")
    with mlflow.start_run(run_name="Baseline"):
        baseline = HistoricalMeanBaseline().fit(df_train)
        preds_baseline = baseline.predict_for_meta(meta_val)
        m = compute_metrics(y_val.values, preds_baseline)
        mlflow.log_metrics(m)
        all_metrics.append({"model": "Baseline", **m})
        trained_models["Baseline"] = baseline
        _append_val_preds(all_val_preds, meta_val, y_val, preds_baseline, "Baseline")
        logger.info(f"Baseline metrics: {m}")

    # ── 2. Ridge Regression ───────────────────────────────────────────────────
    logger.info("Training Ridge Regression...")
    with mlflow.start_run(run_name="Ridge"):
        ridge = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ])
        ridge.fit(X_train, y_train)
        preds_ridge = ridge.predict(X_val)
        m = compute_metrics(y_val.values, preds_ridge)
        mlflow.log_metrics(m)
        all_metrics.append({"model": "Ridge", **m})
        trained_models["Ridge"] = ridge
        _append_val_preds(all_val_preds, meta_val, y_val, preds_ridge, "Ridge")
        logger.info(f"Ridge metrics: {m}")

    # ── 3. Random Forest ──────────────────────────────────────────────────────
    logger.info("Training Random Forest...")
    with mlflow.start_run(run_name="RandomForest"):
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        preds_rf = rf.predict(X_val)
        m = compute_metrics(y_val.values, preds_rf)
        mlflow.log_metrics(m)
        all_metrics.append({"model": "RandomForest", **m})
        trained_models["RandomForest"] = rf
        _append_val_preds(all_val_preds, meta_val, y_val, preds_rf, "RandomForest")
        logger.info(f"RandomForest metrics: {m}")

    # ── 4. XGBoost ────────────────────────────────────────────────────────────
    logger.info("Training XGBoost...")
    with mlflow.start_run(run_name="XGBoost"):
        xgb_model = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds_xgb = xgb_model.predict(X_val)
        m = compute_metrics(y_val.values, preds_xgb)
        mlflow.log_metrics(m)
        all_metrics.append({"model": "XGBoost", **m})
        trained_models["XGBoost"] = xgb_model
        _append_val_preds(all_val_preds, meta_val, y_val, preds_xgb, "XGBoost")
        logger.info(f"XGBoost metrics: {m}")

    # ── 5. LightGBM ───────────────────────────────────────────────────────────
    logger.info("Training LightGBM...")
    with mlflow.start_run(run_name="LightGBM"):
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)]
        )
        preds_lgb = lgb_model.predict(X_val)
        m = compute_metrics(y_val.values, preds_lgb)
        mlflow.log_metrics(m)
        all_metrics.append({"model": "LightGBM", **m})
        trained_models["LightGBM"] = lgb_model
        _append_val_preds(all_val_preds, meta_val, y_val, preds_lgb, "LightGBM")
        logger.info(f"LightGBM metrics: {m}")

    # ── 6. CatBoost ───────────────────────────────────────────────────────────
    logger.info("Training CatBoost...")
    with mlflow.start_run(run_name="CatBoost"):
        cat_model = CatBoostRegressor(
            iterations=200, depth=6, learning_rate=0.1,
            random_seed=42, verbose=0
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        preds_cat = cat_model.predict(X_val)
        m = compute_metrics(y_val.values, preds_cat)
        mlflow.log_metrics(m)
        all_metrics.append({"model": "CatBoost", **m})
        trained_models["CatBoost"] = cat_model
        _append_val_preds(all_val_preds, meta_val, y_val, preds_cat, "CatBoost")
        logger.info(f"CatBoost metrics: {m}")

    return all_metrics, all_val_preds, trained_models, X_train, y_train, X_val, y_val, meta_val, df_train, df_val, feature_names


def _append_val_preds(all_val_preds, meta_val, y_val, preds, model_name):
    tmp = meta_val.copy().reset_index(drop=True)
    tmp["actual"] = y_val.values
    tmp["predicted"] = preds
    tmp["model_name"] = model_name
    all_val_preds.append(tmp)


# ─── Post-training analysis ───────────────────────────────────────────────────

def select_best_model(all_metrics, trained_models):
    df_metrics = pd.DataFrame(all_metrics)
    best_row = df_metrics.loc[df_metrics["MAE"].idxmin()]
    best_name = best_row["model"]
    logger.info(f"Best model by MAE: {best_name} (MAE={best_row['MAE']:.4f})")
    return best_name, trained_models[best_name], df_metrics


def save_feature_importance(best_name, best_model, feature_names):
    tree_models = {"RandomForest", "XGBoost", "LightGBM", "CatBoost"}
    if best_name not in tree_models:
        # Find best tree model for feature importance
        for name in ["LightGBM", "XGBoost", "CatBoost", "RandomForest"]:
            # Can't access trained_models here, handled in main
            pass
        logger.info("Best model is not a tree model; skipping feature importance")
        return

    model = best_model
    if best_name == "RandomForest":
        importances = model.feature_importances_
    elif best_name == "XGBoost":
        importances = model.feature_importances_
    elif best_name == "LightGBM":
        importances = model.feature_importances_
    elif best_name == "CatBoost":
        importances = model.get_feature_importance()

    fi = sorted(zip(feature_names, importances.tolist()), key=lambda x: x[1], reverse=True)
    fi_dict = [{"feature": f, "importance": round(v, 6)} for f, v in fi]
    with open(FEATURE_IMPORTANCE_JSON, "w") as fh:
        json.dump(fi_dict, fh, indent=2)
    logger.info(f"Feature importance saved ({len(fi_dict)} features)")


def save_shap(best_name, best_model, X_train, feature_names):
    """Compute SHAP values for XGBoost or LightGBM."""
    shap_model = None
    shap_model_name = best_name

    if best_name in ("XGBoost", "LightGBM"):
        shap_model = best_model
    else:
        logger.info(f"Best model is {best_name}; SHAP not supported for this type. Skipping SHAP.")
        # Save empty placeholders
        np.save(SHAP_VALUES_NPY, np.array([]))
        with open(SHAP_FEATURE_NAMES_JSON, "w") as fh:
            json.dump(feature_names, fh)
        with open(SHAP_SUMMARY_JSON, "w") as fh:
            json.dump([], fh)
        return

    logger.info(f"Computing SHAP values for {shap_model_name}...")
    # Use a sample for SHAP to keep runtime reasonable
    sample_size = min(500, len(X_train))
    X_sample = X_train.sample(n=sample_size, random_state=42)

    if shap_model_name == "XGBoost":
        explainer = shap.TreeExplainer(shap_model)
    else:
        explainer = shap.TreeExplainer(shap_model)

    shap_values = explainer.shap_values(X_sample)

    np.save(SHAP_VALUES_NPY, shap_values)
    with open(SHAP_FEATURE_NAMES_JSON, "w") as fh:
        json.dump(feature_names, fh)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:10]
    shap_summary = [
        {"feature": feature_names[i], "mean_abs_shap": round(float(mean_abs_shap[i]), 6)}
        for i in top_indices
    ]
    with open(SHAP_SUMMARY_JSON, "w") as fh:
        json.dump(shap_summary, fh, indent=2)

    logger.info(f"SHAP values saved. Top feature: {shap_summary[0]['feature']}")


def generate_7day_forecast(best_name, best_model, df, feature_names):
    """Generate 7-day forecast for all theatres beyond 2024-12-31."""
    last_date = pd.Timestamp("2024-12-31")
    future_dates = [last_date + timedelta(days=i + 1) for i in range(7)]
    theatres = df["theatre_id"].unique()

    # Build validation residuals per theatre for prediction intervals
    val_mask = (df["date"] >= TRAIN_CUTOFF) & (df["date"] <= VAL_END)
    df_val = df.loc[val_mask].copy()

    # We need the actual validation predictions of the best model
    # We'll recompute quickly for the best model
    X_val = df_val[feature_names]

    if best_name == "Baseline":
        train_mask = df["date"] < TRAIN_CUTOFF
        df_train = df.loc[train_mask]
        val_preds = best_model.predict_for_meta(df_val[["theatre_id"]])
    elif best_name == "Ridge":
        val_preds = best_model.predict(X_val)
    else:
        val_preds = best_model.predict(X_val)

    residuals = df_val["audience_count"].values - val_preds
    residual_df = df_val[["theatre_id"]].copy()
    residual_df["residual"] = residuals
    theatre_std = residual_df.groupby("theatre_id")["residual"].std().fillna(0)
    global_std = residual_df["residual"].std()

    # Build feature rows for future dates using last known row per theatre
    forecast_rows = []
    for theatre in theatres:
        theatre_data = df[df["theatre_id"] == theatre].sort_values("date")
        last_row = theatre_data.iloc[-1]
        t_std = theatre_std.get(theatre, global_std)

        for fdate in future_dates:
            # Build feature vector based on date features
            # Lag features will use last known values (simplification)
            row = {}
            for feat in feature_names:
                if feat == "day_of_week":
                    row[feat] = fdate.dayofweek
                elif feat == "is_weekend":
                    row[feat] = int(fdate.dayofweek >= 5)
                elif feat == "month":
                    row[feat] = fdate.month
                elif feat == "day_of_month":
                    row[feat] = fdate.day
                elif feat == "week_of_year":
                    row[feat] = fdate.isocalendar().week
                elif feat == "quarter":
                    row[feat] = fdate.quarter
                elif feat == "is_holiday":
                    row[feat] = 0  # conservative assumption
                else:
                    row[feat] = last_row.get(feat, 0)

            X_future = pd.DataFrame([row])[feature_names]

            if best_name == "Baseline":
                pred = best_model.predict_for_meta(pd.DataFrame({"theatre_id": [theatre]}))
                pred_val = float(pred[0])
            elif best_name == "Ridge":
                pred_val = float(best_model.predict(X_future)[0])
            else:
                pred_val = float(best_model.predict(X_future)[0])

            pred_val = max(0.0, pred_val)
            margin = 1.96 * float(t_std)
            forecast_rows.append({
                "date": fdate.strftime("%Y-%m-%d"),
                "theatre_id": theatre,
                "predicted_audience": round(pred_val, 2),
                "lower_bound": round(max(0.0, pred_val - margin), 2),
                "upper_bound": round(pred_val + margin, 2),
            })

    df_forecast = pd.DataFrame(forecast_rows)
    df_forecast.to_csv(FORECASTS_7DAY_CSV, index=False)
    logger.info(f"7-day forecast saved: {len(df_forecast)} rows")


# ─── Save feature importance (helper for non-best tree) ───────────────────────

def save_feature_importance_any(best_name, best_model, trained_models, feature_names):
    """Save feature importance, falling back to any tree model."""
    tree_models = {"RandomForest", "XGBoost", "LightGBM", "CatBoost"}

    model_to_use = None
    name_to_use = None

    if best_name in tree_models:
        model_to_use = best_model
        name_to_use = best_name
    else:
        for tn in ["LightGBM", "XGBoost", "CatBoost", "RandomForest"]:
            if tn in trained_models:
                model_to_use = trained_models[tn]
                name_to_use = tn
                break

    if model_to_use is None:
        logger.info("No tree model available for feature importance.")
        return

    if name_to_use == "RandomForest":
        importances = model_to_use.feature_importances_
    elif name_to_use == "XGBoost":
        importances = model_to_use.feature_importances_
    elif name_to_use == "LightGBM":
        importances = model_to_use.feature_importances_
    elif name_to_use == "CatBoost":
        importances = model_to_use.get_feature_importance()

    fi = sorted(zip(feature_names, importances.tolist()), key=lambda x: x[1], reverse=True)
    fi_dict = [{"feature": f, "importance": round(v, 6)} for f, v in fi]
    with open(FEATURE_IMPORTANCE_JSON, "w") as fh:
        json.dump(fi_dict, fh, indent=2)
    logger.info(f"Feature importance saved from {name_to_use} ({len(fi_dict)} features)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    df, feature_names = load_data()

    all_metrics, all_val_preds, trained_models, X_train, y_train, X_val, y_val, meta_val, df_train, df_val, feature_names = train_models(df, feature_names)

    # Save model comparison
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(MODEL_COMPARISON_CSV, index=False)
    logger.info(f"Model comparison saved:\n{df_metrics.to_string(index=False)}")

    # Select best model
    best_name, best_model, _ = select_best_model(all_metrics, trained_models)

    # Save best model
    joblib.dump({"model": best_model, "model_name": best_name, "feature_names": feature_names}, BEST_MODEL_PKL)
    logger.info(f"Best model saved to {BEST_MODEL_PKL}")

    # Save feature importance
    save_feature_importance_any(best_name, best_model, trained_models, feature_names)

    # SHAP analysis (use best tree model: prefer XGBoost or LightGBM)
    shap_model = best_model
    shap_name = best_name
    if best_name not in ("XGBoost", "LightGBM"):
        for tn in ["XGBoost", "LightGBM"]:
            if tn in trained_models:
                shap_model = trained_models[tn]
                shap_name = tn
                break
    save_shap(shap_name, shap_model, X_train, feature_names)

    # Save validation predictions
    df_val_preds = pd.concat(all_val_preds, ignore_index=True)
    df_val_preds["date"] = df_val_preds["date"].astype(str)
    df_val_preds.to_csv(VALIDATION_PREDICTIONS_CSV, index=False)
    logger.info(f"Validation predictions saved: {len(df_val_preds)} rows")

    # 7-day forecast
    generate_7day_forecast(best_name, best_model, df, feature_names)

    # Sync generated files to root data/processed if it exists
    root_data_proc = ROOT / "data" / "processed"
    if root_data_proc.exists():
        import shutil
        for fn in [
            "model_comparison.csv", "feature_importance.json", "shap_values.npy",
            "shap_feature_names.json", "shap_summary.json", "validation_predictions.csv",
            "forecasts_7day.csv"
        ]:
            src = DATA_DIR / fn
            if src.exists():
                shutil.copy2(src, root_data_proc / fn)
        logger.info(f"Synced generated artifacts to {root_data_proc}")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best model: {best_name}")
    best_m = next(m for m in all_metrics if m["model"] == best_name)
    logger.info(f"MAE={best_m['MAE']}, RMSE={best_m['RMSE']}, WAPE={best_m['WAPE']}%, R2={best_m['R2']}")
    logger.info("=" * 60)

    return best_name, best_m


if __name__ == "__main__":
    main()
