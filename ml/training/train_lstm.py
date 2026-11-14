"""
BoxOfficeIQ - LSTM Training Script
PyTorch LSTM model for cinema audience count time-series forecasting.
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

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
LSTM_MODEL_PT = MODELS_DIR / "lstm_model.pt"
LSTM_METRICS_JSON = DATA_DIR / "lstm_metrics.json"
LSTM_PREDICTIONS_CSV = DATA_DIR / "lstm_predictions.csv"

TRAIN_CUTOFF = "2024-07-01"
VAL_END = "2024-12-31"
SEQUENCE_LENGTH = 28
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 32
MAX_EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 0.001
EXPERIMENT_NAME = "BoxOfficeIQ_LSTM"
MLFLOW_URI = str(ROOT / "mlflow.db")


# ─── Dataset ──────────────────────────────────────────────────────────────────

class CinemaDataset(Dataset):
    """Sliding window dataset over sorted theatre time-series."""

    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def build_sequences(df, feature_names, seq_len=SEQUENCE_LENGTH):
    """Build overlapping sequences per theatre."""
    sequences, targets, meta = [], [], []
    theatres = df["theatre_id"].unique()

    for theatre in theatres:
        tdf = df[df["theatre_id"] == theatre].sort_values("date").reset_index(drop=True)
        X = tdf[feature_names].values.astype(np.float32)
        y = tdf["audience_count"].values.astype(np.float32)
        dates = tdf["date"].values

        for i in range(seq_len, len(tdf)):
            sequences.append(X[i - seq_len:i])
            targets.append(y[i])
            meta.append({"date": str(dates[i])[:10], "theatre_id": theatre})

    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32), meta


# ─── Model ────────────────────────────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # last time step
        return self.fc(out).squeeze(-1)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    wape = float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "WAPE": round(wape, 4),
        "R2": round(r2, 4)
    }


# ─── Normalisation ────────────────────────────────────────────────────────────

class Normalizer:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def normalize_target(y_train):
    mean = y_train.mean()
    std = y_train.std() + 1e-8
    return (y_train - mean) / std, mean, std


# ─── Training loop ────────────────────────────────────────────────────────────

def train_lstm(train_loader, val_loader, model, device, target_mean, target_std):
    criterion = nn.HuberLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch}/{MAX_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info(f"Loading features from {FEATURES_CSV}")
    df = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    df = df.sort_values(["theatre_id", "date"]).reset_index(drop=True)

    with open(FEATURE_NAMES_JSON) as f:
        fn_data = json.load(f)
    if isinstance(fn_data, dict):
        feature_names = fn_data.get("feature_columns", list(fn_data.keys()))
    else:
        feature_names = fn_data

    # Forward-fill / back-fill NaNs per theatre for sequence models
    df[feature_names] = df.groupby("theatre_id")[feature_names].bfill().ffill().fillna(0)

    input_size = len(feature_names)
    logger.info(f"Dataset: {len(df)} rows, {input_size} features")

    # Split
    train_df = df[df["date"] < TRAIN_CUTOFF].copy()
    val_df = df[(df["date"] >= TRAIN_CUTOFF) & (df["date"] <= VAL_END)].copy()
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Build sequences from full df (val sequences need train context)
    logger.info("Building sequences...")
    X_all, y_all, meta_all = build_sequences(df, feature_names, SEQUENCE_LENGTH)

    # Determine split index based on dates
    meta_dates = [m["date"] for m in meta_all]
    train_indices = [i for i, d in enumerate(meta_dates) if d < TRAIN_CUTOFF]
    val_indices = [i for i, d in enumerate(meta_dates) if TRAIN_CUTOFF <= d <= VAL_END]

    X_train_seq = X_all[train_indices]
    y_train_seq = y_all[train_indices]
    X_val_seq = X_all[val_indices]
    y_val_seq = y_all[val_indices]
    meta_val = [meta_all[i] for i in val_indices]

    logger.info(f"Train sequences: {len(X_train_seq)}, Val sequences: {len(X_val_seq)}")

    # Normalize features
    n_samples, seq_len, n_feats = X_train_seq.shape
    normalizer = Normalizer()
    X_train_flat = X_train_seq.reshape(-1, n_feats)
    normalizer.fit(X_train_flat)

    X_train_norm = normalizer.transform(X_train_seq.reshape(-1, n_feats)).reshape(X_train_seq.shape)
    X_val_norm = normalizer.transform(X_val_seq.reshape(-1, n_feats)).reshape(X_val_seq.shape)

    # Normalize target
    y_train_norm, target_mean, target_std = normalize_target(y_train_seq)
    y_val_norm = (y_val_seq - target_mean) / (target_std + 1e-8)

    # DataLoaders
    train_dataset = CinemaDataset(X_train_norm, y_train_norm)
    val_dataset = CinemaDataset(X_val_norm, y_val_norm)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    model = LSTMForecaster(input_size=input_size).to(device)
    logger.info(f"Model: {model}")

    # MLflow
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_URI}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="LSTM"):
        mlflow.log_params({
            "sequence_length": SEQUENCE_LENGTH,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
        })

        model, history = train_lstm(train_loader, val_loader, model, device, target_mean, target_std)

        # Evaluate on validation set (denormalise predictions)
        model.eval()
        all_preds_norm = []
        with torch.no_grad():
            for Xb, _ in val_loader:
                Xb = Xb.to(device)
                preds = model(Xb)
                all_preds_norm.append(preds.cpu().numpy())

        y_pred_norm = np.concatenate(all_preds_norm)
        y_pred = y_pred_norm * target_std + target_mean
        y_pred = np.maximum(y_pred, 0)

        metrics = compute_metrics(y_val_seq, y_pred)
        mlflow.log_metrics(metrics)
        logger.info(f"LSTM Metrics: {metrics}")

    # Save model
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "sequence_length": SEQUENCE_LENGTH,
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "feature_names": feature_names,
        "normalizer_mean": normalizer.mean_.tolist(),
        "normalizer_std": normalizer.std_.tolist(),
    }, LSTM_MODEL_PT)
    logger.info(f"LSTM model saved to {LSTM_MODEL_PT}")

    # Save metrics
    with open(LSTM_METRICS_JSON, "w") as fh:
        json.dump({"model": "LSTM", **metrics}, fh, indent=2)

    # Save predictions
    pred_rows = []
    for i, m in enumerate(meta_val):
        pred_rows.append({
            "date": m["date"],
            "theatre_id": m["theatre_id"],
            "actual": float(y_val_seq[i]),
            "predicted": float(y_pred[i]),
            "model_name": "LSTM"
        })
    pd.DataFrame(pred_rows).to_csv(LSTM_PREDICTIONS_CSV, index=False)
    logger.info(f"LSTM predictions saved: {len(pred_rows)} rows")

    # Sync to root data/processed if it exists
    root_data_proc = ROOT / "data" / "processed"
    if root_data_proc.exists():
        import shutil
        for fn in ["lstm_metrics.json", "lstm_predictions.csv"]:
            src = DATA_DIR / fn
            if src.exists():
                shutil.copy2(src, root_data_proc / fn)
        logger.info(f"Synced LSTM artifacts to {root_data_proc}")

    logger.info("=" * 60)
    logger.info("LSTM TRAINING COMPLETE")
    logger.info(f"MAE={metrics['MAE']}, RMSE={metrics['RMSE']}, WAPE={metrics['WAPE']}%, R2={metrics['R2']}")
    logger.info("=" * 60)

    return metrics


if __name__ == "__main__":
    main()
