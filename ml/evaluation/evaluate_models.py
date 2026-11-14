"""
BoxOfficeIQ - Model Evaluation Script
Combines ML and LSTM results into a unified comparison table.
"""

import json
import logging
import pandas as pd
from pathlib import Path

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "ml" / "data" / "processed"

MODEL_COMPARISON_CSV = DATA_DIR / "model_comparison.csv"
LSTM_METRICS_JSON = DATA_DIR / "lstm_metrics.json"
ALL_MODEL_COMPARISON_JSON = DATA_DIR / "all_model_comparison.json"


def load_ml_metrics():
    df = pd.read_csv(MODEL_COMPARISON_CSV)
    return df


def load_lstm_metrics():
    with open(LSTM_METRICS_JSON) as f:
        d = json.load(f)
    return pd.DataFrame([d])


def main():
    logger.info("Loading ML model metrics...")
    df_ml = load_ml_metrics()

    logger.info("Loading LSTM metrics...")
    df_lstm = load_lstm_metrics()

    # Harmonise columns
    for df in [df_ml, df_lstm]:
        for col in ["model", "MAE", "RMSE", "WAPE", "R2"]:
            if col not in df.columns:
                df[col] = None

    df_all = pd.concat([df_ml, df_lstm], ignore_index=True)
    df_all = df_all[["model", "MAE", "RMSE", "WAPE", "R2"]]
    df_all = df_all.sort_values("MAE").reset_index(drop=True)

    # Print formatted table
    print("\n" + "=" * 70)
    print("  BoxOfficeIQ — Model Comparison (sorted by MAE, lower is better)")
    print("=" * 70)
    if HAS_TABULATE:
        table_str = tabulate(
            df_all.values.tolist(),
            headers=df_all.columns.tolist(),
            floatfmt=".4f",
            tablefmt="github"
        )
    else:
        table_str = df_all.to_string(index=False)
    print(table_str)
    print("=" * 70)
    print(f"\nBest model: {df_all.iloc[0]['model']}  |  MAE={df_all.iloc[0]['MAE']:.4f}  |  RMSE={df_all.iloc[0]['RMSE']:.4f}  |  WAPE={df_all.iloc[0]['WAPE']:.4f}%")
    print()

    # Save JSON
    result = df_all.to_dict(orient="records")
    with open(ALL_MODEL_COMPARISON_JSON, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info(f"All model comparison saved to {ALL_MODEL_COMPARISON_JSON}")

    # Sync to root data/processed if it exists
    root_data_proc = ROOT / "data" / "processed"
    if root_data_proc.exists():
        import shutil
        shutil.copy2(ALL_MODEL_COMPARISON_JSON, root_data_proc / "all_model_comparison.json")
        logger.info(f"Synced all_model_comparison.json to {root_data_proc}")

    return result


if __name__ == "__main__":
    main()
