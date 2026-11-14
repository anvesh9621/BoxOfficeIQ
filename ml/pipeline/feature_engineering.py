"""
feature_engineering.py
=======================
Loads attendance_clean.csv, engineers a rich feature set for the ML models,
and saves:
  - ml/data/processed/features.csv
  - ml/data/processed/feature_names.json

Feature groups created:
  Calendar  : day_of_week, is_weekend, month, day_of_month, week_of_year, quarter
  Holiday   : is_holiday (Indian public holidays 2022-2024)
  Lag       : lag_1, lag_7, lag_14  (per-theatre, no data leakage)
  Rolling   : rolling_mean_7, rolling_mean_14, rolling_std_7  (per-theatre)
  Theatre   : theatre_type_encoded, capacity_normalized

Usage:
    python ml/pipeline/feature_engineering.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CLEAN_FILE    = PROCESSED_DIR / "attendance_clean.csv"
FEATURES_FILE = PROCESSED_DIR / "features.csv"
FEAT_NAMES    = PROCESSED_DIR / "feature_names.json"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Indian public holidays 2022-2024 ──────────────────────────────────────────
# (Same set used in download_data.py for consistency)
HOLIDAYS: set[date] = {
    # 2022
    date(2022, 1, 1), date(2022, 1, 26), date(2022, 3, 1),
    date(2022, 3, 18), date(2022, 4, 10), date(2022, 4, 14), date(2022, 4, 15),
    date(2022, 5, 3), date(2022, 8, 9), date(2022, 8, 15), date(2022, 10, 2),
    date(2022, 10, 5), date(2022, 10, 24), date(2022, 11, 8), date(2022, 12, 25),
    # 2023
    date(2023, 1, 1), date(2023, 1, 14), date(2023, 1, 26), date(2023, 2, 18),
    date(2023, 3, 7), date(2023, 3, 22), date(2023, 3, 30), date(2023, 4, 4),
    date(2023, 4, 7), date(2023, 4, 14), date(2023, 5, 5), date(2023, 6, 29),
    date(2023, 8, 15), date(2023, 9, 7), date(2023, 9, 19), date(2023, 10, 2),
    date(2023, 10, 24), date(2023, 11, 13), date(2023, 11, 27), date(2023, 12, 25),
    # 2024
    date(2024, 1, 1), date(2024, 1, 14), date(2024, 1, 22), date(2024, 1, 26),
    date(2024, 2, 24), date(2024, 3, 25), date(2024, 3, 29), date(2024, 4, 9),
    date(2024, 4, 14), date(2024, 4, 17), date(2024, 4, 21), date(2024, 5, 23),
    date(2024, 6, 17), date(2024, 7, 17), date(2024, 8, 15), date(2024, 10, 2),
    date(2024, 10, 12), date(2024, 10, 13), date(2024, 11, 1), date(2024, 11, 15),
    date(2024, 12, 25),
}

# ── Theatre type ordinal encoding ─────────────────────────────────────────────
THEATRE_TYPE_MAP = {
    "SingleScreen": 0,
    "DriveIn":      1,
    "Multiplex":    2,
    "IMAX":         3,
}


def load_clean(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[ERROR] Clean file not found: {path}")
        print("        Run ml/pipeline/preprocess.py first.")
        sys.exit(1)
    df = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    print(f"Loaded {len(df):,} rows from {path.name}")
    return df


# ── Calendar features ─────────────────────────────────────────────────────────
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"]   = df["date"].dt.dayofweek          # 0=Mon, 6=Sun
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
    df["month"]         = df["date"].dt.month
    df["day_of_month"]  = df["date"].dt.day
    df["week_of_year"]  = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"]       = df["date"].dt.quarter
    # Cyclical encoding to avoid ordinal discontinuities
    df["month_sin"]     = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]     = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]       = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]       = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


# ── Holiday feature ───────────────────────────────────────────────────────────
def add_holiday_feature(df: pd.DataFrame) -> pd.DataFrame:
    holiday_dates = {pd.Timestamp(d) for d in HOLIDAYS}
    df["is_holiday"] = df["date"].isin(holiday_dates).astype(int)
    # Days to nearest holiday (forward & backward) — useful signal
    sorted_holidays = sorted(holiday_dates)
    holiday_ts = pd.to_datetime(sorted_holidays)

    def days_to_nearest_holiday(dt):
        diffs = (holiday_ts - dt).days
        abs_diffs = np.abs(diffs)
        return int(abs_diffs.min())

    df["days_to_nearest_holiday"] = df["date"].apply(days_to_nearest_holiday)
    return df


# ── Lag features (per theatre, no leakage) ────────────────────────────────────
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag features are computed per theatre (grouped by theatre_id).
    The dataframe must already be sorted by (theatre_id, date).
    """
    df = df.sort_values(["theatre_id", "date"]).copy()

    for lag in [1, 7, 14]:
        col_name = f"lag_{lag}"
        df[col_name] = df.groupby("theatre_id")["audience_count"].shift(lag)

    return df


# ── Rolling features (per theatre, no leakage) ───────────────────────────────
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling statistics computed on shifted data to prevent leakage.
    min_periods=1 avoids NaN at the start of each theatre's series.
    """
    df = df.sort_values(["theatre_id", "date"]).copy()

    # Shift by 1 before computing rolling stats to avoid using current row
    shifted = df.groupby("theatre_id")["audience_count"].shift(1)

    df["rolling_mean_7"]  = (
        shifted.groupby(df["theatre_id"])
               .transform(lambda x: x.rolling(7,  min_periods=1).mean())
    )
    df["rolling_mean_14"] = (
        shifted.groupby(df["theatre_id"])
               .transform(lambda x: x.rolling(14, min_periods=1).mean())
    )
    df["rolling_std_7"]   = (
        shifted.groupby(df["theatre_id"])
               .transform(lambda x: x.rolling(7,  min_periods=2).std())
    )

    # Fill NaN std (first few rows of each theatre)
    df["rolling_std_7"] = df["rolling_std_7"].fillna(0)

    return df


# ── Theatre-level features ────────────────────────────────────────────────────
def add_theatre_features(df: pd.DataFrame) -> pd.DataFrame:
    df["theatre_type_encoded"] = (
        df["theatre_type"].map(THEATRE_TYPE_MAP).fillna(-1).astype(int)
    )
    cap_max = df["capacity"].max()
    df["capacity_normalized"] = df["capacity"] / cap_max
    return df


# ── Online booking ratio ──────────────────────────────────────────────────────
def add_booking_ratio(df: pd.DataFrame) -> pd.DataFrame:
    total = df["audience_count"].replace(0, np.nan)
    df["online_ratio"] = (df["online_bookings"] / total).fillna(0).round(4)
    return df


# ── Build feature list ────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    # Calendar
    "day_of_week", "is_weekend", "month", "day_of_month",
    "week_of_year", "quarter",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    # Holiday
    "is_holiday", "days_to_nearest_holiday",
    # Theatre
    "theatre_type_encoded", "capacity_normalized", "capacity",
    # Bookings
    "online_ratio",
    # Lags
    "lag_1", "lag_7", "lag_14",
    # Rolling
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
]

TARGET_COLUMN = "audience_count"

ID_COLUMNS = ["date", "theatre_id", "theatre_name", "theatre_type", "city", "state"]


def main():
    print("=" * 60)
    print("BoxOfficeIQ — Feature Engineering Pipeline")
    print("=" * 60)

    df = load_clean(CLEAN_FILE)

    print("\n[1] Adding calendar features ...")
    df = add_calendar_features(df)

    print("[2] Adding holiday features ...")
    df = add_holiday_feature(df)

    print("[3] Adding lag features (per theatre) ...")
    df = add_lag_features(df)

    print("[4] Adding rolling window features (per theatre) ...")
    df = add_rolling_features(df)

    print("[5] Adding theatre-level features ...")
    df = add_theatre_features(df)

    print("[6] Adding booking ratio ...")
    df = add_booking_ratio(df)

    print("[7] Sorting by date + theatre_id ...")
    df = df.sort_values(["date", "theatre_id"]).reset_index(drop=True)

    # ── Report NaN counts (lag features will have NaN for first N rows) ───────
    nan_summary = df[FEATURE_COLUMNS].isna().sum()
    nan_nonzero = nan_summary[nan_summary > 0]
    if not nan_nonzero.empty:
        print("\n  NaN counts in feature columns (from lags, expected):")
        for col, cnt in nan_nonzero.items():
            print(f"    {col}: {cnt:,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output_cols = ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
    # Only keep columns that exist
    output_cols = [c for c in output_cols if c in df.columns]
    df_out = df[output_cols]

    df_out.to_csv(FEATURES_FILE, index=False)
    print(f"\nFeature dataset saved → {FEATURES_FILE}")
    print(f"Shape : {df_out.shape}")

    # ── Save feature names ────────────────────────────────────────────────────
    feature_meta = {
        "feature_columns":  FEATURE_COLUMNS,
        "target_column":    TARGET_COLUMN,
        "id_columns":       ID_COLUMNS,
        "total_features":   len(FEATURE_COLUMNS),
        "feature_groups": {
            "calendar":  ["day_of_week", "is_weekend", "month", "day_of_month",
                          "week_of_year", "quarter", "month_sin", "month_cos",
                          "dow_sin", "dow_cos"],
            "holiday":   ["is_holiday", "days_to_nearest_holiday"],
            "theatre":   ["theatre_type_encoded", "capacity_normalized", "capacity"],
            "bookings":  ["online_ratio"],
            "lags":      ["lag_1", "lag_7", "lag_14"],
            "rolling":   ["rolling_mean_7", "rolling_mean_14", "rolling_std_7"],
        },
    }
    with open(FEAT_NAMES, "w", encoding="utf-8") as fh:
        json.dump(feature_meta, fh, indent=2)
    print(f"Feature names saved → {FEAT_NAMES}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print("\nFeature statistics (non-null rows):")
    print(df_out[FEATURE_COLUMNS].describe().round(2).to_string())
    print("\nFeature engineering complete.")


if __name__ == "__main__":
    main()
