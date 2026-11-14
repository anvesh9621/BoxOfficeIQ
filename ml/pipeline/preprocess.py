"""
preprocess.py
=============
Loads cinema_attendance.csv, cleans and standardises the data, and saves
the result to ml/data/processed/attendance_clean.csv.

Steps performed:
  1. Load raw CSV
  2. Parse & validate date column
  3. Convert numeric columns to correct dtypes
  4. Remove fully duplicate rows
  5. Drop rows with missing critical values
  6. Sort by date + theatre_id
  7. Reset index and save

Usage:
    python ml/pipeline/preprocess.py
"""

from pathlib import Path
import sys

import pandas as pd
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parents[2]
RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_FILE      = RAW_DIR / "cinema_attendance.csv"
CLEAN_FILE    = PROCESSED_DIR / "attendance_clean.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Column dtypes ─────────────────────────────────────────────────────────────
INT_COLS = [
    "theatre_id", "capacity", "audience_count",
    "online_bookings", "offline_bookings", "is_holiday",
]
FLOAT_COLS = ["occupancy_rate"]
STR_COLS   = ["theatre_name", "theatre_type", "city", "state"]


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[ERROR] Raw file not found: {path}")
        print("        Run ml/pipeline/download_data.py first.")
        sys.exit(1)
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df):,} rows from {path.name}")
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad = df["date"].isna().sum()
    if bad:
        print(f"  [WARN] Dropping {bad} rows with unparseable dates")
        df = df.dropna(subset=["date"])
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in STR_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    critical = ["date", "theatre_id", "audience_count"]
    before = len(df)
    df = df.dropna(subset=critical)
    dropped = before - len(df)
    if dropped:
        print(f"  [INFO] Dropped {dropped} rows missing critical fields")

    # Fill non-critical numeric nulls with 0
    for col in INT_COLS + FLOAT_COLS:
        if col in df.columns:
            null_cnt = df[col].isna().sum()
            if null_cnt:
                print(f"  [INFO] Filling {null_cnt} nulls in '{col}' with 0")
                df[col] = df[col].fillna(0)

    # Fill string nulls with 'Unknown'
    for col in STR_COLS:
        if col in df.columns:
            null_cnt = df[col].isna().sum()
            if null_cnt:
                print(f"  [INFO] Filling {null_cnt} nulls in '{col}' with 'Unknown'")
                df[col] = df[col].fillna("Unknown")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    full_dups = before - len(df)

    # Remove key-level duplicates (keep first occurrence)
    before2 = len(df)
    df = df.drop_duplicates(subset=["date", "theatre_id"], keep="first")
    key_dups = before2 - len(df)

    if full_dups:
        print(f"  [INFO] Removed {full_dups} fully duplicate rows")
    if key_dups:
        print(f"  [INFO] Removed {key_dups} key-duplicate (date, theatre_id) rows")
    if not full_dups and not key_dups:
        print("  [INFO] No duplicates found")
    return df


def enforce_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Clip nonsensical values rather than dropping them."""
    for col in ["audience_count", "online_bookings", "offline_bookings", "capacity"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg:
                print(f"  [WARN] Clipping {neg} negative values in '{col}' to 0")
            df[col] = df[col].clip(lower=0)

    if "occupancy_rate" in df.columns:
        bad = ((df["occupancy_rate"] < 0) | (df["occupancy_rate"] > 1)).sum()
        if bad:
            print(f"  [WARN] Clipping {bad} out-of-range occupancy_rate values to [0,1]")
        df["occupancy_rate"] = df["occupancy_rate"].clip(0, 1)

    if "is_holiday" in df.columns:
        df["is_holiday"] = df["is_holiday"].clip(0, 1)

    return df


def sort_and_reset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "theatre_id"]).reset_index(drop=True)
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few simple derived columns that are always safe to compute."""
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Mon … 6=Sun
    return df


def main():
    print("=" * 60)
    print("BoxOfficeIQ — Preprocessing Pipeline")
    print("=" * 60)

    df = load_raw(RAW_FILE)
    print(f"\nInitial shape : {df.shape}")

    print("\n[1] Parsing dates ...")
    df = parse_dates(df)

    print("[2] Fixing dtypes ...")
    df = fix_dtypes(df)

    print("[3] Handling missing values ...")
    df = handle_missing(df)

    print("[4] Removing duplicates ...")
    df = remove_duplicates(df)

    print("[5] Enforcing value ranges ...")
    df = enforce_ranges(df)

    print("[6] Sorting & resetting index ...")
    df = sort_and_reset(df)

    print("[7] Adding derived columns ...")
    df = add_derived_columns(df)

    df.to_csv(CLEAN_FILE, index=False)
    print(f"\nCleaned data saved → {CLEAN_FILE}")
    print(f"Final shape : {df.shape}")
    print("\nColumn dtypes:")
    print(df.dtypes.to_string())
    print("\nSample rows:")
    print(df.head(3).to_string())
    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
