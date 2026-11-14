"""
validate_data.py
================
Validates cinema_attendance.csv for schema correctness, value ranges,
duplicates, and missing values. Prints a detailed report.

Usage:
    python ml/pipeline/validate_data.py
"""

from pathlib import Path
import sys

import pandas as pd
import numpy as np

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
ATTENDANCE_FILE = RAW_DIR / "cinema_attendance.csv"

REQUIRED_COLUMNS = [
    "date",
    "theatre_id",
    "theatre_name",
    "theatre_type",
    "city",
    "state",
    "capacity",
    "audience_count",
    "online_bookings",
    "offline_bookings",
    "occupancy_rate",
    "is_holiday",
]

THEATRE_TYPES = {"Multiplex", "SingleScreen", "IMAX", "DriveIn"}


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check_pass(msg: str):
    print(f"  [PASS] {msg}")


def check_fail(msg: str):
    print(f"  [FAIL] {msg}")


def check_warn(msg: str):
    print(f"  [WARN] {msg}")


def validate():
    print("BoxOfficeIQ — Data Validation Report")
    print(f"File: {ATTENDANCE_FILE}")

    # ── File existence ────────────────────────────────────────────────────────
    section("1. File Existence")
    if not ATTENDANCE_FILE.exists():
        check_fail(f"File not found: {ATTENDANCE_FILE}")
        sys.exit(1)
    check_pass(f"File exists: {ATTENDANCE_FILE.name}")

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(ATTENDANCE_FILE)
    section("2. Shape & Load")
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {len(df.columns)}")
    print(f"  Columns : {list(df.columns)}")

    # ── Schema ────────────────────────────────────────────────────────────────
    section("3. Schema Check (required columns)")
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra_cols   = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if not missing_cols:
        check_pass("All required columns present")
    else:
        check_fail(f"Missing columns: {missing_cols}")
    if extra_cols:
        check_warn(f"Extra (unexpected) columns: {extra_cols}")

    # ── Date parsing ──────────────────────────────────────────────────────────
    section("4. Date Parsing")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = df["date"].isna().sum()
    if bad_dates == 0:
        check_pass("All dates parsed successfully")
    else:
        check_fail(f"{bad_dates} rows have invalid dates")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    # ── Missing values ────────────────────────────────────────────────────────
    section("5. Missing Values")
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    any_nulls = False
    for col, cnt in null_counts.items():
        if cnt > 0:
            check_fail(f"'{col}' has {cnt:,} null values")
            any_nulls = True
    if not any_nulls:
        check_pass("No missing values in any required column")

    # ── Duplicate rows ────────────────────────────────────────────────────────
    section("6. Duplicate Rows")
    dup_full = df.duplicated().sum()
    dup_key  = df.duplicated(subset=["date", "theatre_id"]).sum()
    if dup_full == 0:
        check_pass("No fully duplicate rows")
    else:
        check_fail(f"{dup_full:,} fully duplicate rows found")
    if dup_key == 0:
        check_pass("No duplicate (date, theatre_id) keys")
    else:
        check_fail(f"{dup_key:,} duplicate (date, theatre_id) keys found")

    # ── Range checks ──────────────────────────────────────────────────────────
    section("7. Value Range Checks")

    # audience_count >= 0
    neg_audience = (df["audience_count"] < 0).sum()
    if neg_audience == 0:
        check_pass("audience_count >= 0 for all rows")
    else:
        check_fail(f"{neg_audience:,} rows have audience_count < 0")

    # online_bookings >= 0
    neg_online = (df["online_bookings"] < 0).sum()
    if neg_online == 0:
        check_pass("online_bookings >= 0 for all rows")
    else:
        check_fail(f"{neg_online:,} rows have online_bookings < 0")

    # offline_bookings >= 0
    neg_offline = (df["offline_bookings"] < 0).sum()
    if neg_offline == 0:
        check_pass("offline_bookings >= 0 for all rows")
    else:
        check_fail(f"{neg_offline:,} rows have offline_bookings < 0")

    # audience_count == online + offline
    booking_mismatch = (
        df["audience_count"] != df["online_bookings"] + df["offline_bookings"]
    ).sum()
    if booking_mismatch == 0:
        check_pass("audience_count == online_bookings + offline_bookings for all rows")
    else:
        check_fail(f"{booking_mismatch:,} rows where booking sum != audience_count")

    # occupancy_rate in [0, 1]
    occ_bad = ((df["occupancy_rate"] < 0) | (df["occupancy_rate"] > 1)).sum()
    if occ_bad == 0:
        check_pass("occupancy_rate in [0, 1] for all rows")
    else:
        check_fail(f"{occ_bad:,} rows have occupancy_rate outside [0, 1]")

    # capacity > 0
    cap_bad = (df["capacity"] <= 0).sum()
    if cap_bad == 0:
        check_pass("capacity > 0 for all rows")
    else:
        check_fail(f"{cap_bad:,} rows have capacity <= 0")

    # is_holiday in {0, 1}
    hol_bad = (~df["is_holiday"].isin([0, 1])).sum()
    if hol_bad == 0:
        check_pass("is_holiday values are all 0 or 1")
    else:
        check_fail(f"{hol_bad:,} rows have invalid is_holiday values")

    # ── Theatre type validation ───────────────────────────────────────────────
    section("8. Categorical Validation")
    invalid_types = df[~df["theatre_type"].isin(THEATRE_TYPES)]["theatre_type"].unique()
    if len(invalid_types) == 0:
        check_pass(f"All theatre_type values are valid: {sorted(THEATRE_TYPES)}")
    else:
        check_fail(f"Invalid theatre_type values: {invalid_types}")

    n_theatres = df["theatre_id"].nunique()
    n_cities   = df["city"].nunique()
    print(f"  Unique theatres : {n_theatres}")
    print(f"  Unique cities   : {n_cities}")

    # ── Descriptive statistics ────────────────────────────────────────────────
    section("9. Descriptive Statistics (numeric columns)")
    numeric_cols = ["audience_count", "online_bookings", "offline_bookings",
                    "occupancy_rate", "capacity"]
    print(df[numeric_cols].describe().to_string())

    section("10. Theatre-Level Summary")
    theatre_summary = (
        df.groupby(["theatre_id", "theatre_name", "theatre_type"])
        .agg(
            days=("date", "count"),
            avg_audience=("audience_count", "mean"),
            max_audience=("audience_count", "max"),
            avg_occupancy=("occupancy_rate", "mean"),
        )
        .reset_index()
        .round(2)
    )
    print(theatre_summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("Validation complete.")
    print("=" * 60)


if __name__ == "__main__":
    validate()
