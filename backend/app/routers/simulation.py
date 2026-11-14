"""
BoxOfficeIQ - What-If Scenario Simulation Router
Enables cinema operators to simulate operational shock scenarios:
Holiday surges, marketing campaigns, demand elasticities, and operational resource requirements.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from app.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/simulation", tags=["simulation"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
FORECASTS_CSV = BASE_DIR / "ml" / "data" / "processed" / "forecasts_7day.csv"
ATTENDANCE_CSV = BASE_DIR / "ml" / "data" / "processed" / "attendance_clean.csv"


class SimulationRequest(BaseModel):
    theatre_id: int = Field(1, description="Cinema ID to simulate")
    base_date: Optional[str] = Field(None, description="Start date YYYY-MM-DD")
    holiday_factor: float = Field(1.0, ge=0.5, le=3.0, description="Holiday factor multiplier (0.5 - 3.0)")
    demand_change_pct: float = Field(0.0, ge=-50.0, le=100.0, description="Overall demand shock % (-50 to +100)")
    days_to_forecast: int = Field(7, ge=1, le=14, description="Simulation duration in days")


@router.post("/run")
def run_scenario_simulation(req: SimulationRequest, db=Depends(get_db)) -> Dict[str, Any]:
    """Runs counterfactual what-if demand simulation and derives operational recommendations."""
    days_out: List[Dict[str, Any]] = []
    
    # 1. Obtain baseline predictions
    baseline_series: List[float] = []
    dates: List[str] = []

    if FORECASTS_CSV.exists():
        try:
            df = pd.read_csv(FORECASTS_CSV)
            sub = df[df["theatre_id"] == req.theatre_id].head(req.days_to_forecast)
            for _, r in sub.iterrows():
                baseline_series.append(float(r["predicted_audience"]))
                dates.append(str(r["date"]))
        except Exception as e:
            logger.warning(f"Failed reading forecasts_7day.csv for simulation: {e}")

    # Fallback if baseline_series is short
    if len(baseline_series) < req.days_to_forecast:
        start_dt = pd.to_datetime(req.base_date) if req.base_date else datetime.now()
        for i in range(len(baseline_series), req.days_to_forecast):
            cur_dt = start_dt + timedelta(days=i)
            dates.append(cur_dt.strftime("%Y-%m-%d"))
            dow = cur_dt.weekday()
            base_val = 280.0 if dow in (4, 5, 6) else 190.0
            baseline_series.append(base_val)

    # 2. Apply scenario multipliers
    # holiday_factor influences weekends and designated dates more strongly
    demand_mult = 1.0 + (req.demand_change_pct / 100.0)
    simulated_series: List[float] = []

    for d_str, b_val in zip(dates, baseline_series):
        dt = pd.to_datetime(d_str)
        dow = dt.weekday()
        # Holiday factor applied with higher weighting on Fri/Sat/Sun
        h_weight = (req.holiday_factor - 1.0) * (1.2 if dow in (4, 5, 6) else 0.8) + 1.0
        s_val = round(max(10.0, b_val * demand_mult * h_weight), 1)
        simulated_series.append(s_val)
        days_out.append({
            "date": d_str,
            "day_name": dt.strftime("%A"),
            "baseline": round(b_val, 1),
            "simulated": s_val,
        })

    tot_base = round(sum(baseline_series), 1)
    tot_sim = round(sum(simulated_series), 1)
    change_pct = round(((tot_sim - tot_base) / (tot_base + 1e-5)) * 100, 1)

    # 3. Deriving concrete operational decision support
    avg_sim = tot_sim / len(simulated_series)
    staff_count = max(4, int(np.ceil(avg_sim / 35)))
    extra_staff = max(0, int(np.ceil((tot_sim - tot_base) / 100)))

    if change_pct > 20:
        staffing_rec = (
            f"High Surge Scenario (+{change_pct}%): Recommend deploying +{extra_staff} extra floor/usher staff "
            f"and operating 3 active concession registers during peak showtimes."
        )
        inv_rec = (
            f"Increase concession raw inventory (popcorn seed, oil, cups, syrup) by +{int(change_pct * 1.15)}% "
            f"to prevent stockouts during peak afternoon and evening slots."
        )
    elif change_pct < -15:
        staffing_rec = (
            f"Low Demand Scenario ({change_pct}%): Reduce concession lanes to 1 active register. "
            f"Schedule core crew of {staff_count} members to optimize labor overhead."
        )
        inv_rec = (
            f"Reduce perishable fresh preparation by {abs(int(change_pct))}% to minimize food wastage."
        )
    else:
        staffing_rec = (
            f"Balanced Operations ({change_pct:+.1f}%): Maintain standard shift roster of {staff_count} crew members "
            f"with regular rotation between ticket checks and concessions."
        )
        inv_rec = (
            f"Maintain standard par levels with routine weekly restock cycles."
        )

    return {
        "theatre_id": req.theatre_id,
        "baseline_audience": tot_base,
        "simulated_audience": tot_sim,
        "change_pct": change_pct,
        "staffing_recommendation": staffing_rec,
        "inventory_recommendation": inv_rec,
        "days": days_out,
    }
