"""Data loading, preprocessing and construction of the planning table.

Two input modes produce the same PlanningData object:
  * CSV upload of the Kaggle "Uber Request Data" file
  * manual value-by-value entry of per-period demand and drivers
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "Request id",
    "Pickup point",
    "Driver id",
    "Status",
    "Request timestamp",
    "Drop timestamp",
]
COMPLETED = "Trip Completed"
CANCELLED = "Cancelled"
NO_CARS = "No Cars Available"

MANUAL_COLUMNS = ["Period", "City demand", "Airport demand", "Active drivers"]
DEFAULT_Q = 1.15


@dataclass
class PlanningData:
    """Per-period inputs for the optimisation models.

    periods columns: period (str), D_C, D_A, N (int) and, in CSV mode,
    hist_C, hist_A (completed trips per location).
    """

    periods: pd.DataFrame
    q: float
    source: str  # "csv" or "manual"


def load_requests(source) -> pd.DataFrame:
    """Read and clean the raw request file (path or file-like object)."""
    df = pd.read_csv(source)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded file is missing required column(s): {', '.join(missing)}")

    df = df.drop_duplicates(subset="Request id").copy()
    df["Pickup point"] = df["Pickup point"].astype(str).str.strip()
    df["Status"] = df["Status"].astype(str).str.strip()
    # The file mixes "11/7/2016 11:51" and "13-07-2016 08:33:16"; both are day-first.
    for col in ["Request timestamp", "Drop timestamp"]:
        df[col] = pd.to_datetime(df[col], dayfirst=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["Request timestamp"])
    if df.empty:
        raise ValueError("The file has no rows with a valid Request timestamp")

    df["date"] = df["Request timestamp"].dt.date
    df["hour"] = df["Request timestamp"].dt.hour
    df["period"] = df["Request timestamp"].dt.floor("h")
    duration = (df["Drop timestamp"] - df["Request timestamp"]).dt.total_seconds() / 60
    df["duration_min"] = duration.where(df["Status"] == COMPLETED)
    return df.reset_index(drop=True)


def estimate_q(requests: pd.DataFrame) -> float:
    """Rides per driver per hour = 60 / median completed-trip duration (minutes)."""
    median = requests["duration_min"].median()
    if pd.isna(median) or median <= 0:
        return DEFAULT_Q
    return float(60 / median)


def build_planning_data(requests: pd.DataFrame) -> PlanningData:
    """Aggregate requests into date x hour planning periods."""
    demand = (
        requests.pivot_table(index="period", columns="Pickup point", values="Request id",
                             aggfunc="count", fill_value=0)
        .reindex(columns=["City", "Airport"], fill_value=0)
    )
    active = requests[requests["Status"].isin([COMPLETED, CANCELLED])]
    drivers = active.groupby("period")["Driver id"].nunique()
    completed = (
        requests[requests["Status"] == COMPLETED]
        .pivot_table(index="period", columns="Pickup point", values="Request id",
                     aggfunc="count", fill_value=0)
        .reindex(index=demand.index, columns=["City", "Airport"], fill_value=0)
    )

    periods = pd.DataFrame({
        "period": [ts.strftime("%Y-%m-%d %H:00") for ts in demand.index],
        "D_C": demand["City"].astype(int).values,
        "D_A": demand["Airport"].astype(int).values,
        "N": drivers.reindex(demand.index, fill_value=0).astype(int).values,
        "hist_C": completed["City"].fillna(0).astype(int).values,
        "hist_A": completed["Airport"].fillna(0).astype(int).values,
    })
    return PlanningData(periods=periods, q=estimate_q(requests), source="csv")


def example_manual_table() -> pd.DataFrame:
    """Starter rows for manual entry."""
    return pd.DataFrame(
        [("05:00", 40, 12, 20), ("09:00", 60, 15, 25), ("18:00", 15, 70, 30), ("21:00", 20, 55, 28)],
        columns=MANUAL_COLUMNS,
    )


def planning_to_manual_table(data: PlanningData) -> pd.DataFrame:
    p = data.periods
    return pd.DataFrame({
        "Period": p["period"].values,
        "City demand": p["D_C"].values,
        "Airport demand": p["D_A"].values,
        "Active drivers": p["N"].values,
    })


def parse_manual_table(df: pd.DataFrame, q: float) -> tuple[PlanningData, list[str]]:
    """Validate a user-edited table. Bad rows are dropped and reported."""
    rows, errors = [], []
    numeric = {"City demand": "D_C", "Airport demand": "D_A", "Active drivers": "N"}
    for i, row in df.reset_index(drop=True).iterrows():
        label = row.get("Period")
        label = f"Row {i + 1}" if label is None or (isinstance(label, float) and np.isnan(label)) or str(label).strip() == "" else str(label).strip()
        values, problem = {}, None
        for col, key in numeric.items():
            v = pd.to_numeric(row.get(col), errors="coerce")
            if pd.isna(v):
                problem = f"{label}: '{col}' is blank or not a number"
                break
            if v < 0 or v != int(v):
                problem = f"{label}: '{col}' must be a non-negative whole number (got {row.get(col)})"
                break
            values[key] = int(v)
        if problem:
            errors.append(problem)
        else:
            rows.append({"period": label, **values})

    periods = pd.DataFrame(rows, columns=["period", "D_C", "D_A", "N"])
    for col in ["D_C", "D_A", "N"]:
        periods[col] = periods[col].astype(int)
    if not q or q <= 0:
        errors.append("Productivity q must be positive; using default 1.15")
        q = DEFAULT_Q
    return PlanningData(periods=periods, q=float(q), source="manual"), errors
