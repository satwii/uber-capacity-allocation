"""Sensitivity analysis: re-solve a model over a range of one input."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from .data import PlanningData
from .models import solve_gp, solve_ilp

KPI_COLUMNS = ["served", "unmet", "service_rate", "rate_C", "rate_A", "rate_gap", "utilisation"]


def _run(data: PlanningData, model: str, **params):
    if model == "ilp":
        return solve_ilp(data, **params)
    if model == "gp":
        return solve_gp(data, **params)
    raise ValueError(f"Unknown model: {model}")


def _record(label: str, value: float, result) -> dict:
    row = {label: value, "status": result.status}
    row.update({k: result.kpis.get(k) for k in KPI_COLUMNS})
    if "shortfall" in result.kpis:
        row["shortfall"] = result.kpis["shortfall"]
    return row


def capacity_sweep(data: PlanningData, model: str, alphas: list[float], **params) -> pd.DataFrame:
    """Vary the driver-capacity adjustment alpha (e.g. -0.2 = 20% fewer drivers)."""
    params.pop("alpha", None)
    return pd.DataFrame([_record("alpha", a, _run(data, model, alpha=a, **params)) for a in alphas])


def productivity_sweep(data: PlanningData, model: str, qs: list[float], **params) -> pd.DataFrame:
    """Vary rides per driver-hour q."""
    return pd.DataFrame([_record("q", q, _run(replace(data, q=q), model, **params)) for q in qs])


def target_sweep(data: PlanningData, targets: list[float], **gp_params) -> pd.DataFrame:
    """Vary the GP service-level target, applied to both City and Airport."""
    gp_params.pop("g_C", None)
    gp_params.pop("g_A", None)
    return pd.DataFrame([_record("target", g, solve_gp(data, g_C=g, g_A=g, **gp_params)) for g in targets])
