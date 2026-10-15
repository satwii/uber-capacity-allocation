"""Optimisation models for allocating drivers between City and Airport.

Model 1 (ILP):  min  sum_t sum_l w_l U_{l,t}
                s.t. Y_C,t + Y_A,t <= N'_t           driver capacity
                     X_l,t <= q * Y_l,t              productivity
                     X_l,t + U_l,t = D_l,t           demand balance
                     Y, X integer >= 0, U >= 0

Model 2 (GP):   Model 1 constraints plus
                     X_C,t + d-_C,t - d+_C,t = g_C D_C,t           City target
                     X_A,t + d-_A,t - d+_A,t = g_A D_A,t           Airport target
                     X_C,t/D_C,t - X_A,t/D_A,t + b-_t - b+_t = 0   balance (both D > 0)
                Goals (normalised): G1 unmet share, G2 target shortfall share, G3 mean imbalance.

Periods share no constraints and every objective is a sum over periods, so each period is
solved as its own small model. This gives the same optimum as one joint model (also goal by
goal in pre-emptive GP) while keeping every integer programme tiny and fast.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pulp

from .data import PlanningData

LOCS = ("C", "A")
TOL = 1e-6


@dataclass
class ModelResult:
    status: str
    objective: float | None
    table: pd.DataFrame
    kpis: dict = field(default_factory=dict)


def adjusted_drivers(N: pd.Series, alpha: float) -> pd.Series:
    """N'_t = floor(N_t * (1 + alpha)), never negative."""
    return np.floor(N.astype(float) * (1 + alpha) + 1e-9).clip(lower=0).astype(int)


def _solver():
    """HiGHS runs in-process (fast for many small models); fall back to bundled CBC."""
    if pulp.HiGHS(msg=False).available():
        return pulp.HiGHS(msg=False, timeLimit=60)
    return pulp.PULP_CBC_CMD(msg=False, timeLimit=60)


def _solve(prob) -> str:
    prob.solve(_solver())
    return pulp.LpStatus[prob.status]


def _empty(reason: str) -> ModelResult:
    return ModelResult(status=reason, objective=None, table=pd.DataFrame(), kpis={})


def _val(x) -> float:
    return float(x.value() or 0.0)


def _rate(x, d):
    return x / d if d > 0 else np.nan


def _period_model(D: dict, n: int, q: float, t: int, relax: bool = False):
    """Variables and Model-1 constraints for one planning period."""
    cat = "Continuous" if relax else "Integer"
    prob = pulp.LpProblem(f"uber_allocation_{t}", pulp.LpMinimize)
    v = {}
    for l in LOCS:
        v["Y", l] = pulp.LpVariable(f"Y_{l}_{t}", lowBound=0, cat=cat)
        v["X", l] = pulp.LpVariable(f"X_{l}_{t}", lowBound=0, cat=cat)
        v["U", l] = pulp.LpVariable(f"U_{l}_{t}", lowBound=0)
        prob += v["X", l] <= q * v["Y", l], f"prod_{l}"
        prob += v["X", l] + v["U", l] == D[l], f"demand_{l}"
    prob += v["Y", "C"] + v["Y", "A"] <= n, "cap"
    return prob, v


def _periods(data: PlanningData, alpha: float):
    """Yield (index, demand per location, adjusted drivers) for every period."""
    p = data.periods.reset_index(drop=True)
    n_adj = adjusted_drivers(p["N"], alpha)
    for t in range(len(p)):
        yield t, {"C": int(p.at[t, "D_C"]), "A": int(p.at[t, "D_A"])}, int(n_adj[t])


def _row(p: pd.DataFrame, t: int, n: int, v: dict, gp: bool) -> dict:
    r = {"period": p["period"].iloc[t], "D_C": int(p["D_C"].iloc[t]), "D_A": int(p["D_A"].iloc[t]), "N_adj": n}
    for l in LOCS:
        r[f"Y_{l}"] = round(_val(v["Y", l]))
        r[f"X_{l}"] = round(_val(v["X", l]))
        r[f"U_{l}"] = r[f"D_{l}"] - r[f"X_{l}"]
        if gp:
            r[f"dminus_{l}"] = _val(v["dm", l])
            r[f"dplus_{l}"] = _val(v["dp", l])
    return r


def _finish_table(t: pd.DataFrame, q: float) -> pd.DataFrame:
    t["rate_C"] = [_rate(x, d) for x, d in zip(t["X_C"], t["D_C"])]
    t["rate_A"] = [_rate(x, d) for x, d in zip(t["X_A"], t["D_A"])]
    t["rate_gap"] = (t["rate_C"] - t["rate_A"]).abs()
    cap = q * t["N_adj"]
    t["utilisation"] = np.where(cap > 0, (t["X_C"] + t["X_A"]) / cap.where(cap > 0, 1), np.nan)
    return t


def compute_kpis(t: pd.DataFrame, q: float) -> dict:
    served_C, served_A = t["X_C"].sum(), t["X_A"].sum()
    dem_C, dem_A = t["D_C"].sum(), t["D_A"].sum()
    served, demand = served_C + served_A, dem_C + dem_A
    rate_C, rate_A = _rate(served_C, dem_C), _rate(served_A, dem_A)
    cap = q * t["N_adj"].sum()
    return {
        "served": int(served),
        "unmet": int(demand - served),
        "demand": int(demand),
        "service_rate": _rate(served, demand),
        "rate_C": rate_C,
        "rate_A": rate_A,
        "rate_gap": abs(rate_C - rate_A) if not (np.isnan(rate_C) or np.isnan(rate_A)) else np.nan,
        "mean_period_gap": float(t["rate_gap"].mean()) if t["rate_gap"].notna().any() else np.nan,
        "utilisation": served / cap if cap > 0 else np.nan,
    }


def solve_ilp(data: PlanningData, w_C: float = 1.0, w_A: float = 1.0, alpha: float = 0.0) -> ModelResult:
    """Model 1: minimise weighted unmet demand with integer driver allocation."""
    if data.periods.empty:
        return _empty("No data")
    w = {"C": w_C, "A": w_A}
    rows, objective = [], 0.0
    for t, D, n in _periods(data, alpha):
        prob, v = _period_model(D, n, data.q, t)
        prob += pulp.lpSum(w[l] * v["U", l] for l in LOCS)
        status = _solve(prob)
        if status != "Optimal":
            return _empty(f"{status} (period {data.periods['period'].iloc[t]})")
        objective += pulp.value(prob.objective) or 0.0
        rows.append(_row(data.periods, t, n, v, gp=False))
    table = _finish_table(pd.DataFrame(rows), data.q)
    return ModelResult("Optimal", objective, table, compute_kpis(table, data.q))


def solve_gp(data: PlanningData, g_C: float = 0.7, g_A: float = 0.7, P=(10, 3, 1),
             mode: str = "weighted", alpha: float = 0.0) -> ModelResult:
    """Model 2: goal programme over unmet demand, service targets and City/Airport balance."""
    if mode not in ("weighted", "preemptive"):
        raise ValueError(f"Unknown GP mode: {mode}")
    if data.periods.empty:
        return _empty("No data")
    p = data.periods
    g = {"C": g_C, "A": g_A}
    # Normalisers are global so the per-period goals add up to the joint goals.
    total_demand = max(int(p["D_C"].sum() + p["D_A"].sum()), 1)
    total_target = max(g_C * p["D_C"].sum() + g_A * p["D_A"].sum(), 1e-9)
    n_balance = max(int(((p["D_C"] > 0) & (p["D_A"] > 0)).sum()), 1)

    rows, goal_totals = [], [0.0, 0.0, 0.0]
    for t, D, n in _periods(data, alpha):
        prob, v = _period_model(D, n, data.q, t)
        for l in LOCS:
            v["dm", l] = pulp.LpVariable(f"dminus_{l}_{t}", lowBound=0)
            v["dp", l] = pulp.LpVariable(f"dplus_{l}_{t}", lowBound=0)
            prob += v["X", l] + v["dm", l] - v["dp", l] == g[l] * D[l], f"target_{l}"
        imbalance = pulp.LpAffineExpression()
        if D["C"] > 0 and D["A"] > 0:
            bm = pulp.LpVariable(f"bminus_{t}", lowBound=0)
            bp = pulp.LpVariable(f"bplus_{t}", lowBound=0)
            prob += v["X", "C"] * (1 / D["C"]) - v["X", "A"] * (1 / D["A"]) + bm - bp == 0, "balance"
            imbalance = bm + bp
        goals = [pulp.lpSum(v["U", l] for l in LOCS) * (1 / total_demand),
                 pulp.lpSum(v["dm", l] for l in LOCS) * (1 / total_target),
                 imbalance * (1 / n_balance)]

        if mode == "weighted":
            prob += pulp.lpSum(w * goal for w, goal in zip(P, goals))
            status = _solve(prob)
        else:
            for k, goal in enumerate(goals):
                prob.setObjective(goal)
                status = _solve(prob)
                if status != "Optimal":
                    break
                if k < len(goals) - 1:
                    prob += goal <= pulp.value(goal) + TOL, f"lock_goal_{k + 1}"
        if status != "Optimal":
            return _empty(f"{status} (period {p['period'].iloc[t]})")
        for k, goal in enumerate(goals):
            goal_totals[k] += float(pulp.value(goal) or 0.0)
        rows.append(_row(p, t, n, v, gp=True))

    table = _finish_table(pd.DataFrame(rows), data.q)
    kpis = compute_kpis(table, data.q)
    kpis["shortfall"] = float(table[["dminus_C", "dminus_A"]].sum().sum())
    kpis["goals"] = {f"G{k + 1}": val for k, val in enumerate(goal_totals)}
    objective = sum(w * val for w, val in zip(P, goal_totals)) if mode == "weighted" else None
    return ModelResult("Optimal", objective, table, kpis)


def shadow_prices(data: PlanningData, w_C: float = 1.0, w_A: float = 1.0, alpha: float = 0.0) -> pd.DataFrame:
    """Dual value of each period's driver-capacity constraint in the LP relaxation of Model 1.

    Positive value = reduction in weighted unmet demand from one extra driver in that period.
    """
    cols = ["period", "D_C", "D_A", "N_adj", "shadow_price"]
    if data.periods.empty:
        return pd.DataFrame(columns=cols)
    w = {"C": w_C, "A": w_A}
    # The LP relaxation is easy as one joint model; CBC is used because it reports duals.
    joint = pulp.LpProblem("uber_lp_relaxation", pulp.LpMinimize)
    objective, rows = [], []
    for t, D, n in _periods(data, alpha):
        prob, v = _period_model(D, n, data.q, t, relax=True)
        for name, constraint in prob.constraints.items():
            joint.addConstraint(constraint, f"{name}_{t}")
        objective += [w[l] * v["U", l] for l in LOCS]
        rows.append({"period": data.periods["period"].iloc[t], "D_C": D["C"], "D_A": D["A"], "N_adj": n})
    joint += pulp.lpSum(objective)
    joint.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[joint.status] != "Optimal":
        return pd.DataFrame(columns=cols)
    for t, r in enumerate(rows):
        r["shadow_price"] = -(joint.constraints[f"cap_{t}"].pi or 0.0) + 0.0
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("shadow_price", ascending=False, kind="stable").reset_index(drop=True)


def historical_result(data: PlanningData) -> ModelResult | None:
    """What actually happened (completed trips), in the same shape as a model result."""
    if "hist_C" not in data.periods.columns:
        return None
    p = data.periods
    t = pd.DataFrame({
        "period": p["period"], "D_C": p["D_C"], "D_A": p["D_A"], "N_adj": p["N"],
        "Y_C": np.nan, "Y_A": np.nan, "X_C": p["hist_C"], "X_A": p["hist_A"],
    })
    t["U_C"] = t["D_C"] - t["X_C"]
    t["U_A"] = t["D_A"] - t["X_A"]
    t = _finish_table(t, data.q)
    return ModelResult(status="Historical", objective=None, table=t, kpis=compute_kpis(t, data.q))
