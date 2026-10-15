# Uber Service Capacity Allocation — Design Spec

Date: 2026-10-15
Course: Operations Research project (Amrita, AIE23)
Dataset: `Uber Request Data.csv` (Kaggle, 6,745 requests, 11–15 July 2016)

## 1. Goal

Build a Streamlit app that loads Uber request data (CSV upload **or** manual value-by-value
entry), generates statistics on demand, and runs two user-selectable optimisation models
(Integer Programming and Goal Programming), plus sensitivity analysis and a comparison
against historical outcomes.

Course requirements covered: data loading in a UI, on-demand statistics, at least two
user-selectable optimisations, sensitivity analysis.

## 2. Decisions taken (from the review of the first-review PPT)

| # | Issue in PPT | Decision |
|---|---|---|
| 1 | "Hour t" ambiguous | Planning period = **date × hour** (120 periods for the full dataset). |
| 2 | ILP as written is trivial (any split is optimal, LP relaxation is integral, redundant constraints) | Decision is **integer drivers per location** per hour; served requests limited by `q · drivers`. Location weights make the split meaningful. Redundant `X ≤ D` dropped (implied). |
| 3 | GP has no explicit balance goal | Add a **balance goal**: deviation between City and Airport service rates. |
| 4 | Historical comparison is biased (capacity counts cancelling drivers) | Kept, but stated openly as an assumption in the app and report. |
| 5 | Sensitivity only ±20% runs | Add capacity, productivity and target sweeps, plus **shadow prices** from the LP relaxation of Model 1. |

## 3. Data layer

### 3.1 Input mode A — CSV upload
- `st.file_uploader` for a CSV with the Uber schema
  (`Request id, Pickup point, Driver id, Status, Request timestamp, Drop timestamp`).
- A "Use bundled sample dataset" button loads the repo CSV for demos.
- Schema check: missing columns → clear error message, nothing else runs.

Preprocessing:
- Drop duplicate `Request id`s.
- Parse both timestamp formats (`11/7/2016 11:51` and `13-07-2016 08:33:16`) with day-first parsing.
- Keep null `Driver id` (they are the "No Cars Available" requests).
- Derive `date`, `hour`, `period = date + hour`, trip duration (completed trips only).

Derived planning table (one row per period):
- `D_C`, `D_A` = requests at City / Airport in the period.
- `N` = unique non-null drivers on Completed or Cancelled requests in the period.
- `hist_C`, `hist_A` = completed trips per location (for comparison).
- `q = 60 / median completed-trip duration (minutes)` → ≈ 1.15 on this dataset.

### 3.2 Input mode B — manual value-by-value entry
- `st.data_editor` table with columns `Period, City demand, Airport demand, Active drivers`,
  rows can be added or deleted, and every cell can be edited.
- Separate numeric input for `q` (default 1.15).
- Starts with a few example rows; there is also a button to pre-fill it from the uploaded dataset so users can tweak values.
- Validation: non-negative integers; invalid rows are flagged.
- Historical columns are absent in this mode, so the comparison tab shows models only.

Both modes produce the same `PlanningData` object (period table + `q`), so all downstream
code is mode-agnostic.

## 4. Statistics (on demand)

Available only for CSV mode (manual mode shows a summary of the entered table):
- Dataset overview: rows, columns, missing values, date range.
- Status breakdown overall and by pickup point.
- Hourly demand by pickup point (aggregated over the days).
- Supply–demand gap heatmap (hour × pickup point, share of unmet requests).
- Trip-duration distribution, median and `q`.
- Active drivers per period vs demand.

Each block is shown when the user selects it (checkbox / selectbox), matching "statistics on demand".

## 5. Models

Notation: `l ∈ {C, A}`, `t` = period, `D_{l,t}` demand, `N_t` drivers,
`q` rides per driver-hour, `α` capacity adjustment (e.g. −0.2), `N'_t = ⌊N_t (1+α)⌋`.

### 5.1 Model 1 — Integer Linear Program (driver allocation)

Variables (integer ≥ 0): `Y_{l,t}` drivers assigned, `X_{l,t}` requests served;
`U_{l,t} ≥ 0` unmet.

```
min  Σ_t Σ_l w_l · U_{l,t}
s.t. Y_{C,t} + Y_{A,t} ≤ N'_t            (driver capacity)
     X_{l,t} ≤ q · Y_{l,t}               (productivity)
     X_{l,t} + U_{l,t} = D_{l,t}         (demand balance)
     Y, X ∈ Z≥0,  U ≥ 0
```
User inputs: weights `w_C`, `w_A` (default 1, 1), capacity adjustment `α`.

### 5.2 Model 2 — Goal Programming

Same constraints as Model 1, plus goal constraints:
```
X_{C,t} + d⁻_{C,t} − d⁺_{C,t} = g_C · D_{C,t}                 (City target)
X_{A,t} + d⁻_{A,t} − d⁺_{A,t} = g_A · D_{A,t}                 (Airport target)
X_{C,t}/D_{C,t} − X_{A,t}/D_{A,t} + b⁻_t − b⁺_t = 0           (balance; only if both D > 0)
```
Goals (normalised so units are comparable):
- G1 unmet demand: `Σ U / Σ D`
- G2 target shortfall: `Σ (d⁻_C + d⁻_A) / Σ (g·D)`
- G3 imbalance: `Σ (b⁺ + b⁻) / |B|`, where `B` = periods with demand at both locations (balance is undefined otherwise)

Two solve modes (user-selectable):
- **Weighted:** `min P1·G1 + P2·G2 + P3·G3`, user sets `P1, P2, P3` (default 10, 3, 1).
- **Pre-emptive (lexicographic):** solve G1, fix it as a constraint (with a small tolerance), then G2, then G3.

User inputs: `g_C`, `g_A` (default 0.70), priorities, solve mode, `α`.

### 5.3 Outputs (both models)
Per-period table: drivers per location, served, unmet, service rate per location,
capacity utilisation (`served / (q·N')`), rate gap; plus totals/KPIs and charts.
Model 2 also shows deviations from the targets.

Solver: PuLP with bundled CBC. Infeasible / failed solve → status shown, no crash.

## 6. Sensitivity analysis tab

- **Capacity sweep:** user-editable list of α values (default −20%, −10%, 0, +10%, +20%) →
  run chosen model → table + line chart of unmet demand and service rates.
- **Productivity sweep:** range of `q` values → same outputs.
- **Target sweep (GP):** `g` from 0.5 to 0.9 → shortfall, unmet demand, rate gap.
- **Shadow prices:** solve the LP relaxation of Model 1, report the dual of the driver-capacity
  constraint per period (value of one extra driver in that hour) and highlight the top periods.

## 7. Comparison tab

Historical (completed trips) vs Model 1 vs Model 2: served, unmet, service rate per
location, rate gap, utilisation. It states the assumption that the capacity estimate counts
drivers on cancelled trips, so models will exceed historical completions.

## 8. Code structure

```
project/
  app.py                 Streamlit UI (tabs: Data, Statistics, Optimisation, Sensitivity, Comparison)
  uberopt/
    data.py              load/validate CSV, preprocess, build PlanningData, manual-table parsing
    stats.py             statistics tables + Plotly figures
    models.py            solve_ilp, solve_gp, solve_lp_relaxation (shadow prices), result dataclass
    sensitivity.py       sweeps built on models.py
  tests/                 pytest: preprocessing, small hand-checkable model cases
  requirements.txt       pandas, numpy, pulp, streamlit, plotly, pytest
```

## 9. Testing

- Preprocessing: both timestamp formats parse; counts match (6,745 rows, 2,831 completed);
  period table sums to total demand.
- Model 1: a two-period toy case with a known optimum; weights shift drivers as expected;
  capacity never exceeded.
- Model 2: targets met when capacity is ample; the balance gap shrinks as P3 rises.
- Shadow prices: zero in periods with spare capacity, positive in bottleneck periods.
- Manual mode: the edited table produces the same `PlanningData` shape.

## 10. Out of scope (for now)

Report (IEEE format) and final PPT — handled after the project is built.
