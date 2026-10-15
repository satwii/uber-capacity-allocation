# Uber Capacity Allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Streamlit app that loads Uber request data (CSV upload or manual entry), shows statistics on demand, runs an ILP and a Goal Programming model for City/Airport driver allocation, and provides sensitivity analysis and historical comparison.

**Architecture:** A small `uberopt` package with pure functions: `data` (input → `PlanningData`), `stats` (tables/figures), `models` (PuLP models → `ModelResult`), `sensitivity` (sweeps over `models`). `app.py` is a thin Streamlit layer over the package. All model code consumes only `PlanningData`, so both input modes behave identically.

**Tech Stack:** Python 3.12, pandas 2.2, numpy 1.26, PuLP (CBC), Streamlit, Plotly 6, pytest.

**Spec:** `docs/superpowers/specs/2026-10-15-uber-capacity-allocation-design.md`

## Global Constraints

- Planning period = date × hour (120 periods on the full dataset).
- `q = 60 / median completed-trip duration (minutes)`; manual-mode default `q = 1.15`.
- Capacity adjustment: `N'_t = floor(N_t * (1 + alpha))`.
- GP defaults: `g_C = g_A = 0.70`, `P1, P2, P3 = 10, 3, 1`; modes `"weighted"` and `"preemptive"`.
- Sensitivity default capacity list: −20%, −10%, 0, +10%, +20%; GP target sweep 0.5 → 0.9.
- Solver: PuLP default CBC, `msg=False`. A failed or infeasible solve returns a status; it never raises into the UI.
- Not a git repo: "Commit" steps are skipped unless the user runs `git init`.

## Review Focus

1. Periods where one location has zero demand → the balance constraint must be skipped (no division by zero); test in Task 3.
2. Periods with `N_t = 0` (all requests "No Cars Available") → model solves with zero served, no crash; test in Task 3.
3. An uploaded CSV with wrong or missing columns → a readable `ValueError` that the UI shows; test in Task 1.
4. A manual table with blank/negative/non-numeric cells → those rows are rejected with a message, and valid rows are still used; test in Task 1.
5. Very large α (−100%) → zero capacity, everything unmet, no crash; test in Task 4.

---

### Task 1: Data layer (CSV + manual entry)

**Files:**
- Create: `requirements.txt`, `uberopt/__init__.py`, `uberopt/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces:
  - `REQUIRED_COLUMNS: list[str]`
  - `load_requests(source) -> pd.DataFrame` — source is a path or file-like. Validates columns (raises `ValueError` listing the missing ones), drops duplicate `Request id`, parses both timestamp formats (`dayfirst=True, format="mixed"`), and adds `date`, `hour`, `period` (`pd.Timestamp` floored to the hour) and `duration_min` (completed trips only, else NaN).
  - `estimate_q(requests) -> float`
  - `@dataclass PlanningData: periods: pd.DataFrame; q: float; source: str` — `periods` columns: `period` (str label), `D_C`, `D_A`, `N` (int), and optional `hist_C`, `hist_A` (int); `source` ∈ `{"csv","manual"}`.
  - `build_planning_data(requests) -> PlanningData`
  - `MANUAL_COLUMNS = ["Period","City demand","Airport demand","Active drivers"]`
  - `example_manual_table() -> pd.DataFrame` (4 example rows)
  - `planning_to_manual_table(data: PlanningData) -> pd.DataFrame`
  - `parse_manual_table(df, q) -> tuple[PlanningData, list[str]]` — returns data built from valid rows plus a list of error messages for rejected rows.

- [ ] **Step 1:** `pip install pulp streamlit pytest` and write `requirements.txt` (`pandas>=2.2`, `numpy>=1.26`, `pulp>=2.8`, `streamlit>=1.38`, `plotly>=5.20`, `pytest>=8`).
- [ ] **Step 2: Write failing tests** in `tests/test_data.py`:
  - `test_load_full_dataset`: `len == 6745`; `(Status=="Trip Completed").sum()==2831`; no NaT in `Request timestamp`.
  - `test_estimate_q`: `abs(q - 60/52.0833) < 0.01`.
  - `test_planning_totals`: `periods.D_C.sum()==3507`, `D_A.sum()==3238`, `hist_C.sum()+hist_A.sum()==2831`, `len(periods)<=120`.
  - `test_missing_columns_raises`: a CSV without `Status` → `ValueError` whose message contains `"Status"`.
  - `test_manual_parse_rejects_bad_rows`: rows `[("P1",10,5,4),("P2",-1,5,4),("P3","x",5,4),("P4",None,5,4)]` → 1 valid period and 3 errors; `data.source=="manual"`; there are no hist columns.
  - `test_manual_roundtrip`: `parse_manual_table(planning_to_manual_table(data), data.q)` reproduces `D_C, D_A, N`.
- [ ] **Step 3:** Run `pytest tests/test_data.py -v` → FAIL (import error).
- [ ] **Step 4:** Implement `uberopt/data.py`. N = `nunique` of non-null `Driver id` over Status ∈ {Completed, Cancelled} per period. Include periods that have demand but no drivers, with `N=0`.
- [ ] **Step 5:** Run `pytest tests/test_data.py -v` → PASS.

### Task 2: Statistics

**Files:**
- Create: `uberopt/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Consumes: `load_requests`, `PlanningData`.
- Produces (each takes the `requests` DataFrame; figures are `plotly.graph_objects.Figure`):
  - `overview(requests) -> dict` (keys `rows, columns, missing (dict), start, end, pickup_counts (dict), status_counts (dict)`)
  - `status_by_pickup(requests) -> pd.DataFrame` (index pickup, columns statuses)
  - `fig_status_by_pickup`, `fig_hourly_demand`, `fig_gap_heatmap` (hour × pickup, share unmet = not completed), `fig_duration_hist`
  - `fig_drivers_vs_demand(data: PlanningData)` (works for both modes)

- [ ] **Step 1: Failing tests:** `overview(...)["rows"]==6745`; `status_by_pickup(...).loc["City"].sum()==3507`; each `fig_*` returns a Figure with ≥1 trace.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.

### Task 3: Models (ILP, GP, LP shadow prices)

**Files:**
- Create: `uberopt/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `PlanningData`.
- Produces:
  - `adjusted_drivers(N: pd.Series, alpha: float) -> pd.Series` (floor, clipped at 0)
  - `@dataclass ModelResult: status: str; objective: float|None; table: pd.DataFrame; kpis: dict`
    - `table` columns: `period, D_C, D_A, N_adj, Y_C, Y_A, X_C, X_A, U_C, U_A, rate_C, rate_A, rate_gap, utilisation`, plus for GP `dminus_C, dplus_C, dminus_A, dplus_A`.
    - `kpis` keys: `served, unmet, demand, service_rate, rate_C, rate_A, rate_gap (abs, overall), utilisation`.
    - Rates where D=0 are NaN; utilisation = served / (q·N_adj), NaN when capacity is 0.
  - `solve_ilp(data, w_C=1.0, w_A=1.0, alpha=0.0) -> ModelResult`
  - `solve_gp(data, g_C=0.7, g_A=0.7, P=(10,3,1), mode="weighted", alpha=0.0) -> ModelResult`
  - `shadow_prices(data, w_C=1.0, w_A=1.0, alpha=0.0) -> pd.DataFrame` (columns `period, N_adj, shadow_price`, sorted desc)
  - `historical_result(data) -> ModelResult | None` (None in manual mode; X = hist, Y/N columns NaN)
- Formulation exactly as in spec §5.1/§5.2. Preemptive mode solves G1 → adds `G1 <= opt1 + 1e-6`, then G2 → adds `G2 <= opt2 + 1e-6`, then G3. Balance constraint only when `D_C>0 and D_A>0`. Shadow prices: the same Model 1 with continuous variables; the dual of the driver constraint (`.pi`) has its sign flipped so that a positive value = unmet reduced per extra driver.

- [ ] **Step 1: Failing tests** (hand-checkable toy `PlanningData`, q=1):
  - `test_ilp_toy`: periods `[(D_C=5,D_A=5,N=6),(D_C=2,D_A=0,N=4)]` → total unmet = 4, `X_C+X_A<=N` every row, all Y/X integer.
  - `test_ilp_weights_shift`: same period 1 with `w_A=2` → `X_A==5, X_C==1`.
  - `test_ilp_integer_drivers`: q=1.5, `D_C=3, D_A=3, N=3` → `Y_C+Y_A<=3` and `X_l <= floor(1.5*Y_l)`, unmet=2 (LP would give 1.5).
  - `test_zero_drivers`: N=0 → status "Optimal", served=0.
  - `test_gp_meets_targets_when_ample`: N large → all `dminus==0`.
  - `test_gp_balance_reduces_gap`: scarce case `D_C=10, D_A=2, N=6`, `w` equal; the gap with `P=(10,3,5)` is ≤ the gap with `P=(10,3,0)`.
  - `test_gp_zero_demand_location`: `D_A=0` → no error, `rate_A` NaN.
  - `test_gp_preemptive_matches_ilp_unmet`: preemptive unmet == ILP unmet on the toy.
  - `test_shadow_prices`: a period with spare capacity → 0; a bottleneck period → >0.
  - `test_real_data_solves`: full dataset ILP and GP both "Optimal" and served ≥ hist served.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.

### Task 4: Sensitivity sweeps

**Files:**
- Create: `uberopt/sensitivity.py`
- Test: `tests/test_sensitivity.py`

**Interfaces:**
- Consumes: `solve_ilp`, `solve_gp`, `PlanningData`.
- Produces (each returns a DataFrame with one row per value, plus the KPI columns from `ModelResult.kpis` and `status`):
  - `capacity_sweep(data, model: str, alphas: list[float], **params)` — `model` ∈ `{"ilp","gp"}`; column `alpha`
  - `productivity_sweep(data, model, qs: list[float], **params)` — column `q`
  - `target_sweep(data, targets: list[float], **gp_params)` — sets `g_C=g_A=g`; column `target`, plus `shortfall`.

- [ ] **Step 1: Failing tests:** unmet is non-increasing as alpha increases over `[-0.2,-0.1,0,0.1,0.2]`; `alpha=-1.0` → served 0, status Optimal; unmet is non-increasing in q; target sweep has 5 rows for `[0.5,0.6,0.7,0.8,0.9]`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.

### Task 5: Streamlit app

**Files:**
- Create: `app.py`, `README.md` (how to install and run)

**Interfaces:**
- Consumes: everything above. Session state key `"pdata"` holds the current `PlanningData`, and `"requests"` holds the raw DataFrame (CSV mode only).

Layout:
- Sidebar: input-mode radio ("Upload CSV" / "Enter values manually"), capacity adjustment slider (−50…+50%, default 0).
- Tabs:
  1. **Data** — CSV mode: uploader + "Use bundled sample dataset" button, preview table, period table. Manual mode: `st.data_editor(num_rows="dynamic")` seeded from `example_manual_table()`, a `q` number input, a "Pre-fill from uploaded dataset" button (enabled when requests are loaded), and errors from `parse_manual_table` shown via `st.warning`.
  2. **Statistics** — multiselect of the statistic blocks from spec §4; CSV-only blocks are hidden in manual mode.
  3. **Optimisation** — model selectbox (Model 1 ILP / Model 2 GP); model-specific inputs (weights; or targets, P1–P3, mode); a "Solve" button; KPI metrics, a stacked served/unmet chart by period, a rate chart, the result table, and a CSV download.
  4. **Sensitivity** — sub-sections for capacity sweep (editable alpha list), productivity sweep, target sweep (GP), and the shadow-price table + bar chart.
  5. **Comparison** — KPI table + grouped bar for Historical / ILP / GP, with an assumption note (spec §7).
- Every solve is wrapped so a non-"Optimal" status shows `st.error` instead of crashing.

- [ ] **Step 1:** Implement `app.py`.
- [ ] **Step 2:** Smoke test: `python -c "import ast;ast.parse(open('app.py').read())"`, then `streamlit run app.py --server.headless true` started in the background; `curl localhost:8501` returns 200 and the log shows no traceback.
- [ ] **Step 3:** Run `pytest -v` → all pass.
