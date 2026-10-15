# Optimizing Uber Service Capacity Allocation (Integer & Goal Programming)

Operations Research course project. A Streamlit app that loads the Kaggle *Uber Request Data*
(or values typed in by hand), generates statistics on demand, and allocates drivers between
City and Airport pickups with two user-selectable models, plus sensitivity analysis.

## Setup

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the URL shown (usually http://localhost:8501).

## Using the app

| Where | What |
|---|---|
| Sidebar | Choose **Upload CSV** or **Enter values manually**; set the capacity adjustment (% change in drivers). |
| 📁 Data | Upload the CSV (or click *Use bundled sample dataset*), or edit the manual table cell by cell (add/delete rows, set q). |
| 📊 Statistics | Pick which statistics to generate: overview, trip outcomes, hourly demand, supply–demand gap heatmap, trip duration & q, demand vs capacity. |
| ⚙️ Optimisation | Choose **Model 1 – Integer Programming** or **Model 2 – Goal Programming**, set its inputs, press **Solve**. Download results as CSV. |
| 🔍 Sensitivity | Capacity sweep, productivity (q) sweep, service-target sweep, and shadow prices of driver capacity. |
| ⚖️ Comparison | Historical vs Model 1 vs Model 2. |

## Method

- **Periods:** date × hour (120 periods for 11–15 July 2016).
- **Demand** `D_l,t`: requests per location and period.
- **Drivers** `N_t`: unique drivers on completed or cancelled requests in the period.
- **Productivity** `q = 60 / median completed-trip duration` ≈ 60 / 52.1 = 1.152 rides per driver-hour.
- **Model 1 (ILP):** integer drivers `Y` per location, `X ≤ q·Y`, `Y_C + Y_A ≤ ⌊N(1+α)⌋`;
  minimise weighted unmet demand.
- **Model 2 (GP):** Model 1 constraints plus service-target goals (`g_C`, `g_A`) and a City–Airport
  balance goal; weighted (P1, P2, P3) or pre-emptive (G1 → G2 → G3).
- **Shadow prices:** duals of the driver-capacity constraints in the LP relaxation of Model 1.

Each period is solved as its own small model. Periods share no constraints and the objectives
are sums over periods, so this gives the same optimum as one joint model.

## Results on the full dataset (default settings)

| Scenario | Served | Unmet | Service rate | City | Airport | Mean City–Airport gap per period |
|---|---|---|---|---|---|---|
| Historical (completed trips) | 2,831 | 3,914 | 42.0% | 42.9% | 41.0% | 36.7% |
| Model 1 – ILP | 4,506 | 2,239 | 66.8% | 68.7% | 64.7% | 47.8% |
| Model 2 – GP (weighted 10/3/1) | 4,482 | 2,263 | 66.4% | 74.4% | 57.8% | 6.5% |
| Model 2 – GP (pre-emptive) | 4,506 | 2,239 | 66.8% | 75.4% | 57.4% | 15.3% |

- Capacity −20% / baseline / +20% → ILP unmet 3,193 / 2,239 / 1,536.
- The shadow price is 1.152 (= q) in almost every period: capacity binds nearly everywhere.
- Model 1 has many equally optimal City/Airport splits, so its per-period split is arbitrary
  (and can differ between solvers). Model 2 resolves this by choosing balanced splits.
- **Assumption:** capacity counts drivers seen on cancelled trips, so the models exceed the historical
  completions. This shows the potential of better allocation, not a guaranteed improvement.

## Project layout

```
app.py                 Streamlit UI
uberopt/data.py        loading, cleaning, planning table, manual-entry parsing
uberopt/stats.py       statistics and charts
uberopt/models.py      ILP, GP, shadow prices, historical baseline
uberopt/sensitivity.py capacity / productivity / target sweeps
tests/                 pytest suite (python -m pytest)
```
