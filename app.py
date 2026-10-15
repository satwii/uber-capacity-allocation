"""Streamlit UI: Optimizing Uber Service Capacity Allocation using Integer and Goal Programming.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from uberopt import stats
from uberopt.data import (
    DEFAULT_Q,
    PlanningData,
    build_planning_data,
    example_manual_table,
    load_requests,
    parse_manual_table,
    planning_to_manual_table,
)
from uberopt.models import ModelResult, historical_result, shadow_prices, solve_gp, solve_ilp
from uberopt.sensitivity import capacity_sweep, productivity_sweep, target_sweep

SAMPLE_CSV = Path(__file__).parent / "Uber Request Data.csv"
MODEL_1 = "Model 1 – Integer Programming"
MODEL_2 = "Model 2 – Goal Programming"
LOC = stats.LOCATION_COLORS
SER = stats.SERIES_COLORS

st.set_page_config(page_title="Uber Capacity Allocation", page_icon="🚕", layout="wide")


# ---------------------------------------------------------------- cached computation
@st.cache_data(show_spinner=False)
def load_csv_bytes(raw: bytes) -> pd.DataFrame:
    return load_requests(io.BytesIO(raw))


def _pd(periods: pd.DataFrame, q: float, source: str) -> PlanningData:
    return PlanningData(periods=periods, q=q, source=source)


@st.cache_data(show_spinner=False)
def run_ilp(periods, q, source, w_C, w_A, alpha) -> ModelResult:
    return solve_ilp(_pd(periods, q, source), w_C=w_C, w_A=w_A, alpha=alpha)


@st.cache_data(show_spinner=False)
def run_gp(periods, q, source, g_C, g_A, P, mode, alpha) -> ModelResult:
    return solve_gp(_pd(periods, q, source), g_C=g_C, g_A=g_A, P=P, mode=mode, alpha=alpha)


@st.cache_data(show_spinner=False)
def run_shadow(periods, q, source, w_C, w_A, alpha) -> pd.DataFrame:
    return shadow_prices(_pd(periods, q, source), w_C=w_C, w_A=w_A, alpha=alpha)


@st.cache_data(show_spinner=False)
def run_capacity_sweep(periods, q, source, model, alphas, params) -> pd.DataFrame:
    return capacity_sweep(_pd(periods, q, source), model, list(alphas), **dict(params))


@st.cache_data(show_spinner=False)
def run_productivity_sweep(periods, q, source, model, qs, params) -> pd.DataFrame:
    return productivity_sweep(_pd(periods, q, source), model, list(qs), **dict(params))


@st.cache_data(show_spinner=False)
def run_target_sweep(periods, q, source, targets, params) -> pd.DataFrame:
    return target_sweep(_pd(periods, q, source), list(targets), **dict(params))


# ---------------------------------------------------------------- helpers
def pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1%}"


def parse_number_list(text: str, scale: float = 1.0) -> list[float]:
    """'-20, -10, 0' -> [-0.2, -0.1, 0.0] with scale=0.01. Raises ValueError on bad input."""
    values = [float(v) * scale for v in text.replace(";", ",").split(",") if v.strip()]
    if not values:
        raise ValueError("enter at least one number")
    return values


def show_chart_and_table(fig: go.Figure, table: pd.DataFrame, key: str):
    st.plotly_chart(fig, width="stretch", key=f"chart_{key}")
    with st.expander("Show data table"):
        st.dataframe(table, width="stretch", hide_index=True)


# Model parameters live in plain session-state keys (not widget keys): Streamlit drops the
# state of widgets that are not rendered, which would reset one model's inputs while the
# other model is selected.
PARAM_DEFAULTS = {"w_C": 1.0, "w_A": 1.0, "g_C": 70, "g_A": 70, "P1": 10.0, "P2": 3.0, "P3": 1.0,
                  "gp_mode": "Weighted"}
GP_MODES = ["Weighted", "Pre-emptive (G1 → G2 → G3)"]
for _k, _v in PARAM_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def param_widget(widget, label: str, name: str, *args, **kwargs):
    """Render a widget seeded from, and written back to, the persistent key `name`."""
    st.session_state[name] = widget(label, *args, value=st.session_state[name], key=f"wgt_{name}", **kwargs)


def ilp_params() -> dict:
    s = st.session_state
    return {"w_C": float(s["w_C"]), "w_A": float(s["w_A"])}


def gp_params() -> dict:
    s = st.session_state
    return {
        "g_C": s["g_C"] / 100,
        "g_A": s["g_A"] / 100,
        "P": (float(s["P1"]), float(s["P2"]), float(s["P3"])),
        "mode": "preemptive" if s["gp_mode"].startswith("Pre") else "weighted",
    }


def data_fingerprint(data: PlanningData) -> str:
    h = hashlib.sha1(pd.util.hash_pandas_object(data.periods, index=False).values.tobytes())
    h.update(repr((round(data.q, 9), data.source)).encode())
    return h.hexdigest()


def remember(name: str, fingerprint, value):
    st.session_state[name] = (fingerprint, value)


def recall(name: str, fingerprint, rerun_hint: str):
    """Return a stored result only if it was computed from the current inputs."""
    entry = st.session_state.get(name)
    if entry is None:
        return None
    if entry[0] != fingerprint:
        st.info(f"Inputs changed since the last run – press **{rerun_hint}** again to update these results.")
        return None
    return entry[1]


def solve(model: str, data: PlanningData, alpha: float) -> ModelResult:
    args = (data.periods, data.q, data.source)
    if model == MODEL_1:
        p = ilp_params()
        return run_ilp(*args, p["w_C"], p["w_A"], alpha)
    p = gp_params()
    return run_gp(*args, p["g_C"], p["g_A"], p["P"], p["mode"], alpha)


def kpi_row(r: ModelResult):
    k = r.kpis
    cols = st.columns(4)
    cols[0].metric("Requests served", f"{k['served']:,}")
    cols[1].metric("Unmet requests", f"{k['unmet']:,}")
    cols[2].metric("Overall service rate", pct(k["service_rate"]))
    cols[3].metric("Capacity utilisation", pct(k["utilisation"]))
    cols = st.columns(4)
    cols[0].metric("City service rate", pct(k["rate_C"]))
    cols[1].metric("Airport service rate", pct(k["rate_A"]))
    cols[2].metric("Overall City–Airport gap", pct(k["rate_gap"]),
                   help="Difference between the City and Airport service rates over all periods combined")
    cols[3].metric("Mean gap per period", pct(k["mean_period_gap"]),
                   help="Average, over periods, of |City rate − Airport rate| in that hour. "
                        "This is what the Goal Programming balance goal (G3) minimises.")


def fig_allocation(t: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=t["period"], y=t["Y_C"], name="Drivers → City", marker_color=LOC["City"])
    fig.add_bar(x=t["period"], y=t["Y_A"], name="Drivers → Airport", marker_color=LOC["Airport"])
    fig.add_scatter(x=t["period"], y=t["N_adj"], name="Available drivers", mode="lines",
                    line=dict(color="#52514e", width=2, dash="dot"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(type="category", title="Period")
    fig.update_yaxes(title="Drivers")
    return stats.style(fig, "Optimal driver allocation per period")


def fig_rates(t: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=t["period"], y=t["rate_C"], name="City", mode="lines+markers",
                    line=dict(color=LOC["City"], width=2), marker_size=8)
    fig.add_scatter(x=t["period"], y=t["rate_A"], name="Airport", mode="lines+markers",
                    line=dict(color=LOC["Airport"], width=2), marker_size=8)
    fig.update_xaxes(type="category", title="Period")
    fig.update_yaxes(title="Service rate", tickformat=".0%", range=[0, 1.05])
    return stats.style(fig, title)


def fig_sweep(df: pd.DataFrame, x: str, x_title: str, x_fmt: str | None = None):
    """Two single-axis charts: unmet requests, and service rates."""
    f1 = go.Figure(go.Scatter(x=df[x], y=df["unmet"], mode="lines+markers", name="Unmet requests",
                              line=dict(color=SER["Model 1 (ILP)"], width=2), marker_size=8))
    f1.update_yaxes(title="Unmet requests")
    f2 = go.Figure()
    f2.add_scatter(x=df[x], y=df["rate_C"], name="City", mode="lines+markers",
                   line=dict(color=LOC["City"], width=2), marker_size=8)
    f2.add_scatter(x=df[x], y=df["rate_A"], name="Airport", mode="lines+markers",
                   line=dict(color=LOC["Airport"], width=2), marker_size=8)
    f2.update_yaxes(title="Service rate", tickformat=".0%")
    for f in (f1, f2):
        f.update_xaxes(title=x_title, tickformat=x_fmt)
    return stats.style(f1, "Unmet demand"), stats.style(f2, "Service rate by location")


def sweep_table(df: pd.DataFrame, first: str, first_fmt=None) -> pd.DataFrame:
    out = df.copy()
    if first_fmt:
        out[first] = out[first].map(first_fmt)
    for c in ["service_rate", "rate_C", "rate_A", "rate_gap", "utilisation"]:
        out[c] = out[c].map(pct)
    return out


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🚕 Uber capacity allocation")
mode = st.sidebar.radio("Data input", ["Upload CSV", "Enter values manually"], key="input_mode")
returned_to_manual = mode == "Enter values manually" and st.session_state.get("prev_mode", mode) != mode
st.session_state["prev_mode"] = mode
alpha = st.sidebar.slider("Capacity adjustment (drivers, %)", -50, 50, 0, 5,
                          help="Scales the number of active drivers in every period: N' = ⌊N·(1+α)⌋") / 100
st.sidebar.caption("Models: Integer Programming (Model 1) and Goal Programming (Model 2). "
                   "Solver: PuLP with HiGHS / CBC.")

st.title("Optimizing Uber Service Capacity Allocation")
st.caption("Integer Programming and Goal Programming for allocating drivers between City and Airport pickups.")

tab_data, tab_stats, tab_opt, tab_sens, tab_cmp = st.tabs(
    ["📁 Data", "📊 Statistics", "⚙️ Optimisation", "🔍 Sensitivity analysis", "⚖️ Comparison"])

requests: pd.DataFrame | None = st.session_state.get("requests")
data: PlanningData | None = None

# ---------------------------------------------------------------- Data tab
with tab_data:
    if mode == "Upload CSV":
        st.subheader("Upload the Uber request dataset")
        c1, c2 = st.columns([3, 1])
        upload = c1.file_uploader("CSV with columns: Request id, Pickup point, Driver id, Status, "
                                  "Request timestamp, Drop timestamp", type=["csv"])
        use_sample = c2.button("Use bundled sample dataset", disabled=not SAMPLE_CSV.exists())
        try:
            if upload is not None:
                requests = load_csv_bytes(upload.getvalue())
                st.session_state["requests"] = requests
                st.session_state["requests_name"] = upload.name
            elif use_sample:
                requests = load_csv_bytes(SAMPLE_CSV.read_bytes())
                st.session_state["requests"] = requests
                st.session_state["requests_name"] = SAMPLE_CSV.name
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not read the file: {exc}")
            requests = None
            st.session_state["requests"] = None

        if requests is not None:
            data = build_planning_data(requests)
            st.success(f"Loaded **{st.session_state.get('requests_name', 'dataset')}** – "
                       f"{len(requests):,} requests, {len(data.periods)} hourly planning periods, "
                       f"estimated productivity q = {data.q:.3f} rides per driver-hour.")
            st.markdown("**Raw requests (first 200 rows)**")
            st.dataframe(requests.head(200), width="stretch", hide_index=True)
            st.markdown("**Planning table** – demand `D`, active drivers `N` and historical completions per period")
            st.dataframe(data.periods.rename(columns={
                "D_C": "City demand", "D_A": "Airport demand", "N": "Active drivers",
                "hist_C": "City completed", "hist_A": "Airport completed"}), width="stretch", hide_index=True)
        else:
            st.info("Upload a CSV or click **Use bundled sample dataset** to begin.")
    else:
        st.subheader("Enter demand and drivers value by value")
        st.markdown("Edit any cell, add rows with **+** at the bottom, or delete rows. "
                    "Each row is one planning period.")
        if "manual_seed" not in st.session_state:
            st.session_state["manual_seed"] = example_manual_table()
            st.session_state["manual_version"] = 0
        elif returned_to_manual and "manual_saved" in st.session_state:
            # The editor's own state is dropped while hidden; restore the user's last edits.
            st.session_state["manual_seed"] = st.session_state["manual_saved"]
            st.session_state["manual_version"] += 1
        c1, c2, c3 = st.columns([1, 1, 2])
        if c1.button("Pre-fill from loaded dataset", disabled=requests is None,
                     help="Load the CSV first (Upload CSV mode)"):
            base = build_planning_data(requests)
            st.session_state["manual_seed"] = planning_to_manual_table(base)
            st.session_state["manual_q"] = round(base.q, 3)
            st.session_state["manual_version"] += 1
        if c2.button("Reset to example rows"):
            st.session_state["manual_seed"] = example_manual_table()
            st.session_state["manual_version"] += 1
        q_in = c3.number_input("Productivity q (rides per driver per hour)", min_value=0.01, max_value=10.0,
                               value=st.session_state.get("manual_q", DEFAULT_Q), step=0.05, format="%.3f")
        st.session_state["manual_q"] = q_in
        edited = st.data_editor(
            st.session_state["manual_seed"], num_rows="dynamic", width="stretch", hide_index=True,
            key=f"manual_editor_{st.session_state['manual_version']}",
            column_config={
                "Period": st.column_config.TextColumn("Period", help="Any label, e.g. 09:00"),
                "City demand": st.column_config.NumberColumn("City demand", min_value=0, step=1),
                "Airport demand": st.column_config.NumberColumn("Airport demand", min_value=0, step=1),
                "Active drivers": st.column_config.NumberColumn("Active drivers", min_value=0, step=1),
            })
        st.session_state["manual_saved"] = edited
        data, errors = parse_manual_table(edited, q_in)
        for e in errors:
            st.warning(f"Row skipped – {e}")
        if data.periods.empty:
            st.info("Enter at least one valid row to run the models.")
            data = None
        else:
            st.success(f"{len(data.periods)} valid period(s) ready for optimisation.")

# ---------------------------------------------------------------- Statistics tab
with tab_stats:
    if data is None:
        st.info("Load data in the **Data** tab first.")
    elif data.source == "csv" and requests is not None:
        blocks = ["Dataset overview", "Trip outcome by pickup point", "Hourly demand",
                  "Supply–demand gap heatmap", "Trip duration and productivity q", "Demand vs capacity per period"]
        chosen = st.multiselect("Choose statistics to generate", blocks, default=blocks[:2])
        if "Dataset overview" in chosen:
            o = stats.overview(requests)
            st.subheader("Dataset overview")
            c = st.columns(5)
            c[0].metric("Requests", f"{o['rows']:,}")
            c[1].metric("Columns", o["columns"])
            c[2].metric("From", o["start"].strftime("%d %b %Y"))
            c[3].metric("To", o["end"].strftime("%d %b %Y"))
            c[4].metric("Completed", f"{o['status_counts'].get('Trip Completed', 0):,}")
            c1, c2, c3 = st.columns(3)
            c1.markdown("**Requests by pickup point**")
            c1.dataframe(pd.Series(o["pickup_counts"], name="Requests"), width="stretch")
            c2.markdown("**Requests by status**")
            c2.dataframe(pd.Series(o["status_counts"], name="Requests"), width="stretch")
            c3.markdown("**Missing values** (expected: no driver when no car available; no drop when not completed)")
            c3.dataframe(pd.Series(o["missing"], name="Missing"), width="stretch")
        if "Trip outcome by pickup point" in chosen:
            show_chart_and_table(stats.fig_status_by_pickup(requests), stats.status_by_pickup(requests).reset_index(), "status")
        if "Hourly demand" in chosen:
            show_chart_and_table(stats.fig_hourly_demand(requests), stats.hourly_demand(requests).reset_index(), "hourly")
        if "Supply–demand gap heatmap" in chosen:
            gap = stats.gap_by_hour(requests)
            show_chart_and_table(stats.fig_gap_heatmap(requests), gap.map(pct).reset_index(), "gap")
            worst = gap.stack().sort_values(ascending=False).head(3)
            st.caption("Largest gaps: " + "; ".join(f"{loc} at {h}:00 ({v:.0%} not served)" for (loc, h), v in worst.items()))
        if "Trip duration and productivity q" in chosen:
            show_chart_and_table(stats.fig_duration_hist(requests),
                                 requests["duration_min"].describe().to_frame("minutes").reset_index(), "duration")
        if "Demand vs capacity per period" in chosen:
            show_chart_and_table(stats.fig_drivers_vs_demand(data), data.periods, "dvc")
    else:
        st.subheader("Summary of entered values")
        summary = data.periods[["D_C", "D_A", "N"]].rename(
            columns={"D_C": "City demand", "D_A": "Airport demand", "N": "Active drivers"}).describe().T
        st.dataframe(summary, width="stretch")
        show_chart_and_table(stats.fig_drivers_vs_demand(data), data.periods, "dvc_manual")

# ---------------------------------------------------------------- Optimisation tab
with tab_opt:
    model = st.radio("Choose optimisation model", [MODEL_1, MODEL_2], horizontal=True, key="model_choice")
    if model == MODEL_1:
        with st.expander("Formulation", expanded=False):
            st.latex(r"\min Z=\sum_t \left(w_C U_{C,t}+w_A U_{A,t}\right)")
            st.latex(r"Y_{C,t}+Y_{A,t}\le N'_t,\quad X_{l,t}\le q\,Y_{l,t},\quad X_{l,t}+U_{l,t}=D_{l,t}")
            st.latex(r"Y_{l,t},X_{l,t}\in\mathbb{Z}_{\ge 0},\; U_{l,t}\ge 0,\quad N'_t=\lfloor N_t(1+\alpha)\rfloor")
            st.markdown("`Y` = drivers assigned to each location (whole drivers), `X` = requests served, `U` = unmet.")
        c1, c2 = st.columns(2)
        with c1:
            param_widget(st.number_input, "Weight of unmet City request (w_C)", "w_C", 0.0, 10.0, step=0.1)
        with c2:
            param_widget(st.number_input, "Weight of unmet Airport request (w_A)", "w_A", 0.0, 10.0, step=0.1)
    else:
        with st.expander("Formulation", expanded=False):
            st.latex(r"\min\; P_1\frac{\sum U}{\sum D}+P_2\frac{\sum (d^-_C+d^-_A)}{\sum gD}"
                     r"+P_3\frac{\sum_{t\in B} (b^-_t+b^+_t)}{|B|}")
            st.latex(r"X_{l,t}+d^-_{l,t}-d^+_{l,t}=g_l D_{l,t}\qquad \frac{X_{C,t}}{D_{C,t}}-\frac{X_{A,t}}{D_{A,t}}+b^-_t-b^+_t=0"
                     r"\quad (t\in B)")
            st.markdown("Plus all Model 1 constraints. `B` = periods with demand at **both** locations (the balance "
                        "goal is undefined otherwise). Goals: **G1** unmet demand, **G2** service-target shortfall, "
                        "**G3** City–Airport service imbalance. Pre-emptive mode optimises G1, then G2, then G3.")
        c1, c2, c3 = st.columns(3)
        with c1:
            param_widget(st.slider, "City service target g_C (%)", "g_C", 0, 100, step=5)
            param_widget(st.slider, "Airport service target g_A (%)", "g_A", 0, 100, step=5)
        with c2:
            param_widget(st.number_input, "P1 – importance of reducing unmet demand", "P1", 0.0, 100.0, step=1.0)
            param_widget(st.number_input, "P2 – importance of meeting targets", "P2", 0.0, 100.0, step=1.0)
            param_widget(st.number_input, "P3 – importance of balanced service", "P3", 0.0, 100.0, step=1.0)
        st.session_state["gp_mode"] = c3.radio(
            "Solve mode", GP_MODES, index=GP_MODES.index(st.session_state["gp_mode"]), key="wgt_gp_mode",
            help="Pre-emptive ignores P1–P3 and optimises the goals strictly in priority order.")

    data_fp = data_fingerprint(data) if data is not None else None
    opt_fp = (data_fp, model, tuple(sorted((ilp_params() if model == MODEL_1 else gp_params()).items())), alpha)
    if data is None:
        st.info("Load data in the **Data** tab first.")
    elif st.button("Solve", type="primary"):
        with st.spinner("Solving…"):
            remember(f"result_{model}", opt_fp, solve(model, data, alpha))

    res = recall(f"result_{model}", opt_fp, "Solve") if data is not None else None
    if res is not None:
        if res.status != "Optimal":
            st.error(f"Solver status: {res.status}. Check the inputs and try again.")
        else:
            st.caption(f"Status: **{res.status}** · capacity adjustment {alpha:+.0%} · q = {data.q:.3f}")
            kpi_row(res)
            t = res.table
            if model == MODEL_2:
                g = res.kpis["goals"]
                c = st.columns(4)
                c[0].metric("G1 unmet share", pct(g["G1"]))
                c[1].metric("G2 target shortfall share", pct(g["G2"]))
                c[2].metric("G3 mean imbalance per period", pct(g["G3"]))
                c[3].metric("Total shortfall below targets", f"{res.kpis['shortfall']:.1f} requests")
            st.plotly_chart(fig_allocation(t), width="stretch", key="alloc")
            st.plotly_chart(fig_rates(t, "Achieved service rate per period"), width="stretch", key="rates")
            st.markdown("**Results per period**")
            st.dataframe(t, width="stretch", hide_index=True)
            st.download_button("Download results (CSV)", t.to_csv(index=False).encode(),
                               file_name="model1_ilp_results.csv" if model == MODEL_1 else "model2_gp_results.csv",
                               mime="text/csv")

# ---------------------------------------------------------------- Sensitivity tab
with tab_sens:
    if data is None:
        st.info("Load data in the **Data** tab first.")
    else:
        args = (data.periods, data.q, data.source)
        sens_model = st.radio("Model to analyse", [MODEL_1, MODEL_2], horizontal=True, key="sens_model",
                              help="Uses the parameters currently set in the Optimisation tab.")
        key = "ilp" if sens_model == MODEL_1 else "gp"
        params = tuple(sorted((ilp_params() if key == "ilp" else gp_params()).items()))
        base_fp = (data_fp, key, params)

        st.subheader("1. Capacity (number of drivers)")
        cap_text = st.text_input("Capacity changes to test (%)", "-20, -10, 0, 10, 20")
        if st.button("Run capacity sensitivity"):
            try:
                alphas = tuple(parse_number_list(cap_text, 0.01))
                with st.spinner("Re-solving for each capacity level…"):
                    remember("sens_cap", base_fp, run_capacity_sweep(*args, key, alphas, params))
            except ValueError as exc:
                st.error(f"Invalid list: {exc}")
        df = recall("sens_cap", base_fp, "Run capacity sensitivity")
        if df is not None:
            f1, f2 = fig_sweep(df, "alpha", "Capacity change", ".0%")
            c1, c2 = st.columns(2)
            c1.plotly_chart(f1, width="stretch", key="cap1")
            c2.plotly_chart(f2, width="stretch", key="cap2")
            st.dataframe(sweep_table(df, "alpha", lambda a: f"{a:+.0%}"), width="stretch", hide_index=True)

        st.subheader("2. Driver productivity q")
        c1, c2, c3 = st.columns(3)
        q_lo = c1.number_input("From q", 0.1, 5.0, round(max(data.q * 0.8, 0.1), 2), 0.05)
        q_hi = c2.number_input("To q", 0.1, 5.0, round(data.q * 1.2, 2), 0.05)
        q_n = c3.number_input("Number of values", 2, 10, 5, 1)
        if st.button("Run productivity sensitivity"):
            qs = tuple(np.round(np.linspace(min(q_lo, q_hi), max(q_lo, q_hi), int(q_n)), 3))
            with st.spinner("Re-solving for each q…"):
                remember("sens_q", base_fp + (alpha,),
                         run_productivity_sweep(*args, key, qs, params + (("alpha", alpha),)))
        df = recall("sens_q", base_fp + (alpha,), "Run productivity sensitivity")
        if df is not None:
            f1, f2 = fig_sweep(df, "q", "Rides per driver-hour (q)")
            c1, c2 = st.columns(2)
            c1.plotly_chart(f1, width="stretch", key="q1")
            c2.plotly_chart(f2, width="stretch", key="q2")
            st.dataframe(sweep_table(df, "q"), width="stretch", hide_index=True)

        st.subheader("3. Service-level target (Goal Programming)")
        tgt_text = st.text_input("Targets to test for both locations (%)", "50, 60, 70, 80, 90")
        gp = tuple((k, v) for k, v in sorted(gp_params().items()) if k not in ("g_C", "g_A"))
        tgt_fp = (data_fp, gp, alpha)
        if st.button("Run target sensitivity"):
            try:
                targets = tuple(parse_number_list(tgt_text, 0.01))
                with st.spinner("Re-solving the goal programme for each target…"):
                    remember("sens_tgt", tgt_fp, run_target_sweep(*args, targets, gp + (("alpha", alpha),)))
            except ValueError as exc:
                st.error(f"Invalid list: {exc}")
        df = recall("sens_tgt", tgt_fp, "Run target sensitivity")
        if df is not None:
            f1 = go.Figure(go.Scatter(x=df["target"], y=df["shortfall"], mode="lines+markers", name="Shortfall",
                                      line=dict(color=SER["Model 2 (GP)"], width=2), marker_size=8))
            f1.update_xaxes(title="Service target", tickformat=".0%")
            f1.update_yaxes(title="Requests below target")
            _, f2 = fig_sweep(df, "target", "Service target", ".0%")
            c1, c2 = st.columns(2)
            c1.plotly_chart(stats.style(f1, "Total shortfall below targets"), width="stretch", key="t1")
            c2.plotly_chart(f2, width="stretch", key="t2")
            st.dataframe(sweep_table(df, "target", lambda g: f"{g:.0%}"), width="stretch", hide_index=True)

        st.subheader("4. Shadow prices of driver capacity (LP relaxation of Model 1)")
        st.caption("The dual value of each period's driver-capacity constraint in the LP relaxation: the marginal "
                   "reduction in (weighted) unmet requests per extra driver, ≈ q when drivers are the bottleneck "
                   "and 0 when the period has spare drivers. In the integer model a whole extra driver serves "
                   "⌊q⌋ or ⌈q⌉ more requests, so the dual is a marginal rate, not an exact integer gain.")
        p = ilp_params()
        sp_fp = (data_fp, p["w_C"], p["w_A"], alpha)
        if st.button("Compute shadow prices"):
            remember("sens_sp", sp_fp, run_shadow(*args, p["w_C"], p["w_A"], alpha))
        sp = recall("sens_sp", sp_fp, "Compute shadow prices")
        if sp is not None:
            if sp.empty:
                st.error("Could not compute shadow prices for this data.")
            else:
                c = st.columns(3)
                c[0].metric("Periods where capacity binds", f"{(sp['shadow_price'] > 1e-9).sum()} of {len(sp)}")
                c[1].metric("Highest shadow price", f"{sp['shadow_price'].max():.3f}")
                c[2].metric("Periods with spare drivers", f"{(sp['shadow_price'] <= 1e-9).sum()}")
                by_period = sp.sort_values("period")
                fig = go.Figure(go.Bar(x=by_period["period"], y=by_period["shadow_price"],
                                       marker_color=SER["Model 1 (ILP)"], name="Shadow price"))
                fig.update_xaxes(type="category", title="Period")
                fig.update_yaxes(title="Marginal unmet reduction per driver")
                show_chart_and_table(stats.style(fig, "Shadow price per period"), sp, "sp")

# ---------------------------------------------------------------- Comparison tab
with tab_cmp:
    if data is None:
        st.info("Load data in the **Data** tab first.")
    else:
        st.markdown("Compares historical outcomes with both models, using the parameters set in the "
                    "**Optimisation** tab and the sidebar capacity adjustment.")
        cmp_fp = (data_fp, tuple(sorted(ilp_params().items())), tuple(sorted(gp_params().items())), alpha)
        if st.button("Run comparison", type="primary"):
            with st.spinner("Solving both models…"):
                remember("cmp", cmp_fp, {"Model 1 (ILP)": solve(MODEL_1, data, alpha),
                                         "Model 2 (GP)": solve(MODEL_2, data, alpha)})
        cmp = recall("cmp", cmp_fp, "Run comparison")
        if cmp is not None:
            results = {}
            hist = historical_result(data)
            if hist is not None:
                results["Historical"] = hist
            results.update(cmp)
            failed = [n for n, r in results.items() if r.status not in ("Optimal", "Historical")]
            for n in failed:
                st.error(f"{n}: solver status {results[n].status}")
            results = {n: r for n, r in results.items() if n not in failed}
            rows = []
            for name, r in results.items():
                k = r.kpis
                rows.append({"Scenario": name, "Served": k["served"], "Unmet": k["unmet"],
                             "Service rate": pct(k["service_rate"]), "City rate": pct(k["rate_C"]),
                             "Airport rate": pct(k["rate_A"]), "City–Airport gap": pct(k["rate_gap"]),
                             "Mean gap per period": pct(k["mean_period_gap"]),
                             "Utilisation": pct(k["utilisation"])})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            names = list(results)
            fig = go.Figure()
            for loc, key_ in [("City", "rate_C"), ("Airport", "rate_A")]:
                fig.add_bar(x=names, y=[results[n].kpis[key_] for n in names], name=loc, marker_color=LOC[loc],
                            text=[pct(results[n].kpis[key_]) for n in names], textposition="outside")
            fig.update_layout(barmode="group")
            fig.update_yaxes(title="Service rate", tickformat=".0%", range=[0, 1.1])
            fig2 = go.Figure(go.Bar(x=names, y=[results[n].kpis["unmet"] for n in names],
                                    marker_color=[SER[n] for n in names],
                                    text=[f"{results[n].kpis['unmet']:,}" for n in names], textposition="outside"))
            fig2.update_yaxes(title="Unmet requests")
            c1, c2 = st.columns(2)
            c1.plotly_chart(stats.style(fig, "Service rate by location"), width="stretch", key="cmp1")
            c2.plotly_chart(stats.style(fig2, "Unmet requests"), width="stretch", key="cmp2")
            if hist is not None:
                st.info("**Assumption:** capacity is estimated from drivers seen on completed *or cancelled* trips "
                        "(N), times productivity q. Cancellations are treated as capacity that could have served "
                        "a request, so both models will serve more than the historical completions. The comparison "
                        "shows the potential of better allocation, not a guaranteed improvement.")
            else:
                st.caption("Historical outcomes are not available for manually entered data.")
