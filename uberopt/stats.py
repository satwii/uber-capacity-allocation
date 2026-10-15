"""Descriptive statistics and Plotly figures for the request data."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .data import COMPLETED, PlanningData

# Fixed colour identities (validated palette): an entity keeps its colour on every chart.
LOCATION_COLORS = {"City": "#2a78d6", "Airport": "#eb6834"}
STATUS_COLORS = {"Trip Completed": "#1baf7a", "Cancelled": "#eda100", "No Cars Available": "#6250d6"}
SERIES_COLORS = {"Historical": "#1baf7a", "Model 1 (ILP)": "#2a78d6", "Model 2 (GP)": "#eb6834"}
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Recessive axes/grid, legend on top, thin bar gaps."""
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        bargap=0.15,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    return fig


def overview(requests: pd.DataFrame) -> dict:
    return {
        "rows": int(len(requests)),
        "columns": 6,
        "missing": {c: int(requests[c].isna().sum()) for c in ["Driver id", "Drop timestamp"]},
        "start": requests["Request timestamp"].min(),
        "end": requests["Request timestamp"].max(),
        "pickup_counts": requests["Pickup point"].value_counts().to_dict(),
        "status_counts": requests["Status"].value_counts().to_dict(),
    }


def status_by_pickup(requests: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(requests["Pickup point"], requests["Status"]).reindex(
        columns=[s for s in STATUS_COLORS if s in requests["Status"].unique()]
    )


def hourly_demand(requests: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(requests["hour"], requests["Pickup point"]).reindex(range(24), fill_value=0)


def gap_by_hour(requests: pd.DataFrame) -> pd.DataFrame:
    """Share of requests NOT completed (supply-demand gap), hour x pickup point."""
    unmet = requests.assign(unmet=requests["Status"] != COMPLETED)
    return unmet.pivot_table(index="Pickup point", columns="hour", values="unmet", aggfunc="mean").reindex(columns=range(24))


def fig_status_by_pickup(requests: pd.DataFrame) -> go.Figure:
    long = status_by_pickup(requests).reset_index().melt(id_vars="Pickup point", var_name="Status", value_name="Requests")
    fig = px.bar(long, x="Pickup point", y="Requests", color="Status", barmode="group",
                 color_discrete_map=STATUS_COLORS, text="Requests")
    fig.update_traces(textposition="outside")
    return style(fig, "Trip outcome by pickup point")


def fig_hourly_demand(requests: pd.DataFrame) -> go.Figure:
    long = hourly_demand(requests).reset_index().melt(id_vars="hour", var_name="Pickup point", value_name="Requests")
    fig = px.line(long, x="hour", y="Requests", color="Pickup point", markers=True,
                  color_discrete_map=LOCATION_COLORS)
    fig.update_traces(line_width=2, marker_size=8)
    fig.update_xaxes(dtick=2, title="Hour of day")
    return style(fig, "Requests by hour of day (all 5 days)")


def fig_gap_heatmap(requests: pd.DataFrame) -> go.Figure:
    g = gap_by_hour(requests)
    fig = go.Figure(go.Heatmap(
        z=g.values, x=list(g.columns), y=list(g.index), colorscale=SEQUENTIAL, zmin=0, zmax=1,
        colorbar=dict(title="Unmet share", tickformat=".0%"),
        hovertemplate="Hour %{x}, %{y}<br>Unmet: %{z:.0%}<extra></extra>",
        xgap=2, ygap=2,
    ))
    fig.update_xaxes(dtick=2, title="Hour of day")
    fig = style(fig, "Supply–demand gap: share of requests not completed")
    fig.update_layout(hovermode="closest")
    return fig


def fig_duration_hist(requests: pd.DataFrame) -> go.Figure:
    d = requests["duration_min"].dropna()
    fig = px.histogram(d, x=d, nbins=40, color_discrete_sequence=[LOCATION_COLORS["City"]])
    med = d.median()
    fig.add_vline(x=med, line_dash="dash", line_width=2,
                  annotation_text=f"median {med:.1f} min → q = {60 / med:.2f} rides/driver-hour")
    fig.update_xaxes(title="Completed-trip duration (minutes)")
    fig.update_yaxes(title="Trips")
    fig = style(fig, "Trip duration distribution")
    fig.update_layout(showlegend=False, hovermode="closest")
    return fig


def fig_drivers_vs_demand(data: PlanningData) -> go.Figure:
    p = data.periods
    fig = go.Figure()
    fig.add_bar(x=p["period"], y=p["D_C"], name="City demand", marker_color=LOCATION_COLORS["City"])
    fig.add_bar(x=p["period"], y=p["D_A"], name="Airport demand", marker_color=LOCATION_COLORS["Airport"])
    fig.add_scatter(x=p["period"], y=p["N"] * data.q, name="Estimated capacity (q × drivers)",
                    mode="lines", line=dict(color="#52514e", width=2))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Period", type="category")
    fig.update_yaxes(title="Requests")
    return style(fig, "Demand vs estimated service capacity per period")
