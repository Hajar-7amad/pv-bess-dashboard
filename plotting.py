"""
Plotly figure factory with a consistent, restrained engineering style.

Every figure goes through ``style()`` so axes, fonts and legends look identical across
pages. Colours follow the entity (battery or variable) – never its rank – so filtering
never repaints the remaining series.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import ALL_VARIABLES

# Muted categorical sequence for variables (fixed order; batteries use their own colours).
SERIES_COLORS = ["#2F5D8A", "#C2693E", "#4C8C6B", "#7A5C99", "#8C7A3E", "#3E7F8C",
                 "#9C4F5E", "#5B6B7F"]
INK = "#1F2933"
MUTED = "#6B7785"
GRID = "rgba(120,130,145,0.18)"
FONT = "Inter, Segoe UI, Helvetica, Arial, sans-serif"
SEQ_SCALE = [[0, "#F2F5F8"], [0.5, "#7FA3C6"], [1, "#1E3F5F"]]
DIV_SCALE = [[0, "#2F5D8A"], [0.5, "#EEF0F2"], [1, "#C2693E"]]


def label(col: Optional[str]) -> str:
    if not col:
        return ""
    v = ALL_VARIABLES.get(col)
    if v:
        return v.display
    return {"timestamp": "Time", "battery": "Battery", "operating_mode": "Operating mode",
            "actual": "Actual R_cn [h⁻¹]", "predicted": "Predicted R_cn [h⁻¹]",
            "residual": "Residual [h⁻¹]"}.get(col, col.replace("_", " "))


def style(fig: go.Figure, title: str = "", height: int = 420, legend: bool = True,
          xlabel: Optional[str] = None, ylabel: Optional[str] = None) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=15, color=INK)),
        template="plotly_white", height=height, font=dict(family=FONT, size=12, color=INK),
        margin=dict(l=60, r=24, t=56 if title else 24, b=48),
        showlegend=legend, hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor="#C9D0D8",
                     ticks="outside", tickcolor="#C9D0D8")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor="#C9D0D8",
                     ticks="outside", tickcolor="#C9D0D8")
    if xlabel is not None:
        fig.update_xaxes(title_text=xlabel, selector=dict(anchor="y"))
    if ylabel is not None:
        fig.update_yaxes(title_text=ylabel, selector=dict(anchor="x"))
    return fig


def plotly_config(fmt: str = "png", scale: int = 2, filename: str = "pv_bess_figure") -> dict:
    """Modebar config: zoom/pan/hover/legend toggles plus PNG or SVG download."""
    return {
        "displaylogo": False,
        "toImageButtonOptions": {"format": fmt, "filename": filename, "scale": scale},
        "modeBarButtonsToAdd": ["drawline", "eraseshape"],
    }


def color_for(names: Sequence[str], fixed: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    fixed = dict(fixed or {})
    out, i = {}, 0
    for n in names:
        if n in fixed:
            out[n] = fixed[n]
        else:
            out[n] = SERIES_COLORS[i % len(SERIES_COLORS)]
            i += 1
    return out


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------
def break_gaps(d: pd.DataFrame, x: str = "timestamp", factor: float = 3.0) -> pd.DataFrame:
    """Insert empty rows where the time step exceeds ``factor`` × median step, so lines are
    not drawn across data gaps or across observations removed by a filter."""
    if x != "timestamp" or len(d) < 3:
        return d
    d = d.sort_values(x)
    dt = d[x].diff()
    med = dt[dt > pd.Timedelta(0)].median()
    if pd.isna(med):
        return d
    m = dt > factor * med
    if not m.any():
        return d
    gaps = pd.DataFrame({x: d.loc[m, x] - med / 2})
    return pd.concat([d, gaps], ignore_index=True).sort_values(x, kind="stable")


def _unit(c: str) -> str:
    v = ALL_VARIABLES.get(c)
    return v.unit if v and v.unit else ""


def time_series(df: pd.DataFrame, cols: List[str], mode: str = "panels", secondary: Optional[str] = None,
                title: str = "", height: Optional[int] = None, x: str = "timestamp",
                group: Optional[str] = None, colors: Optional[Dict[str, str]] = None,
                markers: bool = False, rangeslider: bool = True) -> go.Figure:
    """Multi-variable time-series chart.

    mode = 'panels'    one stacked panel per variable sharing the time axis (recommended
                       when units differ)
           'shared'    all series on one y-axis
           'secondary' series on the primary axis + ``secondary`` on a right-hand axis
    ``group`` (e.g. 'battery') splits each variable into one trace per group value.
    """
    cols = [c for c in cols if c in df]
    if not cols:
        return style(go.Figure(), title or "No variable selected")
    groups = [None] if not group or group not in df else list(pd.unique(df[group].dropna()))
    gcolors = color_for([g for g in groups if g is not None], colors)
    vcolors = color_for(cols)
    trace_mode = "lines+markers" if markers else "lines"

    def _color(c, g):
        return gcolors.get(g) if g is not None else vcolors[c]

    if mode == "panels" and len(cols) > 1:
        fig = make_subplots(rows=len(cols), cols=1, shared_xaxes=True, vertical_spacing=min(0.07, 0.3 / len(cols)))
        for i, c in enumerate(cols, start=1):
            for g in groups:
                d = break_gaps(df if g is None else df[df[group] == g], x)
                fig.add_trace(go.Scattergl(x=d[x], y=d[c], mode=trace_mode, line=dict(width=1.6, color=_color(c, g)),
                                           marker=dict(size=4), name=(f"{g}" if g is not None else label(c)),
                                           legendgroup=str(g), showlegend=(i == 1 and g is not None) or g is None,
                                           hovertemplate=f"{label(c)}: %{{y:.4g}}<br>%{{x}}<extra>{g or ''}</extra>"),
                              row=i, col=1)
            fig.update_yaxes(title_text=_unit(c), title_font=dict(size=11), row=i, col=1)
            ax = "" if i == 1 else str(i)
            fig.add_annotation(xref=f"x{ax} domain", yref=f"y{ax} domain", x=0.0, y=1.0, xanchor="left",
                               yanchor="bottom", showarrow=False, text=ALL_VARIABLES[c].label if c in ALL_VARIABLES else c,
                               font=dict(size=11, color=MUTED))
        fig = style(fig, title, height or max(260, 170 * len(cols)), legend=groups != [None])
        fig.update_layout(margin=dict(t=72 if title else 36))
        fig.update_layout(hovermode="x unified")
        return fig

    fig = make_subplots(specs=[[{"secondary_y": mode == "secondary" and bool(secondary)}]])
    for c in cols:
        for g in groups:
            d = break_gaps(df if g is None else df[df[group] == g], x)
            nm = label(c) if g is None else (f"{g} – {label(c)}" if len(cols) > 1 else str(g))
            fig.add_trace(go.Scattergl(x=d[x], y=d[c], mode=trace_mode, name=nm,
                                       line=dict(width=1.6, color=_color(c, g)), marker=dict(size=4)),
                          secondary_y=False)
    if mode == "secondary" and secondary and secondary in df:
        for g in groups:
            d = break_gaps(df if g is None else df[df[group] == g], x)
            fig.add_trace(go.Scattergl(x=d[x], y=d[secondary], mode=trace_mode,
                                       name=f"{label(secondary)} (right axis)" + (f" – {g}" if g else ""),
                                       line=dict(width=1.6, dash="dot", color="#5B6B7F" if g is None else gcolors[g])),
                          secondary_y=True)
        fig.update_yaxes(title_text=label(secondary), secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text=label(cols[0]) if len(cols) == 1 else "Value", secondary_y=False)
    fig = style(fig, title, height or 440)
    fig.update_layout(hovermode="x unified")
    if rangeslider and x == "timestamp":
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06),
                         rangeselector=dict(buttons=[
                             dict(count=1, label="1d", step="day", stepmode="backward"),
                             dict(count=7, label="1w", step="day", stepmode="backward"),
                             dict(count=1, label="1m", step="month", stepmode="backward"),
                             dict(step="all", label="All")]))
    return fig


# ---------------------------------------------------------------------------
# Generic chart builder (Data Explorer & Plot Builder)
# ---------------------------------------------------------------------------
def build_chart(df: pd.DataFrame, chart: str, x: Optional[str], ys: List[str], y2: Optional[str] = None,
                color: Optional[str] = None, title: str = "", xlabel: str = "", ylabel: str = "",
                log_x: bool = False, log_y: bool = False, legend: bool = True, ts_mode: str = "shared",
                colors: Optional[Dict[str, str]] = None, nbins: int = 50, height: int = 480) -> go.Figure:
    ys = [y for y in ys if y in df]
    cmap = colors if color == "battery" else None
    if chart == "Line chart":
        if x == "timestamp" or x is None:
            fig = time_series(df, ys, mode=ts_mode if not y2 else "secondary", secondary=y2, title=title,
                              group=color if color in ("battery", "operating_mode") else None, colors=colors,
                              height=height)
        else:
            d = df.sort_values(x)
            fig = go.Figure()
            vc = color_for(ys)
            for y in ys:
                fig.add_trace(go.Scatter(x=d[x], y=d[y], mode="lines", name=label(y), line=dict(color=vc[y], width=1.6)))
            fig = style(fig, title, height)
    elif chart == "Scatter plot":
        long = df.melt(id_vars=[c for c in {x, color} if c and c not in ys], value_vars=ys,
                       var_name="series", value_name="value") if len(ys) > 1 else None
        if long is not None:
            long["series"] = long["series"].map(label)
            fig = px.scatter(long, x=x, y="value", color=color or "series", symbol="series" if color else None,
                             color_discrete_map=cmap, opacity=0.65, render_mode="webgl")
        else:
            cont = color and color in df and pd.api.types.is_numeric_dtype(df[color]) and df[color].nunique() > 12
            fig = px.scatter(df, x=x, y=ys[0] if ys else None, color=color, opacity=0.65, render_mode="webgl",
                             color_discrete_map=cmap, color_continuous_scale=SEQ_SCALE if cont else None,
                             labels={c: label(c) for c in df.columns})
        fig.update_traces(marker=dict(size=6, line=dict(width=0)))
        fig = style(fig, title, height)
    elif chart == "Bar chart":
        if x and ys:
            d = df.groupby([x] + ([color] if color and color != x else []), as_index=False)[ys].mean()
            long = d.melt(id_vars=[c for c in d.columns if c not in ys], value_vars=ys, var_name="series", value_name="value")
            long["series"] = long["series"].map(label)
            fig = px.bar(long, x=x, y="value", color=color if color and len(ys) == 1 else "series",
                         barmode="group", color_discrete_map=cmap)
        else:
            fig = go.Figure()
        fig = style(fig, title, height)
    elif chart == "Histogram":
        col = ys[0] if ys else x
        fig = px.histogram(df, x=col, color=color, nbins=nbins, barmode="overlay", opacity=0.6,
                           color_discrete_map=cmap, labels={col: label(col)})
        fig = style(fig, title, height)
        fig.update_yaxes(title_text="Count")
    elif chart == "Box plot":
        long = df.melt(id_vars=[c for c in {color} if c], value_vars=ys, var_name="series", value_name="value")
        long["series"] = long["series"].map(label)
        fig = px.box(long, x="series", y="value", color=color, points=False, color_discrete_map=cmap)
        fig = style(fig, title, height)
    elif chart == "Heatmap":
        # value of ys[0] by hour-of-day × date (or by x × color bins)
        col = ys[0] if ys else None
        if col is None:
            fig = go.Figure()
        else:
            d = df.dropna(subset=[col]).copy()
            d["date"] = d["timestamp"].dt.date
            d["hour"] = d["timestamp"].dt.hour
            pv = d.pivot_table(index="hour", columns="date", values=col, aggfunc="mean")
            fig = go.Figure(go.Heatmap(z=pv.values, x=[str(c) for c in pv.columns], y=pv.index,
                                       colorscale=SEQ_SCALE, colorbar=dict(title=label(col)),
                                       hovertemplate="%{x} %{y}:00<br>%{z:.4g}<extra></extra>"))
            fig = style(fig, title or f"{label(col)} – hour of day × date", height)
            fig.update_yaxes(title_text="Hour of day")
    elif chart == "Correlation matrix":
        fig = correlation_matrix(df, ys, title, height)
    else:
        fig = go.Figure()
    fig.update_layout(showlegend=legend)
    if xlabel:
        fig.update_xaxes(title_text=xlabel)
    elif x and chart in ("Scatter plot", "Bar chart") or (chart == "Line chart" and x != "timestamp"):
        fig.update_xaxes(title_text=label(x))
    if ylabel:
        fig.update_layout(yaxis_title=ylabel)
    elif len(ys) == 1 and chart in ("Scatter plot", "Bar chart", "Box plot"):
        fig.update_layout(yaxis_title=label(ys[0]))
    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def correlation_matrix(df: pd.DataFrame, cols: List[str], title: str = "", height: int = 520,
                       method: str = "pearson") -> go.Figure:
    cols = [c for c in cols if c in df and pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) < 2:
        return style(go.Figure(), "Select at least two numeric variables")
    c = df[cols].corr(method=method)
    labels = [label(x) for x in cols]
    fig = go.Figure(go.Heatmap(z=c.values, x=labels, y=labels, zmin=-1, zmax=1, colorscale=DIV_SCALE,
                               text=np.round(c.values, 2), texttemplate="%{text}",
                               hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
                               colorbar=dict(title="r")))
    fig = style(fig, title or f"Correlation matrix ({method})", max(height, 40 * len(cols) + 120), legend=False)
    fig.update_xaxes(tickangle=-35, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
def grouped_box(long: pd.DataFrame, col: str, colors: Dict[str, str], title: str = "",
                height: int = 360) -> go.Figure:
    fig = go.Figure()
    for b in long["battery"].unique():
        d = long.loc[long["battery"] == b, col].dropna()
        fig.add_trace(go.Box(y=d, name=b, marker_color=colors.get(b), boxmean=True,
                             line=dict(width=1.4)))
    fig = style(fig, title or label(col), height, legend=False)
    fig.update_yaxes(title_text=_unit(col) if not title else label(col))
    return fig


def grouped_hist(long: pd.DataFrame, col: str, colors: Dict[str, str], title: str = "",
                 norm: str = "probability density", nbins: int = 40, height: int = 340) -> go.Figure:
    fig = go.Figure()
    for b in long["battery"].unique():
        d = long.loc[long["battery"] == b, col].dropna()
        fig.add_trace(go.Histogram(x=d, name=b, marker_color=colors.get(b), opacity=0.55,
                                   histnorm=norm, nbinsx=nbins))
    fig.update_layout(barmode="overlay")
    fig = style(fig, title or label(col), height)
    fig.update_xaxes(title_text=label(col))
    fig.update_yaxes(title_text=norm.capitalize() if norm else "Count")
    return fig


def grouped_ecdf(long: pd.DataFrame, col: str, colors: Dict[str, str], title: str = "",
                 height: int = 340) -> go.Figure:
    fig = go.Figure()
    for b in long["battery"].unique():
        x = np.sort(long.loc[long["battery"] == b, col].dropna().to_numpy())
        if len(x) == 0:
            continue
        fig.add_trace(go.Scatter(x=x, y=np.arange(1, len(x) + 1) / len(x), mode="lines", name=b,
                                 line=dict(color=colors.get(b), width=2, shape="hv")))
    fig = style(fig, title or f"Cumulative distribution – {label(col)}", height)
    fig.update_xaxes(title_text=label(col))
    fig.update_yaxes(title_text="Cumulative fraction", range=[0, 1.02])
    return fig


def grouped_scatter(long: pd.DataFrame, x: str, y: str, colors: Dict[str, str], title: str = "",
                    height: int = 380) -> go.Figure:
    fig = go.Figure()
    for b in long["battery"].unique():
        d = long[long["battery"] == b]
        fig.add_trace(go.Scattergl(x=d[x], y=d[y], mode="markers", name=b, opacity=0.6,
                                   marker=dict(size=6, color=colors.get(b), line=dict(width=0))))
    fig = style(fig, title or f"{label(y)} vs {label(x)}", height)
    fig.update_xaxes(title_text=label(x))
    fig.update_yaxes(title_text=label(y))
    return fig


def bar_compare(df: pd.DataFrame, x: str, y: str, color: str, colors: Dict[str, str], title: str = "",
                height: int = 360, ytitle: str = "") -> go.Figure:
    fig = px.bar(df, x=x, y=y, color=color, barmode="group", color_discrete_map=colors)
    fig = style(fig, title, height)
    fig.update_yaxes(title_text=ytitle or label(y))
    fig.update_xaxes(title_text="")
    return fig


def actual_vs_predicted(preds: pd.DataFrame, title: str = "", color: str = "#2F5D8A",
                        height: int = 400) -> go.Figure:
    fig = go.Figure()
    for s, sym, op in (("train", "circle-open", 0.35), ("test", "circle", 0.8)):
        d = preds[preds["set"] == s]
        fig.add_trace(go.Scattergl(x=d["actual"], y=d["predicted"], mode="markers", name=s.capitalize(),
                                   marker=dict(size=6, color=color, symbol=sym, opacity=op),
                                   hovertemplate="Actual %{x:.4f}<br>Predicted %{y:.4f}<extra>" + s + "</extra>"))
    lo = float(np.nanmin(preds[["actual", "predicted"]].values)) if len(preds) else 0
    hi = float(np.nanmax(preds[["actual", "predicted"]].values)) if len(preds) else 1
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="1:1",
                             line=dict(color=MUTED, dash="dash", width=1)))
    fig = style(fig, title or "Actual vs predicted R_cn", height)
    fig.update_xaxes(title_text="Actual R_cn [h⁻¹]")
    fig.update_yaxes(title_text="Predicted R_cn [h⁻¹]")
    return fig


def residual_plot(preds: pd.DataFrame, color: str = "#2F5D8A", height: int = 340) -> go.Figure:
    d = preds[preds["set"] == "test"]
    fig = go.Figure(go.Scattergl(x=d["predicted"], y=d["residual"], mode="markers",
                                 marker=dict(size=6, color=color, opacity=0.7), name="Test"))
    fig.add_hline(y=0, line=dict(color=MUTED, dash="dash", width=1))
    fig = style(fig, "Residuals vs predicted (test)", height, legend=False)
    fig.update_xaxes(title_text="Predicted R_cn [h⁻¹]")
    fig.update_yaxes(title_text="Residual (actual − predicted) [h⁻¹]")
    return fig


def importance_bar(df: pd.DataFrame, value: str, labelcol: str, title: str, color: str = "#2F5D8A",
                   height: Optional[int] = None, xtitle: str = "") -> go.Figure:
    d = df.sort_values(value)
    fig = go.Figure(go.Bar(x=d[value], y=d[labelcol], orientation="h", marker_color=color,
                           hovertemplate="%{y}: %{x:.4g}<extra></extra>"))
    fig = style(fig, title, height or max(260, 26 * len(d) + 90), legend=False)
    fig.update_xaxes(title_text=xtitle or value)
    return fig


# ---------------------------------------------------------------------------
# Energy-flow diagram
# ---------------------------------------------------------------------------
def _fmt(v, unit, nd=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "unavailable"
    return f"{v:,.{nd}f} {unit}"


def energy_flow_diagram(snap: Dict[str, float], battery_name: str, nominal_kwh: float,
                        height: int = 430) -> go.Figure:
    """Schematic PV → DC bus → Battery / Inverter → AC load ↔ Grid with live values."""
    fig = go.Figure()
    boxes = {
        "PV array": (0.5, 0.92, f"260 kWp<br>{_fmt(snap.get('pv_power'), 'kW')}<br>"
                                f"{_fmt(snap.get('solar_irradiance'), 'W/m²', 0)}"),
        "DC bus": (0.5, 0.66, "1000 VDC"),
        f"Battery – {battery_name}": (0.2, 0.40, f"{nominal_kwh:,.0f} kWh<br>SOC {_fmt(snap.get('battery_soc'), '%')}<br>"
                                                 f"⇣ {_fmt(snap.get('battery_charge_power'), 'kW')}  "
                                                 f"⇡ {_fmt(snap.get('battery_discharge_power'), 'kW')}"),
        "Inverter": (0.8, 0.40, f"{_fmt(snap.get('inverter_temperature'), '°C')}"),
        "AC load": (0.5, 0.12, _fmt(snap.get("load_power"), "kW")),
        "Grid": (0.88, 0.12, _fmt(snap.get("grid_power"), "kW")),
    }
    for name, (x, y, txt) in boxes.items():
        fig.add_shape(type="rect", x0=x - 0.13, x1=x + 0.13, y0=y - 0.085, y1=y + 0.085,
                      line=dict(color="#8796A8", width=1.2), fillcolor="rgba(47,93,138,0.06)",
                      xref="x", yref="y")
        fig.add_annotation(x=x, y=y + 0.045, text=f"<b>{name}</b>", showarrow=False, font=dict(size=12))
        fig.add_annotation(x=x, y=y - 0.025, text=txt, showarrow=False, font=dict(size=11, color=MUTED))
    arrows = [((0.5, 0.835), (0.5, 0.745)), ((0.45, 0.575), (0.25, 0.485)), ((0.55, 0.575), (0.75, 0.485)),
              ((0.78, 0.315), (0.56, 0.205)), ((0.63, 0.12), (0.75, 0.12))]
    for i, ((x0, y0), (x1, y1)) in enumerate(arrows):
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2 if i < 4 else 0, arrowsize=1.1, arrowwidth=1.6,
                           arrowcolor="#5B6B7F")
    fig.add_annotation(x=0.69, y=0.155, text="↔", showarrow=False, font=dict(size=16, color="#5B6B7F"))
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1.02])
    return style(fig, "", height, legend=False).update_layout(margin=dict(l=10, r=10, t=10, b=10))
