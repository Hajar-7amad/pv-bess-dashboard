"""Data Explorer page: time-series analysis, free chart exploration, correlation."""
from __future__ import annotations

import streamlit as st

from modules import state, ui
from modules.explorer import color_options, data_controls
from modules.plotting import build_chart, correlation_matrix, time_series

ui.header("Data Explorer", "Select any period, parameters and aggregation; zoom, pan, hover and export every chart.")
ui.require_data()

df, ch = data_controls("dx")
vars_ = ui.numeric_vars(df)
colors = state.colors()

t1, t2, t3 = st.tabs(["Time-series analysis", "Chart explorer", "Correlation matrix"])

with t1:
    default = [v for v in ["battery_soc", "pv_power", "load_power", "solar_irradiance", "battery_charge_power"]
               if v in vars_]
    sel = st.multiselect("Parameters", vars_, default=default, format_func=ui.var_label, key="dx_ts_vars")
    c1, c2, c3 = st.columns([1.4, 1.4, 1])
    mode = c1.radio("Layout", ["panels", "shared", "secondary"], horizontal=True, key="dx_ts_mode",
                    format_func={"panels": "Stacked panels", "shared": "One axis",
                                 "secondary": "Secondary y-axis"}.get,
                    help="Stacked panels keep each unit on its own axis (recommended).")
    y2 = None
    if mode == "secondary":
        y2 = c2.selectbox("Secondary y-axis parameter", vars_, format_func=ui.var_label, key="dx_ts_y2")
    markers = c3.checkbox("Markers", False, key="dx_ts_mk")
    group = "battery" if len(ch["keys"]) > 1 else None
    if sel:
        fig = time_series(df, sel, mode=mode, secondary=y2, group=group, colors=colors, markers=markers,
                          title="EMS parameters over the selected period")
        ui.show_fig(fig, "dx_ts", "time_series")
    else:
        st.info("Select at least one parameter.")

with t2:
    c1, c2, c3, c4 = st.columns(4)
    chart = c1.selectbox("Chart type", ["Line chart", "Scatter plot", "Bar chart", "Histogram", "Box plot",
                                        "Heatmap", "Correlation matrix"], key="dx_chart")
    xopts = ["timestamp"] + vars_ + (["hour", "day_of_week", "month"] if "hour" not in vars_ else [])
    x = c2.selectbox("X-axis", xopts, format_func=lambda c: "Time" if c == "timestamp" else ui.var_label(c),
                     key="dx_x", index=0 if chart == "Line chart" else min(1, len(xopts) - 1))
    ys = c3.multiselect("Y-axis (one or more)", vars_, default=vars_[:1], format_func=ui.var_label, key="dx_y")
    color = c4.selectbox("Colour / group", color_options(df), key="dx_color",
                         format_func=lambda c: c if c in ("(none)", "battery", "operating_mode") else ui.var_label(c))
    y2 = None
    if chart == "Line chart" and x == "timestamp":
        y2 = st.selectbox("Optional secondary y-axis", ["(none)"] + vars_, format_func=lambda c: c if c == "(none)" else ui.var_label(c),
                          key="dx_y2")
        y2 = None if y2 == "(none)" else y2
    fig = build_chart(df, chart, x, ys, y2=y2, color=None if color == "(none)" else color, colors=colors,
                      title=f"{chart}: {', '.join(ui.var_label(y) for y in ys)}")
    ui.show_fig(fig, "dx_chart_fig", "explorer_chart")

with t3:
    sel = st.multiselect("Variables", vars_, default=[v for v in vars_ if v not in ("day", "month")][:12],
                         format_func=ui.var_label, key="dx_corr")
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True, key="dx_corr_m")
    ui.show_fig(correlation_matrix(df, sel, method=method), "dx_corr_fig", "correlation_matrix")
    st.caption("Correlation describes linear (Pearson) or monotonic (Spearman) association only.")
