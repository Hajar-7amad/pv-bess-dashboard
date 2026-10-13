"""Data Quality page."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules import data_quality as dq
from modules import state, ui
from modules.plotting import SEQ_SCALE, style

ui.header("Data Quality", "Coverage, sampling, gaps, invalid values, outliers and sensor behavior – before cleaning.")
ui.require_data()
s = state.settings()

c1, _ = st.columns([1, 3])
with c1:
    key = ui.battery_picker(key="dq_bat")
d = state.get_battery_data(key)
spec, master = d.spec, d.master

q = d.quality
ui.kpi_grid([
    ("Raw observations", q.get("Raw observations (all files)")),
    ("Synchronized obs.", q.get("Synchronized observations")),
    ("Usable obs.", q.get("Usable observations (SOC & charge power present)")),
    ("Missing data", q.get("Missing-data percentage [%]"), "%"),
    ("Duplicate timestamps", q.get("Duplicate timestamps")),
    ("Timestamp coverage", q.get("Timestamp coverage [%]"), "%"),
    ("Median interval", q.get("Median sampling interval [min]"), "min"),
    ("Monitoring duration", q.get("Monitoring duration [days]"), "days"),
], 4, spec.color)

warns = dq.physical_warnings(master)
gaps = dq.timestamp_gaps(master["timestamp"], s["gap_factor"])
# sensor-behavior checks at the native resolution of each source file (before resampling)
_fz, _jp = [], []
for _name, _f in d.std_frames.items():
    _a = dq.frozen_sensors(_f, s["frozen_min_samples"])
    _b = dq.sudden_jumps(_f, s["jump_mad_k"])
    if len(_a):
        _fz.append(_a.assign(file=_name))
    if len(_b):
        _jp.append(_b.assign(file=_name))
frozen = pd.concat(_fz, ignore_index=True) if _fz else pd.DataFrame()
jumps = pd.concat(_jp, ignore_index=True) if _jp else pd.DataFrame()
inv = d.processed.invalid
if len(inv):
    for r in inv.to_dict("records"):
        warns.append(f"{ui.var_label(r['variable'])}: {r['invalid_values']} invalid values in {r.get('stage', '')} "
                     f"(range seen {r['min_seen']:.4g} … {r['max_seen']:.4g}; bounds {r['bounds']}).")
if len(gaps):
    warns.append(f"{len(gaps)} timestamp gaps longer than {s['gap_factor']:g}× the median interval "
                 f"(largest {gaps['gap_hours'].max():.1f} h).")
if len(frozen):
    warns.append("Possible frozen sensors: " + ", ".join(f"{ui.var_label(v)} (longest run {n} samples)" for v, n in
                                                         zip(frozen["variable"], frozen["longest_run_samples"])))
if len(jumps):
    warns.append("Sudden jumps: " + ", ".join(f"{ui.var_label(v)} ({n})" for v, n in zip(jumps["variable"], jumps["jumps"])))
with st.container(border=True):
    st.markdown("**Warnings**")
    if warns:
        for w in warns:
            st.markdown(f"- ⚠️ {w}")
    else:
        st.markdown("No warnings.")

tabs = st.tabs(["Statistics", "Missing data", "Sampling & gaps", "Invalid & outliers", "Sensor behavior",
                "Battery comparison"])

with tabs[0]:
    stats = dq.summary_stats(master)
    st.dataframe(stats, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4g") for c in
                                ["min", "max", "mean", "median", "std"]})
    ui.download_df(stats, f"{key}_quality_statistics.csv", key="dq_dl_stats")
    st.caption("Statistics of the synchronized data before cleaning. Availability = non-missing samples ÷ samples "
               "expected from the median interval over the monitoring span.")

with tabs[1]:
    stats = dq.summary_stats(master)
    if len(stats):
        fig = px.bar(stats.sort_values("missing_pct"), x="missing_pct", y="label", orientation="h",
                     color_discrete_sequence=[spec.color])
        st.plotly_chart(style(fig, "Missing values by parameter", max(260, 26 * len(stats) + 80), legend=False,
                              xlabel="Missing [%]", ylabel=""), width="stretch", key="dq_miss")
    freq = st.radio("Heatmap resolution", ["1D", "6h", "1h"], horizontal=True, key="dq_hm",
                    format_func={"1D": "Daily", "6h": "6-hourly", "1h": "Hourly"}.get)
    hm = dq.missing_heatmap_matrix(master, freq)
    if len(hm):
        fig = go.Figure(go.Heatmap(z=hm.values, x=hm.columns, y=[ui.var_label(v) for v in hm.index], zmin=0, zmax=100,
                                   colorscale=SEQ_SCALE, colorbar=dict(title="Available [%]"),
                                   hovertemplate="%{y}<br>%{x}<br>%{z:.0f} % available<extra></extra>"))
        ui.show_fig(style(fig, "Data availability (light = missing)", max(300, 26 * len(hm) + 120), legend=False),
                    "dq_hmfig", f"{key}_availability_heatmap")

with tabs[2]:
    rows = []
    for name, f in d.std_frames.items():
        sa = dq.sampling_analysis(f["timestamp"])
        rows.append(dict(file=name, rows=len(f), median_min=sa["median_s"] / 60, mode_min=sa["mode_s"] / 60,
                         min_min=sa["min_s"] / 60, max_min=sa["max_s"] / 60, distinct_intervals=sa["n_distinct"]))
    st.markdown("**Sampling interval per source file (before synchronization)**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    fname = st.selectbox("Interval distribution for", list(d.std_frames), key="dq_f")
    sa = dq.sampling_analysis(d.std_frames[fname]["timestamp"])
    if len(sa["intervals_min"]):
        iv = sa["intervals_min"].clip(upper=sa["intervals_min"].quantile(0.99) * 1.5)
        fig = px.histogram(iv, nbins=60, color_discrete_sequence=[spec.color])
        st.plotly_chart(style(fig, f"Sampling intervals – {fname}", 300, legend=False, xlabel="Interval [min]",
                              ylabel="Count"), width="stretch", key="dq_ivh")
    st.markdown(f"**Timestamp gaps in synchronized data** (> {s['gap_factor']:g}× median interval)")
    st.dataframe(gaps, hide_index=True, width="stretch")
    if len(gaps):
        ui.download_df(gaps, f"{key}_timestamp_gaps.csv", key="dq_dl_gaps")

with tabs[3]:
    st.markdown("**Invalid / physically impossible values** (handled per Settings: "
                f"`{s['invalid_policy']}`)")
    st.dataframe(inv if len(inv) else pd.DataFrame({"result": ["none found"]}), hide_index=True, width="stretch")
    st.markdown(f"**Outliers** (method `{s['outlier_method']}`, policy `{s['outlier_policy']}`)")
    out = d.processed.outliers
    st.dataframe(out if len(out) else pd.DataFrame({"result": ["none found"]}), hide_index=True, width="stretch")
    if len(out):
        v = st.selectbox("Inspect outliers of", out["variable"].tolist(), format_func=ui.var_label, key="dq_out_v")
        dfp = d.df
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=dfp["timestamp"], y=master.set_index("timestamp")[v].reindex(dfp["timestamp"]).values
                                   if v in master else dfp[v], mode="markers", name="All",
                                   marker=dict(size=4, color="#AAB4C0")))
        o = dfp[dfp["flag_outlier"]] if "flag_outlier" in dfp else dfp.iloc[0:0]
        mo = master.set_index("timestamp")[v].reindex(o["timestamp"]) if v in master else o[v]
        fig.add_trace(go.Scattergl(x=o["timestamp"], y=mo.values, mode="markers", name="Flagged (any variable)",
                                   marker=dict(size=6, color=spec.color, symbol="x")))
        ui.show_fig(style(fig, f"{ui.var_label(v)} – flagged rows", 360, ylabel=ui.var_label(v)), "dq_outfig",
                    f"{key}_outliers_{v}", export=False)
    st.markdown("**Preprocessing log**")
    st.dataframe(d.processed.log, hide_index=True, width="stretch")

with tabs[4]:
    st.markdown(f"**Frozen / constant sensors** (≥ {s['frozen_min_samples']} identical consecutive readings at the "
                "file's native resolution; zeros ignored for power, irradiance and wind)")
    st.caption("A constant SOC while the battery is idle is normal; check flagged runs against the operating mode.")
    st.dataframe(frozen if len(frozen) else pd.DataFrame({"result": ["none found"]}), hide_index=True, width="stretch")
    st.markdown(f"**Sudden jumps** (|Δ| > {s['jump_mad_k']:g} × robust spread of first differences)")
    st.dataframe(jumps if len(jumps) else pd.DataFrame({"result": ["none found"]}), hide_index=True, width="stretch")
    st.caption("Large legitimate steps (e.g. PV at sunrise under clouds, charge start/stop) can also be flagged; "
               "review before excluding data.")

with tabs[5]:
    st.caption("Unequal data availability can influence apparent performance differences between batteries.")
    allq = {}
    for b in state.batteries_with_data():
        bd = state.get_battery_data(b.key)
        allq[b.name] = bd.quality
    qt = pd.DataFrame(allq)
    st.dataframe(qt, width="stretch")
    ui.download_df(qt.reset_index(names="indicator"), "battery_data_quality_comparison.csv", key="dq_dl_cmp")
    ql = qt.T.reset_index(names="battery")
    colors = state.colors()
    c1, c2 = st.columns(2)
    for c, col in zip((c1, c2), ["Missing-data percentage [%]", "Charging observations (filtered)"]):
        if col in ql:
            fig = px.bar(ql, x="battery", y=col, color="battery", color_discrete_map=colors)
            c.plotly_chart(style(fig, col, 300, legend=False, xlabel=""), width="stretch", key=f"dq_c_{col}")
