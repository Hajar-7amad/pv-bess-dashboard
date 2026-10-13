"""Battery Performance Comparison page."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules import battery_analysis as ba
from modules import battery_comparison as bc
from modules import state, ui
from modules.plotting import (actual_vs_predicted, bar_compare, break_gaps, grouped_box, grouped_ecdf, grouped_hist,
                              grouped_scatter, style)

ui.header("Battery Performance Comparison",
          "Raw · capacity-normalized · matched-operating-condition · model-performance comparison")
ui.require_data()

specs = state.batteries_with_data()
if len(specs) < 2:
    st.info("Load data for at least two batteries to compare them (Data Import).")
    st.stop()

nm = state.names()
colors = state.colors()
s = state.settings()
keys_all = [b.key for b in specs]
c1, c2, c3 = st.columns([1, 1, 2])
ka = c1.selectbox("Battery A (reference)", keys_all, index=0, format_func=nm.get, key="bc_a")
kb = c2.selectbox("Battery B", [k for k in keys_all if k != ka], index=0, format_func=nm.get, key="bc_b")
extra = c3.multiselect("Additional batteries (plots only)", [k for k in keys_all if k not in (ka, kb)],
                       format_func=nm.get, key="bc_extra")
sel = [ka, kb] + extra

st.info("**Fair-comparison notice.** " + bc.DISCLAIMER + " The banks were dispatched sequentially, so side-by-side "
        "averages are *not* a controlled experiment. Use *Matched conditions* for a like-for-like subset.", icon="ℹ️")

# ------------------------------------------------------------------------------- filters
data = {k: state.get_battery_data(k) for k in sel}
union = pd.concat([data[k].df for k in sel], ignore_index=True)
with st.expander("Comparison filters", expanded=False):
    f1, f2, f3, f4 = st.columns(4)
    d0, d1 = union["timestamp"].min().date(), union["timestamp"].max().date()
    dr = f1.date_input("Date range", (d0, d1), min_value=d0, max_value=d1, key="bc_dr")
    hr = f2.slider("Hour of day", 0, 23, (0, 23), key="bc_hr")
    dows = f3.multiselect("Day of week", list(range(7)), default=list(range(7)), key="bc_dow",
                          format_func=lambda i: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i])
    charging_only = f4.checkbox("Charging observations only (paper rule) for all statistics", False, key="bc_conly")
    ranges = {}
    rcols = st.columns(3)
    for i, (col, lbl) in enumerate([("battery_soc", "SOC [%]"), ("pv_power", "PV power [kW]"),
                                     ("load_power", "Load [kW]"), ("solar_irradiance", "Irradiance [W/m²]"),
                                     ("ambient_temperature", "Ambient temperature [°C]"),
                                     ("pv_cell_temperature", "PV-cell temperature [°C]")]):
        if col in union and union[col].notna().any():
            lo, hi = float(np.floor(union[col].min())), float(np.ceil(union[col].max()))
            if hi > lo:
                ranges[col] = rcols[i % 3].slider(lbl, lo, hi, (lo, hi), key=f"bc_r_{col}")
    ev_sel = {}
    for k in sel:
        ev = data[k].events
        if len(ev):
            ev_sel[k] = st.multiselect(f"Specific charging events – {nm[k]} (empty = all)", ev["event_id"].tolist(),
                                       key=f"bc_ev_{k}")


def _apply(df: pd.DataFrame, k: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(dr, tuple) and len(dr) == 2:
        df = df[(df["timestamp"].dt.date >= dr[0]) & (df["timestamp"].dt.date <= dr[1])]
    df = df[df["timestamp"].dt.hour.between(*hr) & df["timestamp"].dt.dayofweek.isin(dows)]
    for col, (lo, hi) in ranges.items():
        if col in df and (lo, hi) != (float(np.floor(union[col].min())), float(np.ceil(union[col].max()))):
            df = df[df[col].between(lo, hi)]
    if ev_sel.get(k):
        df = df[df["charging_event_id"].isin(ev_sel[k])] if "charging_event_id" in df else df
    return df


F, CH, EV, K = {}, {}, {}, {}
for k in sel:
    d = data[k]
    base = d.charging if charging_only else d.df
    F[k] = _apply(base, k)
    CH[k] = _apply(d.charging, k)
    ev = d.events
    if len(ev):
        keep_ids = set(F[k]["charging_event_id"].dropna().unique()) if "charging_event_id" in F[k] else set()
        ev = ev[ev["event_id"].isin(keep_ids)]
    EV[k] = ev
    K[k] = ba.battery_kpis(F[k], d.spec, CH[k], EV[k], s)
    if k in st.session_state["models"]:
        m = st.session_state["models"][k].metrics
        K[k].update({"Model test R²": m["test_r2"], "Model test MAE [h⁻¹]": m["test_mae"],
                     "Model test RMSE [h⁻¹]": m["test_rmse"], "Cross-validation R² (mean)": m["cv_r2_mean"]})
filtered_note = ", ".join(f"{nm[k]}: {len(F[k]):,} obs / {len(CH[k]):,} charging" for k in sel)
st.caption("After filters – " + filtered_note)

long = bc.stack(F, nm)
long_ch = bc.stack(CH, nm)

tabs = st.tabs(["KPI cards", "Scorecard", "Normalized", "Time series", "Distributions", "Scatter",
                "Matched conditions", "Daily / weekly", "Energy balance", "Utilization", "Charging events",
                "ML comparison", "Interpretation"])

# ------------------------------------------------------------------------------- KPI cards
with tabs[0]:
    groups = {
        "Dataset": ["Nominal capacity [kWh]", "Observations", "Charging observations (filtered)", "Monitoring start",
                    "Monitoring end", "Days with data"],
        "State of charge": ["Average SOC [%]", "Minimum SOC [%]", "Maximum SOC [%]", "SOC operating range P5–P95 [%-pts]"],
        "Power": ["Average charging power [kW]", "Peak charging power [kW]", "Average discharge power [kW]",
                  "Peak discharge power [kW]"],
        "Capacity-normalized": ["Average R_cn (filtered charging) [h⁻¹]", "Median R_cn (filtered charging) [h⁻¹]",
                                "Maximum R_cn (filtered charging) [h⁻¹]", "Throughput / E_nom (equivalent full cycles)"],
        "Energy & time": ["Total charging energy [kWh]", "Total discharge energy [kWh]", "Charging duration [h]",
                          "Discharging duration [h]", "Standby duration [h]"],
        "Conditions during charging": ["Avg PV power during charging [kW]", "Avg load during charging [kW]",
                                       "Avg irradiance during charging [W/m²]", "Avg ambient temp. during charging [°C]",
                                       "Avg PV-cell temp. during charging [°C]"],
    }
    for g, keys in groups.items():
        st.markdown(f"**{g}**")
        cols = st.columns(len(sel))
        for c, k in zip(cols, sel):
            with c:
                spec = data[k].spec
                st.markdown(f"<span style='color:{spec.color}'>■</span> **{spec.name}**", unsafe_allow_html=True)
                ui.kpi_from_dict(K[k], keys, ncols=2 if len(keys) > 3 else len(keys), color=spec.color)

# ------------------------------------------------------------------------------- scorecard
with tabs[1]:
    st.caption("Separate engineering metrics; no overall winner is computed unless you explicitly enable and "
               "define it below. Relative difference = (B − A) / A × 100.")
    card = bc.scorecard(K, ka, kb, nm)
    st.dataframe(card, width="stretch", hide_index=True,
                 column_config={nm[ka]: st.column_config.NumberColumn(format="%.4g"),
                                nm[kb]: st.column_config.NumberColumn(format="%.4g"),
                                card.columns[3]: st.column_config.NumberColumn(format="%.4g"),
                                "Relative difference [%]": st.column_config.NumberColumn(format="%.1f")})
    ui.download_df(card, "battery_comparison_scorecard.csv", "Download comparison table (CSV)", key="bc_dl_card")
    if not any(k in st.session_state["models"] for k in (ka, kb)):
        st.caption("Model metrics appear after training models on the Machine Learning page.")
    full = bc.scorecard(K, ka, kb, nm, metrics=[m for m in K[ka] if not isinstance(K[ka][m], pd.Timestamp)])
    with st.expander("All indicators"):
        st.dataframe(full, width="stretch", hide_index=True)
        ui.download_df(full, "battery_comparison_all_indicators.csv", key="bc_dl_full")
    if s.get("show_winner"):
        st.markdown("**Opt-in preference tally** – you define which direction is preferable for each metric.")
        prefs_df = pd.DataFrame({"Metric": card["Metric"], "Preferred direction": "ignore"})
        ed = st.data_editor(prefs_df, hide_index=True, key="bc_prefs",
                            column_config={"Preferred direction": st.column_config.SelectboxColumn(
                                options=["ignore", "higher", "lower"])})
        tally = bc.preference_tally(card, dict(zip(ed["Metric"], ed["Preferred direction"])), nm[ka], nm[kb])
        if len(tally):
            st.dataframe(tally, hide_index=True)
            st.caption("This tally only reflects the preferences you defined; it is not a scientific ranking.")
    else:
        st.caption("A preference tally can be enabled in Settings ('Allow user-defined preference tally').")

# ------------------------------------------------------------------------------- normalized
with tabs[2]:
    st.info(bc.NORMALIZATION_NOTE, icon="ℹ️")
    rows = []
    for k in sel:
        kk, cap = K[k], data[k].spec.nominal_kwh
        rows += [
            dict(battery=nm[k], metric="Mean R_cn (charge power / E_nom)", value=kk["Average R_cn (filtered charging) [h⁻¹]"]),
            dict(battery=nm[k], metric="Median R_cn", value=kk["Median R_cn (filtered charging) [h⁻¹]"]),
            dict(battery=nm[k], metric="Max R_cn", value=kk["Maximum R_cn (filtered charging) [h⁻¹]"]),
            dict(battery=nm[k], metric="Mean discharge power / E_nom", value=kk["Average discharge power [kW]"] / cap),
            dict(battery=nm[k], metric="Peak discharge power / E_nom", value=kk["Peak discharge power / E_nom [h⁻¹]"]),
        ]
    nd = pd.DataFrame(rows)
    ui.show_fig(bar_compare(nd, "metric", "value", "battery", colors, "Capacity-normalized power indicators [h⁻¹]",
                            ytitle="kW per kWh of nominal capacity [h⁻¹]"), "bc_norm1", "normalized_power")
    rows = []
    for k in sel:
        kk = K[k]
        rows += [dict(battery=nm[k], metric="Equivalent full cycles (throughput / E_nom)",
                      value=kk["Throughput / E_nom (equivalent full cycles)"]),
                 dict(battery=nm[k], metric="Daily throughput / E_nom [cycles/day]",
                      value=kk["Daily throughput / E_nom [cycles/day]"]),
                 dict(battery=nm[k], metric="Charging time per SOC increase [h/%-pt]",
                      value=kk.get("Charging time per SOC increase [h / %-pt]", np.nan))]
    nd2 = pd.DataFrame(rows)
    c1, c2 = st.columns(2)
    with c1:
        ui.show_fig(bar_compare(nd2[nd2.metric != "Charging time per SOC increase [h/%-pt]"], "metric", "value", "battery",
                                colors, "Normalized energy throughput", ytitle="Cycles"), "bc_norm2", "normalized_throughput")
    with c2:
        ui.show_fig(bar_compare(nd2[nd2.metric == "Charging time per SOC increase [h/%-pt]"], "metric", "value", "battery",
                                colors, "Charging time relative to SOC increase", ytitle="h per %-pt SOC"),
                    "bc_norm3", "charging_time_per_soc")
    st.dataframe(pd.concat([nd, nd2]).pivot(index="metric", columns="battery", values="value"), width="stretch")

# ------------------------------------------------------------------------------- time series
with tabs[3]:
    align = st.radio("Time axis", ["Calendar time", "Elapsed days since each battery's first record"], horizontal=True,
                     key="bc_align", help="The banks may have operated in different periods; elapsed-time alignment "
                                          "overlays them for shape comparison only.")
    tsd = long.copy()
    x = "timestamp"
    if align.startswith("Elapsed"):
        tsd["elapsed_days"] = tsd.groupby("battery")["timestamp"].transform(lambda t: (t - t.min()).dt.total_seconds() / 86400)
        x = "elapsed_days"
    for col, ttl in [("battery_soc", "SOC over time"), ("battery_charge_power", "Charging power"),
                     ("normalized_charging_rate", "Capacity-normalized charging rate"),
                     ("battery_discharge_power", "Discharge power")]:
        if col in tsd:
            fig = go.Figure()
            for b in tsd["battery"].unique():
                g = tsd[tsd["battery"] == b]
                g = break_gaps(g, "timestamp") if x == "timestamp" else g
                fig.add_trace(go.Scattergl(x=g[x], y=g[col], mode="lines", name=b, line=dict(color=colors.get(b), width=1.4)))
            fig = style(fig, ttl, 320, xlabel="Elapsed days" if x != "timestamp" else "", ylabel=ui.var_label(col))
            ui.show_fig(fig, f"bc_ts_{col}", f"compare_{col}", export=False)
    if len(long_ch):
        st.markdown("**Plant conditions during charging events**")
        c1, c2 = st.columns(2)
        for c, col in zip((c1, c2), ("pv_power", "load_power")):
            if col in long_ch:
                with c:
                    ui.show_fig(grouped_hist(long_ch, col, colors, f"{ui.var_label(col)} during charging"),
                                f"bc_cond_{col}", f"charging_{col}", export=False)

# ------------------------------------------------------------------------------- distributions
with tabs[4]:
    src = st.radio("Data", ["All (filtered) observations", "Charging observations (paper rule)"], horizontal=True,
                   key="bc_dist_src")
    L = long if src.startswith("All") else long_ch
    st.markdown("**Box plots** (box = IQR, line = median, dashed = mean)")
    bvars = [v for v in ["battery_soc", "battery_charge_power", "normalized_charging_rate", "battery_discharge_power",
                         "pv_power", "load_power", "solar_irradiance", "ambient_temperature", "pv_cell_temperature"]
             if v in L and L[v].notna().any()]
    for i in range(0, len(bvars), 3):
        cols = st.columns(3)
        for c, v in zip(cols, bvars[i:i + 3]):
            with c:
                ui.show_fig(grouped_box(L, v, colors, height=320), f"bc_box_{v}_{src[:3]}", f"box_{v}", export=False)
    st.markdown("**Histograms (density) and cumulative distributions**")
    daily = []
    for k in sel:
        p = ba.period_summary(F[k], "1D", s["grid_sign"])
        if len(p) and "throughput_kwh" in p:
            daily.append(p.assign(battery=nm[k]))
    daily = pd.concat(daily, ignore_index=True) if daily else pd.DataFrame()
    dvars = [("battery_soc", L), ("battery_charge_power", L), ("normalized_charging_rate", L)]
    if len(daily):
        dvars.append(("throughput_kwh", daily))
    norm = st.selectbox("Histogram normalisation", ["probability density", "probability", ""], key="bc_hn",
                        format_func=lambda x: x or "count")
    for v, D in dvars:
        if v not in D:
            continue
        c1, c2 = st.columns(2)
        with c1:
            fig = grouped_hist(D, v, colors, norm=norm)
            if v == "throughput_kwh":
                fig.update_xaxes(title_text="Daily energy throughput [kWh]")
                fig.update_layout(title_text="Daily energy throughput")
            ui.show_fig(fig, f"bc_h_{v}", f"hist_{v}", export=False)
        with c2:
            fig = grouped_ecdf(D, v, colors)
            if v == "throughput_kwh":
                fig.update_xaxes(title_text="Daily energy throughput [kWh]")
                fig.update_layout(title_text="Cumulative distribution – daily energy throughput")
            ui.show_fig(fig, f"bc_e_{v}", f"ecdf_{v}", export=False)
    for v in ["solar_irradiance", "ambient_temperature"]:
        if v in L:
            ui.show_fig(grouped_hist(L, v, colors, norm=norm), f"bc_h2_{v}", f"hist_{v}", export=False)

# ------------------------------------------------------------------------------- scatter
with tabs[5]:
    st.caption("Charging observations (paper rule); colour = battery technology.")
    for i, xv in enumerate(["battery_soc", "pv_power", "load_power", "solar_irradiance", "ambient_temperature",
                            "pv_cell_temperature"]):
        if xv in long_ch and long_ch[xv].notna().any() and "normalized_charging_rate" in long_ch:
            ui.show_fig(grouped_scatter(long_ch, xv, "normalized_charging_rate", colors), f"bc_sc_{xv}",
                        f"scatter_rcn_vs_{xv}", export=False)

# ------------------------------------------------------------------------------- matched
with tabs[6]:
    st.markdown("**Compare Similar Operating Conditions**")
    st.caption("Keeps only charging observations of each battery that have at least one observation of the other "
               "battery within *all* tolerance bands. This reduces – but does not eliminate – differences in "
               "operating conditions; it is still observational data, not a controlled experiment.")
    enable = st.toggle("Compare Similar Operating Conditions", value=True, key="bc_match_on")
    if enable:
        cols = st.columns(6)
        spec_t = [("battery_soc", "SOC ± [%]", "match_soc_tol"), ("pv_power", "PV power ± [kW]", "match_pv_tol"),
                  ("load_power", "Load ± [kW]", "match_load_tol"), ("solar_irradiance", "Irradiance ± [W/m²]", "match_irr_tol"),
                  ("ambient_temperature", "Temperature ± [°C]", "match_temp_tol"), ("hour", "Hour of day ± [h]", "match_hour_tol")]
        tol = {}
        for c, (col, lbl, sk) in zip(cols, spec_t):
            use = c.checkbox(lbl.split(" ±")[0], value=col in ("battery_soc", "pv_power", "load_power"), key=f"bc_mu_{col}")
            v = c.number_input(lbl, 0.0, 1e4, float(s[sk]), key=f"bc_mt_{col}")
            if use:
                tol[col] = v
        ma, mb, info = bc.matched_conditions(CH[ka], CH[kb], tol)
        if info.get("error"):
            st.warning(info["error"])
        else:
            ui.kpi_grid([
                (f"{nm[ka]} matched", info.get("n_a_matched", 0), "", f"of {info.get('n_a_complete', 0)} complete charging obs."),
                (f"{nm[kb]} matched", info.get("n_b_matched", 0), "", f"of {info.get('n_b_complete', 0)} complete charging obs."),
                (f"{nm[ka]} mean R_cn (matched)", ma["normalized_charging_rate"].mean() if len(ma) else None, "h⁻¹"),
                (f"{nm[kb]} mean R_cn (matched)", mb["normalized_charging_rate"].mean() if len(mb) else None, "h⁻¹"),
                (f"{nm[ka]} median R_cn (matched)", ma["normalized_charging_rate"].median() if len(ma) else None, "h⁻¹"),
                (f"{nm[kb]} median R_cn (matched)", mb["normalized_charging_rate"].median() if len(mb) else None, "h⁻¹"),
            ], 6)
            if len(ma) < 10 or len(mb) < 10:
                st.warning("Very few matched observations – matched statistics are not reliable. Widen the tolerances "
                           "or use fewer matching variables.")
            if len(ma) and len(mb):
                mlong = bc.stack({ka: ma, kb: mb}, nm)
                c1, c2 = st.columns(2)
                with c1:
                    ui.show_fig(grouped_hist(mlong, "normalized_charging_rate", colors, "Matched R_cn distribution"),
                                "bc_m_h", "matched_rcn_hist", export=False)
                with c2:
                    ui.show_fig(grouped_box(mlong, "normalized_charging_rate", colors, "Matched R_cn"), "bc_m_b",
                                "matched_rcn_box", export=False)
                env = [c for c in ["battery_soc", "pv_power", "load_power", "solar_irradiance", "ambient_temperature",
                                   "pv_cell_temperature", "hour"] if c in mlong]
                st.markdown("**Operating / environmental conditions of the matched subsets (mean)**")
                st.dataframe(mlong.groupby("battery")[env + ["normalized_charging_rate"]].agg(["mean", "median", "count"]),
                             width="stretch")
                ui.download_df(mlong, "matched_condition_subsets.csv", key="bc_dl_m")

# ------------------------------------------------------------------------------- daily / weekly
with tabs[7]:
    per = st.radio("Period", ["Day", "Week", "Month"], horizontal=True, key="bc_per")
    rule = {"Day": "1D", "Week": "W", "Month": "MS"}[per]
    P = []
    for k in sel:
        p = ba.period_summary(F[k], rule, s["grid_sign"])
        if len(p):
            P.append(p.assign(battery=nm[k]))
    P = pd.concat(P, ignore_index=True) if P else pd.DataFrame()
    metrics = [("charge_energy_kwh", "Charging energy [kWh]"), ("discharge_energy_kwh", "Discharge energy [kWh]"),
               ("avg_soc_pct", "Average SOC [%]"), ("avg_rcn_charging_h", "Average R_cn while charging [h⁻¹]"),
               ("peak_charge_kw", "Peak charging power [kW]"), ("pv_energy_kwh", "PV generation [kWh]"),
               ("load_energy_kwh", "Load demand [kWh]")]
    for i in range(0, len(metrics), 2):
        cols = st.columns(2)
        for c, (m, lbl) in zip(cols, metrics[i:i + 2]):
            if m in P:
                with c:
                    fig = px.line(P, x="period", y=m, color="battery", markers=True, color_discrete_map=colors)
                    ui.show_fig(style(fig, f"{per}ly {lbl.split(' [')[0].lower()}", 320, ylabel=lbl, xlabel=""),
                                f"bc_p_{m}", f"{per.lower()}_{m}", export=False)
    if len(P):
        ui.download_df(P, f"{per.lower()}_comparison.csv", key="bc_dl_p")

# ------------------------------------------------------------------------------- energy balance
with tabs[8]:
    st.caption("Energy integrated from power over each sampling interval during each battery's own monitoring "
               "window; values marked unavailable where the measurement is missing.")
    eb = pd.DataFrame({nm[k]: ba.energy_balance(F[k], s["grid_sign"]) for k in sel})
    st.dataframe(eb, width="stretch")
    ui.download_df(eb.reset_index(names="quantity"), "energy_balance.csv", key="bc_dl_eb")
    ebl = eb.reset_index(names="quantity").melt(id_vars="quantity", var_name="battery", value_name="kWh").dropna()
    if len(ebl):
        ui.show_fig(bar_compare(ebl, "quantity", "kWh", "battery", colors, "Energy balance per battery period",
                                ytitle="Energy [kWh]"), "bc_eb", "energy_balance")
        days = {nm[k]: max(F[k]["timestamp"].dt.normalize().nunique(), 1) for k in sel}
        ebd = ebl.assign(kwh_day=ebl.apply(lambda r: r["kWh"] / days[r["battery"]], axis=1))
        ui.show_fig(bar_compare(ebd, "quantity", "kwh_day", "battery", colors,
                                "Energy balance per day with data (removes period-length effect)", ytitle="kWh/day"),
                    "bc_ebd", "energy_balance_per_day")
    for k in sel:
        st.markdown(f"<span style='color:{data[k].spec.color}'>■</span> **{nm[k]} energy-flow summary:** " +
                    " · ".join(f"{q.split(' [')[0]} {ui.fmt(v, unit='kWh')}" for q, v in
                               ba.energy_balance(F[k], s["grid_sign"]).items()), unsafe_allow_html=True)

# ------------------------------------------------------------------------------- utilization
with tabs[9]:
    rows, rows2 = [], []
    for k in sel:
        sb = ba.soc_band_shares(F[k])
        if len(sb):
            rows.append(sb.assign(battery=nm[k]))
        ms = ba.mode_shares(F[k])
        if len(ms):
            rows2.append(ms.assign(battery=nm[k]))
    c1, c2 = st.columns(2)
    if rows:
        with c1:
            ui.show_fig(bar_compare(pd.concat(rows), "band", "pct", "battery", colors, "Time in SOC bands",
                                    ytitle="Time [%]"), "bc_sb", "soc_bands", export=False)
    if rows2:
        with c2:
            ui.show_fig(bar_compare(pd.concat(rows2), "mode", "pct", "battery", colors, "Time per operating state",
                                    ytitle="Time [%]"), "bc_ms", "operating_states", export=False)
    if "battery_soc" in long:
        ui.show_fig(grouped_hist(long, "battery_soc", colors, "SOC histogram", norm="probability"), "bc_soch",
                    "soc_histogram", export=False)
    ut = pd.DataFrame({nm[k]: {
        "SOC range min–max [%-pts]": K[k]["Maximum SOC [%]"] - K[k]["Minimum SOC [%]"],
        "SOC range P5–P95 [%-pts]": K[k]["SOC operating range P5–P95 [%-pts]"],
        "Charging events": K[k].get("Charging events", 0),
        "Avg charging-event duration [h]": K[k].get("Avg charging-event duration [h]", np.nan),
        "Longest charging event [h]": K[k].get("Longest charging event [h]", np.nan),
        "Avg SOC increase per event [%-pts]": K[k].get("Avg SOC increase per event [%-pts]", np.nan),
    } for k in sel})
    st.dataframe(ut, width="stretch")

# ------------------------------------------------------------------------------- charging events
with tabs[10]:
    evl = pd.concat([EV[k] for k in sel if len(EV[k])], ignore_index=True) if any(len(EV[k]) for k in sel) else pd.DataFrame()
    if evl.empty:
        st.info("No charging events in the filtered data.")
    else:
        summ = evl.groupby("battery").agg(events=("event_id", "count"), mean_duration_h=("duration_h", "mean"),
                                          median_duration_h=("duration_h", "median"), longest_h=("duration_h", "max"),
                                          mean_soc_increase=("soc_increase", "mean"), mean_peak_rcn=("peak_rcn_h", "mean"),
                                          mean_avg_rcn=("avg_rcn_h", "mean"), total_energy_kwh=("energy_kwh", "sum"))
        st.dataframe(summ, width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            ui.show_fig(grouped_hist(evl, "duration_h", colors, "Charging-event duration distribution", norm="probability")
                        .update_xaxes(title_text="Duration [h]"), "bc_evd", "event_duration", export=False)
        with c2:
            ui.show_fig(grouped_box(evl, "avg_rcn_h", colors, "Event-average R_cn").update_yaxes(title_text="R_cn [h⁻¹]"),
                        "bc_evr", "event_rcn", export=False)
        ui.show_fig(grouped_scatter(evl, "soc_start", "soc_increase", colors, "SOC increase vs starting SOC per event")
                    .update_xaxes(title_text="Starting SOC [%]").update_yaxes(title_text="SOC increase [%-pts]"),
                    "bc_evs", "event_soc", export=False)
        st.dataframe(evl, width="stretch", hide_index=True)
        ui.download_df(evl, "charging_events_all_batteries.csv", "Download charging-event table (CSV)", key="bc_dl_ev")

# ------------------------------------------------------------------------------- ML comparison
with tabs[11]:
    st.info(bc.ML_NOTE, icon="ℹ️")
    models = {k: st.session_state["models"][k] for k in sel if k in st.session_state["models"]}
    if len(models) < 2:
        st.warning("Train models for both batteries on the **Machine Learning** page to compare model performance.")
        st.page_link("pages/machine_learning.py", label="Go to Machine Learning", icon="🧠")
    else:
        mt = pd.DataFrame([{"battery": r.battery_name, "Test R²": r.metrics["test_r2"],
                            "Test MAE [h⁻¹]": r.metrics["test_mae"], "Test RMSE [h⁻¹]": r.metrics["test_rmse"],
                            "CV R² (mean)": r.metrics["cv_r2_mean"], "CV R² (std)": r.metrics["cv_r2_std"],
                            "Test observations": r.meta["n_test"], "Split": r.meta["split"]} for r in models.values()])
        st.dataframe(mt, hide_index=True, width="stretch")
        ml = mt.melt(id_vars="battery", value_vars=["Test R²", "CV R² (mean)"], var_name="metric")
        me = mt.melt(id_vars="battery", value_vars=["Test MAE [h⁻¹]", "Test RMSE [h⁻¹]"], var_name="metric")
        c1, c2 = st.columns(2)
        with c1:
            ui.show_fig(bar_compare(ml, "metric", "value", "battery", colors, "R² (test and cross-validation)",
                                    ytitle="R²"), "bc_ml_r2", "ml_r2", export=False)
        with c2:
            ui.show_fig(bar_compare(me, "metric", "value", "battery", colors, "Errors (test)", ytitle="h⁻¹"),
                        "bc_ml_err", "ml_errors", export=False)
        cols = st.columns(len(models))
        for c, r in zip(cols, models.values()):
            with c:
                ui.show_fig(actual_vs_predicted(r.predictions, f"Actual vs predicted – {r.battery_name}",
                                                colors.get(r.battery_name)), f"bc_avp_{r.battery_key}",
                            f"avp_{r.battery_key}", export=False)
        res = pd.concat([r.predictions[r.predictions["set"] == "test"] for r in models.values()])
        c1, c2 = st.columns(2)
        with c1:
            ui.show_fig(grouped_hist(res, "residual", colors, "Residual distribution (test)", norm="probability density")
                        .update_xaxes(title_text="Residual [h⁻¹]"), "bc_ml_res", "ml_residuals", export=False)
        with c2:
            res = res.assign(abs_error=res["residual"].abs())
            ui.show_fig(grouped_ecdf(res, "abs_error", colors, "Absolute prediction error (test) – cumulative")
                        .update_xaxes(title_text="|Error| [h⁻¹]"), "bc_ml_ae", "ml_abs_error", export=False)

# ------------------------------------------------------------------------------- interpretation
with tabs[12]:
    st.markdown("**Automatically generated descriptive summary**")
    for line in bc.interpretation(K, nm, ka, kb, CH):
        st.markdown(f"- {line}")
    st.warning("**Disclaimer.** " + bc.DISCLAIMER)
    st.caption("This summary is descriptive. It does not claim universal chemistry superiority, causality, "
               "degradation, efficiency or lifetime differences, which are not directly measured here.")
