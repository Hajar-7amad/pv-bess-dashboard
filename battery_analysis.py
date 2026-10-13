"""Battery Analysis page (one battery at a time)."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules import state, ui
from modules import battery_analysis as ba
from modules.plotting import build_chart, style, time_series

ui.header("Battery Analysis", "Operating behavior, charging rate, throughput and utilization of one battery system.")
ui.require_data()

c1, c2 = st.columns([1, 3])
with c1:
    key = ui.battery_picker(key="ba_bat")
d = state.get_battery_data(key)
spec, s = d.spec, state.settings()
start, end = ui.time_window(d.df, "ba")
df = ui.clip_time(d.df, start, end)
charging = ui.clip_time(d.charging, start, end)
events = d.events[(d.events["start"] >= start) & (d.events["start"] <= end)] if len(d.events) else d.events
k = ba.battery_kpis(df, spec, charging, events, s)

st.markdown(f"#### {spec.name} · {spec.technology} · {spec.nominal_kwh:,.0f} kWh")
ui.kpi_from_dict(k, [
    "Peak charging power [kW]", "Peak discharge power [kW]", "Average R_cn (filtered charging) [h⁻¹]",
    "Maximum R_cn (filtered charging) [h⁻¹]", "Minimum SOC [%]", "Maximum SOC [%]",
    "SOC operating range P5–P95 [%-pts]", "Throughput / E_nom (equivalent full cycles)",
    "Charging duration [h]", "Discharging duration [h]", "Standby duration [h]",
    "Operating hours (charge + discharge) [h]", "Total charging energy [kWh]", "Total discharge energy [kWh]",
    "Daily throughput [kWh/day]", "Charging events",
], 4, spec.color)

tabs = st.tabs(["Time series", "Energy throughput", "Charging relationships", "Utilization"])

with tabs[0]:
    cols = [c for c in ["battery_soc", "battery_charge_power", "battery_discharge_power",
                        "normalized_charging_rate", "normalized_discharge_rate"] if c in df]
    ui.show_fig(time_series(df, cols, mode="panels", title=f"{spec.name} – SOC, power and capacity-normalized rates"),
                "ba_ts", f"{key}_timeseries")
    ui.note_unavailable(["battery_soc", "battery_charge_power", "battery_discharge_power"], df)

with tabs[1]:
    daily = ba.period_summary(df, "1D", s["grid_sign"])
    if len(daily) and {"charge_energy_kwh", "discharge_energy_kwh"} & set(daily.columns):
        fig = go.Figure()
        if "charge_energy_kwh" in daily:
            fig.add_bar(x=daily["period"], y=daily["charge_energy_kwh"], name="Charge energy", marker_color=spec.color)
        if "discharge_energy_kwh" in daily:
            fig.add_bar(x=daily["period"], y=-daily["discharge_energy_kwh"], name="Discharge energy (shown negative)",
                        marker_color="#8796A8")
        fig.update_layout(barmode="relative")
        ui.show_fig(style(fig, "Daily charge / discharge energy", 380, ylabel="Energy [kWh]"), "ba_daily", f"{key}_daily_energy")
        cum = df[["timestamp"]].copy()
        if "charge_energy" in df:
            cum["Cumulative charge energy"] = df["charge_energy"].fillna(0).cumsum()
        if "discharge_energy" in df:
            cum["Cumulative discharge energy"] = df["discharge_energy"].fillna(0).cumsum()
        fig2 = px.line(cum, x="timestamp", y=[c for c in cum.columns if c != "timestamp"],
                       color_discrete_sequence=[spec.color, "#8796A8"])
        ui.show_fig(style(fig2, "Cumulative energy throughput", 340, ylabel="Energy [kWh]", xlabel=""), "ba_cum",
                    f"{key}_cumulative")
        with st.expander("Daily summary table"):
            st.dataframe(daily, width="stretch", hide_index=True)
            ui.download_df(daily, f"{key}_daily_summary.csv")
    else:
        st.info("Charge/discharge power not available – energy throughput cannot be computed.")

with tabs[2]:
    src = st.radio("Observations", ["Filtered charging data (paper rule)", "All processed data"], horizontal=True,
                   key="ba_rel_src")
    data = charging if src.startswith("Filtered") else df
    pairs = [("pv_power", "battery_charge_power"), ("load_power", "battery_charge_power"),
             ("battery_soc", "battery_charge_power"), ("battery_soc", "normalized_charging_rate"),
             ("solar_irradiance", "normalized_charging_rate"), ("ambient_temperature", "normalized_charging_rate"),
             ("pv_cell_temperature", "normalized_charging_rate"), ("grid_power", "battery_charge_power")]
    avail = [(x, y) for x, y in pairs if x in data and y in data and data[x].notna().any()]
    miss = [x for x, y in pairs if (x, y) not in avail]
    color_by = st.selectbox("Colour points by", ["(none)", "hour", "battery_soc", "month"], key="ba_rel_c",
                            format_func=lambda c: c if c == "(none)" else ui.var_label(c))
    for i in range(0, len(avail), 2):
        cols = st.columns(2)
        for c, (x, y) in zip(cols, avail[i:i + 2]):
            with c:
                fig = build_chart(data, "Scatter plot", x, [y], color=None if color_by == "(none)" else color_by,
                                  title=f"{ui.var_label(y).split(' [')[0]} vs {ui.var_label(x).split(' [')[0]}",
                                  height=360)
                if color_by == "(none)":
                    fig.update_traces(marker_color=spec.color)
                ui.show_fig(fig, f"ba_sc_{x}_{y}", f"{key}_{y}_vs_{x}", export=False)
    if miss:
        st.caption("Not shown (variable unavailable): " + ", ".join(sorted(set(ui.var_label(m) for m in miss))))

with tabs[3]:
    c1, c2 = st.columns(2)
    ms = ba.mode_shares(df)
    if len(ms):
        fig = px.bar(ms, x="mode", y="pct", text=ms["pct"].round(1).astype(str) + " %",
                     color_discrete_sequence=[spec.color])
        c1.plotly_chart(style(fig, "Share of time per operating state", 320, legend=False, ylabel="Time [%]", xlabel=""),
                        width="stretch", key="ba_ms")
    sb = ba.soc_band_shares(df)
    if len(sb):
        fig = px.bar(sb, x="band", y="pct", text=sb["pct"].round(1).astype(str) + " %",
                     color_discrete_sequence=[spec.color])
        c2.plotly_chart(style(fig, "Share of time per SOC band", 320, legend=False, ylabel="Time [%]", xlabel="SOC band"),
                        width="stretch", key="ba_sb")
    if "battery_soc" in df:
        fig = px.histogram(df, x="battery_soc", nbins=50, color_discrete_sequence=[spec.color])
        st.plotly_chart(style(fig, "SOC distribution", 300, legend=False, xlabel="Battery SOC [%]", ylabel="Count"),
                        width="stretch", key="ba_soc_hist")
