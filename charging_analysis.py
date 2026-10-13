"""Capacity-Normalized Charging Analysis page (R_cn, charging filter, charging events)."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import OPERATING_MODES
from modules import state, ui
from modules.normalized_rate import rcn_summary, select_modes
from modules.plotting import style, time_series

ui.header("Capacity-Normalized Charging Analysis",
          "R_cn = P_charge / E_nominal · charging-observation selection · charging events")
ui.require_data()

s = state.settings()
c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
with c1:
    key = ui.battery_picker(key="ca_bat")
with c2:
    thr = st.number_input("SOC charging cutoff [%]", 1.0, 100.0, float(s["soc_threshold"]), 1.0, key="ca_thr",
                          help="Paper default: 80 %. Applies to all pages; models are retrained on request.")
    if thr != s["soc_threshold"]:
        s["soc_threshold"] = thr
        state.invalidate_models()
        st.rerun()
d = state.get_battery_data(key)
spec = d.spec
with c3:
    st.latex(r"R_{cn} = \frac{P_{charge}\,[\mathrm{kW}]}{E_{nominal}\,[\mathrm{kWh}]}\quad[\mathrm{h^{-1}}]")
    st.caption(f"E_nominal = {spec.nominal_kwh:,.0f} kWh for {spec.name} (change in Settings). "
               "Column: `normalized_charging_rate`.")

st.markdown("#### Observations remaining after each step")
left, right = st.columns([1.3, 1])
with left:
    st.markdown("**Preprocessing (nothing removed silently)**")
    log = d.processed.log
    st.dataframe(log, width="stretch", hide_index=True)
with right:
    st.markdown("**Charging-observation filter (paper rule)**")
    steps = d.charging_steps
    st.dataframe(steps, width="stretch", hide_index=True)
    if len(steps):
        fig = go.Figure(go.Funnel(y=steps["step"], x=steps["remaining"], marker_color=spec.color,
                                  textinfo="value+percent initial"))
        st.plotly_chart(style(fig, "", 300, legend=False).update_layout(margin=dict(l=10, r=10, t=10, b=10)),
                        width="stretch", key="ca_funnel")

st.markdown("#### Operating-state selection")
sel = st.radio("Show", ["Charging only (paper filter)", "Charging only (all SOC)", "Discharging only", "Standby",
                        "All operating states"], horizontal=True, key="ca_sel")
df = d.df
if sel.startswith("Charging only (paper"):
    view = d.charging
elif sel.startswith("Charging only (all"):
    view = select_modes(df, ["Charging"])
elif sel == "Discharging only":
    view = select_modes(df, ["Discharging"])
elif sel == "Standby":
    view = select_modes(df, ["Standby"])
else:
    view = df
counts = df["operating_mode"].value_counts()
ui.kpi_grid([("Selected observations", len(view))] +
            [(f"{m} observations", int(counts.get(m, 0))) for m in OPERATING_MODES] +
            [("Paper-filtered charging", len(d.charging), "", f"SOC < {s['soc_threshold']:g} %")], 5, spec.color)

if "normalized_charging_rate" not in df:
    st.warning("Battery charge power is not mapped, so R_cn cannot be computed.")
    st.stop()

st.markdown("#### Capacity-normalized charging rate")
summ = rcn_summary(d.charging["normalized_charging_rate"]) if len(d.charging) else rcn_summary(df["normalized_charging_rate"].iloc[0:0])
ui.kpi_grid([("n (filtered charging)", summ["n"]), ("Mean R_cn", summ["mean"], "h⁻¹"),
             ("Median R_cn", summ["median"], "h⁻¹"), ("Std. dev.", summ["std"], "h⁻¹"),
             ("95th percentile", summ["p95"], "h⁻¹"), ("Maximum R_cn", summ["max"], "h⁻¹"),
             ("Equivalent C-rate at max", summ["max"], "C", "R_cn in h⁻¹ equals the C-rate on an energy basis"),
             ("Nominal capacity", spec.nominal_kwh, "kWh")], 4, spec.color)

c1, c2 = st.columns(2)
with c1:
    fig = px.histogram(view, x="normalized_charging_rate", nbins=50, color_discrete_sequence=[spec.color])
    ui.show_fig(style(fig, "R_cn distribution (selection)", 340, legend=False, xlabel="R_cn [h⁻¹]", ylabel="Count"),
                "ca_hist", f"{key}_rcn_hist", export=False)
with c2:
    if "battery_soc" in view:
        fig = px.scatter(view, x="battery_soc", y="normalized_charging_rate", opacity=0.6,
                         color_discrete_sequence=[spec.color], render_mode="webgl")
        fig.add_vline(x=s["soc_threshold"], line=dict(dash="dash", color="#6B7785"),
                      annotation_text=f"cutoff {s['soc_threshold']:g} %")
        ui.show_fig(style(fig, "R_cn vs SOC (selection)", 340, legend=False, xlabel="Battery SOC [%]",
                          ylabel="R_cn [h⁻¹]"), "ca_soc", f"{key}_rcn_vs_soc", export=False)
cols = [c for c in ["normalized_charging_rate", "battery_charge_power", "battery_soc"] if c in view]
ui.show_fig(time_series(view.sort_values("timestamp"), cols, mode="panels", markers=True,
                        title="R_cn, charge power and SOC over time (selection)"), "ca_ts", f"{key}_rcn_timeseries")
c1, c2 = st.columns(2)
c1.download_button("Download filtered charging dataset (CSV)", d.charging.to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"{key}_filtered_charging.csv", key="ca_dl_ch")
c2.download_button("Download current selection (CSV)", view.to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"{key}_selection.csv", key="ca_dl_sel")

st.markdown("#### Charging events")
st.caption(f"Event = continuous period with P_charge > {s['charge_power_min_kw']:g} kW; a gap longer than "
           f"{s['event_max_gap_min']:g} min starts a new event (Settings). Events use all SOC levels.")
ev = d.events
if ev is None or ev.empty:
    st.info("No charging events detected.")
else:
    ui.kpi_grid([("Events", len(ev)), ("Avg duration", ev["duration_h"].mean(), "h"),
                 ("Longest event", ev["duration_h"].max(), "h"), ("Avg SOC increase", ev["soc_increase"].mean(), "%-pts"),
                 ("Avg peak R_cn", ev["peak_rcn_h"].mean(), "h⁻¹"), ("Total charge energy", ev["energy_kwh"].sum(), "kWh")],
                6, spec.color)
    st.dataframe(ev, width="stretch", hide_index=True,
                 column_config={"start": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                                "end": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")})
    ui.download_df(ev, f"{key}_charging_events.csv", "Download charging-event table (CSV)", key="ca_dl_ev")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(ev, x="duration_h", nbins=30, color_discrete_sequence=[spec.color])
        st.plotly_chart(style(fig, "Event duration distribution", 300, legend=False, xlabel="Duration [h]",
                              ylabel="Events"), width="stretch", key="ca_evh")
    with c2:
        fig = px.scatter(ev, x="soc_start", y="soc_increase", size="energy_kwh", color_discrete_sequence=[spec.color],
                         hover_data=["event_id", "start", "duration_h"])
        st.plotly_chart(style(fig, "SOC increase vs starting SOC (size = energy)", 300, legend=False,
                              xlabel="Starting SOC [%]", ylabel="SOC increase [%-pts]"), width="stretch", key="ca_evs")
    eid = st.selectbox("Inspect event", ev["event_id"].tolist(), key="ca_eid",
                       format_func=lambda i: f"#{i} · {ev.loc[ev.event_id == i, 'start'].iloc[0]:%Y-%m-%d %H:%M} · "
                                             f"{ev.loc[ev.event_id == i, 'duration_h'].iloc[0]:.1f} h")
    one = df[df["charging_event_id"] == eid]
    cols = [c for c in ["battery_charge_power", "normalized_charging_rate", "battery_soc", "pv_power", "load_power"]
            if c in one]
    ui.show_fig(time_series(one, cols, mode="panels", markers=True, title=f"Charging event #{eid}"), "ca_ev_ts",
                f"{key}_event_{eid}", export=False)
