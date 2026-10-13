"""System Overview page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import SYSTEM_INFO
from modules import state, ui
from modules.battery_analysis import latest_snapshot
from modules.plotting import energy_flow_diagram, time_series

ui.header(
    "System Overview",
    f"{SYSTEM_INFO['pv_array_kwp']:.0f} kWp PV · {SYSTEM_INFO['dc_bus']} · "
    f"{SYSTEM_INFO['ems']} EMS · {len(state.batteries())} battery systems configured",
)

if not state.has_data():
    st.markdown("#### Configured system")
    cols = st.columns(len(state.batteries()) + 1)
    cols[0].markdown(
        ui.kpi_card(
            "PV array",
            SYSTEM_INFO["pv_array_kwp"],
            "kWp",
            SYSTEM_INFO["ems"],
        ),
        unsafe_allow_html=True,
    )
    for c, b in zip(cols[1:], state.batteries()):
        c.markdown(
            ui.kpi_card(
                b.name,
                b.nominal_kwh,
                "kWh",
                b.technology,
                b.color,
            ),
            unsafe_allow_html=True,
        )
    ui.require_data()

c1, c2 = st.columns([1, 2])

with c1:
    key = ui.battery_picker("Selected battery", key="ov_bat")

d = state.get_battery_data(key)
spec, df = d.spec, d.df

with c2:
    mode = st.radio(
        "Instantaneous values",
        ["Latest record", "Choose a timestamp"],
        horizontal=True,
        key="ov_mode",
    )

if df.empty:
    st.warning("No processed observations for this battery – check the mapping on Data Import.")
    st.stop()

if mode == "Latest record":
    snap = latest_snapshot(df)
else:
    ts_opts = df["timestamp"].tolist()
    selected_ts = st.selectbox(
        "Timestamp",
        ts_opts,
        index=len(ts_opts) - 1,
        key="ov_timestamp",
    )
    snap = df.loc[df["timestamp"] == selected_ts].iloc[-1]

# Instantaneous snapshot
snap_items = []
for col, label, unit in [
    ("battery_soc", "Battery SOC", "%"),
    ("battery_charge_power", "Charge power", "kW"),
    ("battery_discharge_power", "Discharge power", "kW"),
    ("pv_power", "PV power", "kW"),
    ("load_power", "Load power", "kW"),
    ("grid_power", "Grid power", "kW"),
]:
    if col in snap.index and pd.notna(snap[col]):
        snap_items.append((label, snap[col], unit))

if snap_items:
    ui.kpi_grid(snap_items, 3)

k = d.kpis

ui.kpi_grid([
    ("Selected battery", spec.name, "", spec.technology),
    ("Nominal capacity", spec.nominal_kwh, "kWh"),
    ("Records", len(df)),
    ("Charging observations", len(d.charging), "", f"P>0 & SOC<{state.settings()['soc_threshold']:g}%"),
    ("Period start", k.get("Monitoring start")),
    ("Period end", k.get("Monitoring end")),
    ("Average SOC", k.get("Average SOC [%]"), "%"),
    ("Average R_cn", k.get("Average R_cn (filtered charging) [h⁻¹]"), "h⁻¹"),
    ("Peak PV generation", df["pv_power"].max() if "pv_power" in df else None, "kW"),
    ("Peak charging power", k.get("Peak charging power [kW]"), "kW"),
    ("Peak discharge power", k.get("Peak discharge power [kW]"), "kW"),
    ("Peak load", df["load_power"].max() if "load_power" in df else None, "kW"),
], 3)

st.markdown("#### Recent operation")
days = st.slider("Days shown", 1, 31, 7, key="ov_days")
recent = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(days=days)]
cols = [
    c for c in [
        "pv_power",
        "load_power",
        "battery_charge_power",
        "battery_discharge_power",
        "battery_soc",
    ]
    if c in recent and recent[c].notna().any()
]
if cols:
    ui.show_fig(
        time_series(
            recent,
            cols,
            mode="panels",
            title=f"{spec.name} – last {days} days",
        ),
        "ov_ts_fig",
        "overview_recent",
    )

with st.expander("System description (from the paper)"):
    st.markdown(
        f"""
* **PV array:** {SYSTEM_INFO['pv_array_kwp']:.0f} kWp, {SYSTEM_INFO['dc_bus']}
* **EMS:** {SYSTEM_INFO['ems']} · **Load:** {SYSTEM_INFO['load']}
* **Dispatch:** {SYSTEM_INFO['dispatch']}
"""
        + "\n".join(
            f"* **{b.name}:** {b.technology}, {b.chemistry}, {b.nominal_kwh:,.0f} kWh"
            + (f" ({b.manufacturer})" if b.manufacturer else "")
            for b in state.batteries()
        )
    )
