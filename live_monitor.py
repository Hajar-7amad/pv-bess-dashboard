"""Live Monitor (preview) – architecture for future real-time EMS monitoring.

The page polls ``DataSource.latest()`` on a fixed refresh interval inside an
``st.fragment(run_every=...)``. Today the source is the CSVDataSource, so the page either
shows the last historical record or *replays* the history record-by-record to demonstrate
the auto-refresh loop. Connecting ``LiveEMSDataSource`` (modules/data_sources.py) is the
only change needed for real-time operation. No control commands exist anywhere.
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd
import streamlit as st

from modules import state, ui
from modules.data_sources import CSVDataSource
from modules.plotting import energy_flow_diagram, time_series

ui.header("Live Monitor (preview)", "Auto-refresh architecture for future direct EMS connection · read-only.")
ui.require_data()
st.info("Current source: historical CSV files. **Replay mode** steps through the recorded data to demonstrate the "
        "auto-refresh loop. A live EMS connection plugs into `LiveEMSDataSource.latest()` without changing this page.",
        icon="📡")

s = state.settings()
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    key = ui.battery_picker(key="lv_bat")
mode = c2.radio("Mode", ["Latest record", "Replay history"], key="lv_mode")
run = c3.toggle("Auto-refresh", value=False, key="lv_run", help=f"Every {s['live_refresh_s']} s (Settings).")
d = state.get_battery_data(key)
df = d.df
source = CSVDataSource({})
st.session_state.setdefault("lv_pos", max(len(df) - 200, 0))


@st.fragment(run_every=timedelta(seconds=int(s["live_refresh_s"])) if run else None)
def live_panel():
    if mode == "Replay history":
        pos = st.session_state["lv_pos"]
        if run:
            pos = (pos + 1) % len(df)
            st.session_state["lv_pos"] = pos
        view = df.iloc[: pos + 1]
    else:
        view = df
    rec = source.latest(view)
    if rec is None:
        st.warning("No record available.")
        return
    v = {k: (float(rec[k]) if k in rec and pd.notna(rec[k]) else None) for k in rec.index
         if k not in ("timestamp", "battery", "operating_mode")}
    status = "Data OK" if v.get("battery_soc") is not None else "SOC missing"
    st.markdown(f"**{pd.Timestamp(rec['timestamp']):%Y-%m-%d %H:%M}** · active battery **{d.spec.name}** · "
                f"operating mode **{rec.get('operating_mode', 'Unknown')}** · system status: {status}")
    ui.kpi_grid([
        ("PV power", v.get("pv_power"), "kW"), ("Load", v.get("load_power"), "kW"),
        ("SOC", v.get("battery_soc"), "%"),
        ("Battery power", (v.get("battery_charge_power") or 0) - (v.get("battery_discharge_power") or 0)
         if v.get("battery_charge_power") is not None or v.get("battery_discharge_power") is not None else None,
         "kW", "+ charging / − discharging"),
        ("Grid", v.get("grid_power"), "kW"), ("R_cn", v.get("normalized_charging_rate"), "h⁻¹"),
        ("Irradiance", v.get("solar_irradiance"), "W/m²"), ("Mode", rec.get("operating_mode", "Unknown")),
    ], 4, d.spec.color)
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.plotly_chart(energy_flow_diagram(v, d.spec.name, d.spec.nominal_kwh, 380), width="stretch",
                        key=f"lv_flow_{rec['timestamp']}")
    with c2:
        tail = view.tail(48)
        cols = [c for c in ["pv_power", "load_power", "battery_soc"] if c in tail]
        st.plotly_chart(time_series(tail, cols, mode="panels", height=380), width="stretch",
                        key=f"lv_ts_{rec['timestamp']}")


live_panel()
st.caption("Monitoring, analytics, prediction and explainability only – the platform has no battery control path.")
