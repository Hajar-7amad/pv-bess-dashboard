"""Shared data-selection and filter controls for the Data Explorer and Plot Builder."""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from config import AGGREGATION_INTERVALS, OPERATING_MODES
from modules import state, ui

STAGES = {
    "Processed data": "processed",
    "Filtered charging data (paper rule)": "charging",
    "Synchronized raw data (before cleaning)": "master",
}


def stage_frame(key: str, stage: str) -> pd.DataFrame:
    d = state.get_battery_data(key)
    if d is None:
        return pd.DataFrame()
    if stage == "charging":
        f = d.charging
    elif stage == "master":
        f = d.master.assign(battery=d.spec.name)
    else:
        f = d.df
    return f


def data_controls(prefix: str, show_soc: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """Battery / stage / period / mode / aggregation controls. Returns (data, choices)."""
    specs = state.batteries_with_data()
    c1, c2, c3, c4 = st.columns([1.6, 1.4, 1.2, 1.2])
    keys = c1.multiselect("Battery", [b.key for b in specs], default=[specs[0].key],
                          format_func=lambda k: state.names()[k], key=f"{prefix}_bats")
    stage_lbl = c2.selectbox("Data stage", list(STAGES), key=f"{prefix}_stage",
                             help="Processed = cleaned and derived; Filtered charging = P_charge > 0 and SOC "
                                  "below the cutoff; Raw = synchronized data before cleaning.")
    stage = STAGES[stage_lbl]
    modes = c3.multiselect("Operating mode", ["All"] + OPERATING_MODES, default=["All"], key=f"{prefix}_modes",
                           disabled=stage != "processed")
    agg_lbl = c4.selectbox("Aggregation", list(AGGREGATION_INTERVALS), key=f"{prefix}_agg",
                           help="Powers, temperatures, SOC: mean · interval energies: sum · cumulative "
                                "energy registers: last value · wind direction: circular mean.")
    frames = [stage_frame(k, stage) for k in keys]
    df = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame()
    if df.empty:
        st.info("No data for the selected battery/stage.")
        st.stop()
    start, end = ui.time_window(df, prefix)
    df = ui.clip_time(df, start, end)
    if stage == "processed" and modes and "All" not in modes:
        df = df[df["operating_mode"].isin(modes)]
    if show_soc and "battery_soc" in df:
        lo, hi = st.slider("SOC range [%]", 0.0, 100.0, (0.0, 100.0), 1.0, key=f"{prefix}_soc")
        if (lo, hi) != (0.0, 100.0):
            df = df[df["battery_soc"].between(lo, hi)]
    rule = AGGREGATION_INTERVALS[agg_lbl]
    if rule:
        df = ui.aggregate(df, rule)
    st.caption(f"{len(df):,} observations selected · stage: {stage_lbl} · aggregation: {agg_lbl}")
    return df, dict(keys=keys, stage=stage, rule=rule, start=start, end=end)


def color_options(df: pd.DataFrame) -> List[str]:
    opts = ["(none)"]
    if "battery" in df and df["battery"].nunique() > 0:
        opts.append("battery")
    if "operating_mode" in df:
        opts.append("operating_mode")
    opts += ui.numeric_vars(df)
    return opts
