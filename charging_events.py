"""
Charging-event detection.

A charging event is a continuous period in which Battery Charge Power > threshold.
A new event starts when charging stops or when the time between two consecutive
observations exceeds the configurable maximum gap (so a data outage never merges two
separate charging periods into one).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

_OPTIONAL_MEANS = {
    "avg_pv_kw": "pv_power",
    "avg_load_kw": "load_power",
    "avg_irradiance_wm2": "solar_irradiance",
    "avg_ambient_temp_c": "ambient_temperature",
    "avg_pv_cell_temp_c": "pv_cell_temperature",
    "avg_grid_kw": "grid_power",
}


def label_events(df: pd.DataFrame, max_gap_min: float = 90.0,
                 charge_min: float = 0.0) -> pd.Series:
    """Return an event-id series (NaN where not charging)."""
    if "battery_charge_power" not in df or df.empty:
        return pd.Series(np.nan, index=df.index)
    is_ch = df["battery_charge_power"].fillna(0) > charge_min
    gap = df["timestamp"].diff() > pd.Timedelta(minutes=max_gap_min)
    new = is_ch & (~is_ch.shift(1, fill_value=False) | gap)
    ids = new.cumsum().where(is_ch)
    return ids


def detect_events(df: pd.DataFrame, max_gap_min: float = 90.0, min_duration_min: float = 0.0,
                  charge_min: float = 0.0) -> Tuple[pd.DataFrame, pd.Series]:
    """Detect charging events. Returns (event table, per-row event id)."""
    ids = label_events(df, max_gap_min, charge_min)
    if ids.notna().sum() == 0:
        return pd.DataFrame(), ids
    d = df.assign(_eid=ids).dropna(subset=["_eid"])
    if "interval_h" not in d:
        d = d.assign(interval_h=0.0)
    rows = []
    for eid, g in d.groupby("_eid", sort=True):
        g = g.sort_values("timestamp")
        dur_h = (g["timestamp"].iloc[-1] - g["timestamp"].iloc[0]).total_seconds() / 3600 \
            + float(g["interval_h"].iloc[-1])
        soc = g["battery_soc"] if "battery_soc" in g else pd.Series(dtype=float)
        soc_valid = soc.dropna()
        row = dict(
            event_id=int(eid),
            start=g["timestamp"].iloc[0],
            end=g["timestamp"].iloc[-1],
            duration_h=round(dur_h, 3),
            n_obs=len(g),
            soc_start=float(soc_valid.iloc[0]) if len(soc_valid) else np.nan,
            soc_end=float(soc_valid.iloc[-1]) if len(soc_valid) else np.nan,
            avg_charge_kw=float(g["battery_charge_power"].mean()),
            peak_charge_kw=float(g["battery_charge_power"].max()),
            avg_rcn_h=float(g["normalized_charging_rate"].mean()) if "normalized_charging_rate" in g else np.nan,
            peak_rcn_h=float(g["normalized_charging_rate"].max()) if "normalized_charging_rate" in g else np.nan,
            energy_kwh=float(g["charge_energy"].sum()) if "charge_energy" in g else np.nan,
        )
        row["soc_increase"] = row["soc_end"] - row["soc_start"]
        row["soc_gain_per_h"] = row["soc_increase"] / dur_h if dur_h > 0 else np.nan
        for out, col in _OPTIONAL_MEANS.items():
            row[out] = float(g[col].mean()) if col in g and g[col].notna().any() else np.nan
        rows.append(row)
    ev = pd.DataFrame(rows)
    if min_duration_min > 0:
        keep = ev["duration_h"] * 60 >= min_duration_min
        dropped = set(ev.loc[~keep, "event_id"])
        ev = ev[keep].reset_index(drop=True)
        ids = ids.where(~ids.isin(dropped))
    return ev, ids


def event_summary(ev: pd.DataFrame) -> dict:
    if ev is None or ev.empty:
        return dict(n_events=0, avg_duration_h=np.nan, longest_h=np.nan,
                    avg_soc_increase=np.nan, total_energy_kwh=np.nan)
    return dict(n_events=int(len(ev)), avg_duration_h=float(ev["duration_h"].mean()),
                longest_h=float(ev["duration_h"].max()),
                avg_soc_increase=float(ev["soc_increase"].mean()),
                total_energy_kwh=float(ev["energy_kwh"].sum()))
