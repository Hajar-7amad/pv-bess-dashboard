"""
Session state and the cached end-to-end processing pipeline.

Data organisation
-----------------
* Each battery has its own group of uploaded files (e.g. SOC.csv and the battery's
  charge/discharge export).
* A "shared" group holds plant-level files used by every battery (irradiance,
  temperatures, PV power, load, grid) because both banks sit on the same PV-BESS.
* Column mappings are stored per (group, file), so one combined file containing columns
  for both batteries can be mapped differently for each battery.

``get_battery_data(key)`` runs   read -> map -> synchronize -> quality -> preprocess ->
charging filter -> charging events -> KPIs   and memoises the result on a fingerprint
of the inputs, so pages never recompute (or retrain) unnecessarily.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from config import DEFAULT_BATTERIES, DEFAULT_SETTINGS, BatterySpec
from modules import battery_analysis as ba
from modules.charging_events import detect_events
from modules.data_loader import FileMapping, StandardizeReport, auto_mapping, standardize
from modules.data_quality import quality_score_table
from modules.data_sources import CSVDataSource
from modules.normalized_rate import charging_filter
from modules.preprocessing import ProcessedData, run_pipeline, screen_invalid
from modules.synchronization import synchronize

SHARED = "__shared__"

_PIPELINE_SETTINGS = [
    "soc_threshold", "charge_power_min_kw", "discharge_power_min_kw", "battery_power_sign",
    "sync_method", "resample_interval", "merge_tolerance_min", "duplicate_policy",
    "invalid_policy", "outlier_method", "outlier_iqr_k", "outlier_z", "outlier_policy",
    "event_max_gap_min", "event_min_duration_min", "energy_max_gap_min", "grid_sign",
]


@dataclass
class BatteryData:
    spec: BatterySpec
    files: List[str]
    raw_rows: int
    reports: List[StandardizeReport]
    std_frames: Dict[str, pd.DataFrame]
    sync_stats: pd.DataFrame
    sync_summary: dict
    master: pd.DataFrame
    processed: ProcessedData
    charging: pd.DataFrame
    charging_steps: pd.DataFrame
    events: pd.DataFrame
    kpis: dict
    quality: dict
    errors: List[str] = field(default_factory=list)

    @property
    def df(self) -> pd.DataFrame:
        return self.processed.df


# ---------------------------------------------------------------------------
# Initialisation & accessors
# ---------------------------------------------------------------------------
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("settings", copy.deepcopy(DEFAULT_SETTINGS))
    ss.setdefault("batteries", [b.to_dict() for b in DEFAULT_BATTERIES])
    ss.setdefault("uploads", {})        # group -> {file_name: bytes}
    ss.setdefault("mappings", {})       # "group::file" -> FileMapping
    ss.setdefault("pipeline_cache", {})  # battery key -> (fingerprint, BatteryData)
    ss.setdefault("models", {})         # battery key -> ModelResult
    ss.setdefault("model_features", {})  # battery key -> feature list
    ss.setdefault("shap", {})           # battery key -> ShapResult
    ss.setdefault("synthetic_demo", False)
    ss.setdefault("active_battery", ss["batteries"][0]["key"])


def settings() -> dict:
    return st.session_state["settings"]


def batteries() -> List[BatterySpec]:
    return [BatterySpec.from_dict(b) for b in st.session_state["batteries"]]


def battery(key: str) -> Optional[BatterySpec]:
    return next((b for b in batteries() if b.key == key), None)


def names() -> Dict[str, str]:
    return {b.key: b.name for b in batteries()}


def colors() -> Dict[str, str]:
    """Battery display name -> colour (colour follows the battery, not its rank)."""
    return {b.name: b.color for b in batteries()}


def group_files(group: str) -> Dict[str, bytes]:
    return st.session_state["uploads"].get(group, {})


def add_files(group: str, files: Dict[str, bytes]) -> None:
    up = st.session_state["uploads"].setdefault(group, {})
    for n, data in files.items():
        up[n] = data
        mkey = f"{group}::{n}"
        if mkey not in st.session_state["mappings"]:
            try:
                raw = CSVDataSource({n: data}).read_table(n).df
                st.session_state["mappings"][mkey] = auto_mapping(n, raw)
            except Exception:  # unreadable files are reported on the import page
                pass


def remove_file(group: str, name: str) -> None:
    st.session_state["uploads"].get(group, {}).pop(name, None)
    st.session_state["mappings"].pop(f"{group}::{name}", None)


def clear_all_data() -> None:
    for k in ("uploads", "mappings", "pipeline_cache", "models", "shap", "model_features"):
        st.session_state[k] = {}
    st.session_state["synthetic_demo"] = False


def batteries_with_data() -> List[BatterySpec]:
    return [b for b in batteries() if group_files(b.key)]


def has_data() -> bool:
    return bool(batteries_with_data())


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _fingerprint(spec: BatterySpec, s: dict) -> str:
    h = hashlib.sha1()
    for group in (spec.key, SHARED):
        for n, data in sorted(group_files(group).items()):
            h.update(n.encode())
            h.update(hashlib.sha1(data).digest())
            m = st.session_state["mappings"].get(f"{group}::{n}")
            if m:
                h.update(json.dumps([m.ts.__dict__, m.columns, m.scale], sort_keys=True, default=str).encode())
    h.update(json.dumps({k: s.get(k) for k in _PIPELINE_SETTINGS}, sort_keys=True).encode())
    h.update(f"{spec.nominal_kwh}|{spec.name}".encode())
    return h.hexdigest()


def process_battery(spec: BatterySpec, s: dict, files_by_group: Dict[str, Dict[str, bytes]],
                    mappings: Dict[str, FileMapping]) -> BatteryData:
    """Pure pipeline (no Streamlit calls) – also used by tests and scripts."""
    errors: List[str] = []
    std_frames, reports, raw_rows, file_list = {}, [], 0, []
    inv_tables: List[pd.DataFrame] = []
    for group in (spec.key, SHARED):
        files = files_by_group.get(group, {})
        if not files:
            continue
        src = CSVDataSource(files)
        for n in src.list_tables():
            label = n if group == spec.key else f"[shared] {n}"
            try:
                raw = src.read_table(n).df
                m = mappings.get(f"{group}::{n}") or auto_mapping(n, raw)
                if not any(v for v in m.columns.values()):
                    errors.append(f"{label}: no columns mapped – file skipped.")
                    continue
                std, rep = standardize(raw, m)
                rep.file_name = label
                std, inv, bad = screen_invalid(std, s["invalid_policy"])
                if s["invalid_policy"] == "drop_row":
                    std = std[~bad].reset_index(drop=True)
                if len(inv):
                    inv_tables.append(inv.assign(file=label))
                raw_rows += rep.rows_in
                reports.append(rep)
                std_frames[label] = std
                file_list.append(label)
            except Exception as e:  # report and continue with other files
                errors.append(f"{label}: {e}")

    master, sync_stats, sync_summary = synchronize(
        std_frames, method=s["sync_method"], interval=s["resample_interval"],
        tolerance_min=s["merge_tolerance_min"], duplicate_policy=s["duplicate_policy"])

    # Shared plant files may span several battery periods (sequential dispatch): keep only
    # the time range covered by this battery's own files.
    own = [f for n, f in std_frames.items() if not n.startswith("[shared]")]
    if own and not master.empty:
        t0 = min(f["timestamp"].min() for f in own)
        t1 = max(f["timestamp"].max() for f in own)
        n_before = len(master)
        master = master[(master["timestamp"] >= t0.floor(s["resample_interval"]) if s["sync_method"] == "resample"
                         else master["timestamp"] >= t0) & (master["timestamp"] <= t1)].reset_index(drop=True)
        sync_summary["clipped_to_battery_period"] = n_before - len(master)
        sync_summary["battery_period"] = (t0, t1)

    if master.empty:
        empty = pd.DataFrame(columns=["timestamp"])
        proc = ProcessedData(spec.key, empty, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        return BatteryData(spec, file_list, raw_rows, reports, std_frames, sync_stats, sync_summary,
                           empty, proc, empty, pd.DataFrame(), pd.DataFrame(), {}, {}, errors)

    pre_inv = pd.concat(inv_tables, ignore_index=True) if inv_tables else pd.DataFrame()
    proc = run_pipeline(master, spec, s, sync_summary, pre_inv)
    events, ids = detect_events(proc.df, s["event_max_gap_min"], s["event_min_duration_min"],
                                s["charge_power_min_kw"])
    proc.df["charging_event_id"] = ids
    if len(events):
        events.insert(0, "battery", spec.name)
    charging, steps = charging_filter(proc.df, s["soc_threshold"], s["charge_power_min_kw"])
    kpis = ba.battery_kpis(proc.df, spec, charging, events, s)
    dup_total = sum(int(r.get("duplicate_timestamps", 0)) for r in sync_stats.to_dict("records")) \
        if len(sync_stats) else 0
    quality = quality_score_table(raw_rows, master, proc.df, charging, dup_total)
    return BatteryData(spec, file_list, raw_rows, reports, std_frames, sync_stats, sync_summary,
                       master, proc, charging, steps, events, kpis, quality, errors)


def get_battery_data(key: str) -> Optional[BatteryData]:
    spec = battery(key)
    if spec is None or not group_files(key):
        return None
    s = settings()
    fp = _fingerprint(spec, s)
    cache = st.session_state["pipeline_cache"]
    if key in cache and cache[key][0] == fp:
        return cache[key][1]
    with st.spinner(f"Processing {spec.name} data…"):
        files = {spec.key: group_files(spec.key), SHARED: group_files(SHARED)}
        data = process_battery(spec, s, files, st.session_state["mappings"])
    cache[key] = (fp, data)
    return data


def all_battery_data() -> Dict[str, BatteryData]:
    out = {}
    for b in batteries_with_data():
        d = get_battery_data(b.key)
        if d is not None and not d.df.empty:
            out[b.key] = d
    return out


def invalidate_models(key: Optional[str] = None) -> None:
    for store in ("models", "shap"):
        if key:
            st.session_state[store].pop(key, None)
        else:
            st.session_state[store] = {}
