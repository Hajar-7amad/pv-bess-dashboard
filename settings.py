"""Settings page – battery registry and every analysis parameter."""
from __future__ import annotations

import copy
import json
import re

import streamlit as st

from config import (BATTERY_COLORS, DEFAULT_BATTERIES, DEFAULT_SETTINGS, EXAMPLE_SETTINGS_FILE,
                    FUTURE_BATTERY_TEMPLATES, BatterySpec)
from modules import state, ui
from modules.export import settings_json

ui.header("Settings", "Battery systems, charging filter, synchronization, preprocessing, ML and comparison parameters.")
s = state.settings()
before = copy.deepcopy(s)

# ----------------------------------------------------------------------------- batteries
st.markdown("#### Battery systems")
bats = st.session_state["batteries"]
for i, b in enumerate(bats):
    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns([1.3, 1.8, 1.6, 1.1, 0.6])
        b["name"] = c1.text_input("Name", b["name"], key=f"st_bn_{b['key']}")
        b["technology"] = c2.text_input("Technology", b["technology"], key=f"st_bt_{b['key']}")
        b["chemistry"] = c3.text_input("Chemistry", b["chemistry"], key=f"st_bc_{b['key']}")
        cap = c4.number_input("Nominal capacity [kWh]", 1.0, 1e6, float(b["nominal_kwh"]), 1.0, key=f"st_bk_{b['key']}",
                              help="E_nominal in R_cn = P_charge / E_nominal")
        if cap != b["nominal_kwh"]:
            b["nominal_kwh"] = cap
            state.invalidate_models(b["key"])
        b["color"] = c5.color_picker("Colour", b["color"], key=f"st_bcol_{b['key']}")
        if not b.get("builtin") and st.button("Remove battery", key=f"st_brm_{b['key']}"):
            bats.pop(i)
            st.session_state["uploads"].pop(b["key"], None)
            state.invalidate_models(b["key"])
            st.rerun()

with st.expander("Add a battery technology (e.g. Vanadium Redox Flow Battery)"):
    c1, c2, c3 = st.columns(3)
    tmpl = c1.selectbox("Template", list(FUTURE_BATTERY_TEMPLATES), key="st_new_t")
    name = c2.text_input("Name", tmpl if tmpl != "Other" else "New battery", key="st_new_n")
    cap = c3.number_input("Nominal capacity [kWh] (user-defined)", 1.0, 1e6, 500.0, key="st_new_c")
    if st.button("Add battery", key="st_new_add"):
        key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "battery"
        while any(x["key"] == key for x in bats):
            key += "_2"
        spec = BatterySpec(key=key, name=name, nominal_kwh=cap,
                           color=BATTERY_COLORS[len(bats) % len(BATTERY_COLORS)],
                           **FUTURE_BATTERY_TEMPLATES[tmpl])
        bats.append(spec.to_dict())
        st.success(f"Added {name}. Upload its files on the Data Import page; it gets its own model.")
        st.rerun()

# ----------------------------------------------------------------------------- parameters
t = st.tabs(["Charging filter", "Synchronization", "Preprocessing", "Charging events", "Machine learning",
             "Comparison", "Data quality & display"])
with t[0]:
    c1, c2, c3 = st.columns(3)
    s["soc_threshold"] = c1.number_input("SOC charging cutoff [%]", 1.0, 100.0, float(s["soc_threshold"]), 1.0,
                                         help="Paper: SOC < 80 %.")
    s["charge_power_min_kw"] = c2.number_input("Charging if P_charge > [kW]", 0.0, 1000.0, float(s["charge_power_min_kw"]),
                                               0.5, help="Paper: > 0 kW. A small dead-band can suppress sensor noise.")
    s["discharge_power_min_kw"] = c3.number_input("Discharging if P_discharge > [kW]", 0.0, 1000.0,
                                                  float(s["discharge_power_min_kw"]), 0.5)
    c1, c2 = st.columns(2)
    s["battery_power_sign"] = c1.selectbox("Signed battery-power convention", ["positive_is_discharge", "positive_is_charge"],
                                           index=["positive_is_discharge", "positive_is_charge"].index(s["battery_power_sign"]))
    s["grid_sign"] = c2.selectbox("Grid-power convention", ["positive_is_import", "positive_is_export"],
                                  index=["positive_is_import", "positive_is_export"].index(s["grid_sign"]))
with t[1]:
    c1, c2, c3 = st.columns(3)
    s["sync_method"] = c1.selectbox("Time alignment", ["resample", "nearest", "exact"],
                                    index=["resample", "nearest", "exact"].index(s["sync_method"]),
                                    format_func={"resample": "Resampling to common interval", "nearest": "Nearest timestamp",
                                                 "exact": "Exact timestamp"}.get)
    s["resample_interval"] = c2.selectbox("Resampling interval", ["5min", "10min", "15min", "30min", "1h", "1D"],
                                          index=["5min", "10min", "15min", "30min", "1h", "1D"].index(s["resample_interval"]),
                                          help="Paper analysis used hourly observations.")
    s["merge_tolerance_min"] = c3.number_input("Nearest-timestamp tolerance [min]", 0.0, 1440.0,
                                               float(s["merge_tolerance_min"]), 1.0)
with t[2]:
    c1, c2, c3 = st.columns(3)
    s["duplicate_policy"] = c1.selectbox("Duplicate timestamps", ["keep_first", "mean"],
                                         index=["keep_first", "mean"].index(s["duplicate_policy"]))
    s["invalid_policy"] = c2.selectbox("Invalid values", ["set_nan", "drop_row", "keep"],
                                       index=["set_nan", "drop_row", "keep"].index(s["invalid_policy"]),
                                       format_func={"set_nan": "Set to missing (reported)", "drop_row": "Remove row (reported)",
                                                    "keep": "Keep (flag only)"}.get)
    s["outlier_policy"] = c3.selectbox("Outliers", ["retain", "remove"], index=["retain", "remove"].index(s["outlier_policy"]),
                                       format_func={"retain": "Retain (flagged)", "remove": "Remove (set to missing, reported)"}.get)
    c1, c2, c3 = st.columns(3)
    s["outlier_method"] = c1.selectbox("Outlier method", ["iqr", "zscore", "none"],
                                       index=["iqr", "zscore", "none"].index(s["outlier_method"]),
                                       format_func={"iqr": "IQR fences", "zscore": "Robust z-score (MAD)", "none": "None"}.get)
    s["outlier_iqr_k"] = c2.number_input("IQR multiplier k", 0.5, 20.0, float(s["outlier_iqr_k"]), 0.5)
    s["outlier_z"] = c3.number_input("Robust z threshold", 1.0, 20.0, float(s["outlier_z"]), 0.5)
with t[3]:
    c1, c2, c3 = st.columns(3)
    s["event_max_gap_min"] = c1.number_input("Max gap inside an event [min]", 1.0, 1440.0, float(s["event_max_gap_min"]), 5.0)
    s["event_min_duration_min"] = c2.number_input("Minimum event duration [min]", 0.0, 1440.0,
                                                  float(s["event_min_duration_min"]), 5.0)
    s["energy_max_gap_min"] = c3.number_input("Max interval integrated for energy [min]", 1.0, 1440.0,
                                              float(s["energy_max_gap_min"]), 5.0,
                                              help="Longer gaps are not integrated as if the battery ran through them.")
with t[4]:
    c1, c2, c3, c4 = st.columns(4)
    s["test_size"] = c1.slider("Test fraction", 0.1, 0.4, float(s["test_size"]), 0.05)
    s["split_method"] = c2.selectbox("Split", ["chronological", "random"], index=["chronological", "random"].index(s["split_method"]))
    s["cv_folds"] = int(c3.number_input("CV folds", 2, 20, int(s["cv_folds"])))
    s["random_seed"] = int(c4.number_input("Random seed", 0, 10**6, int(s["random_seed"])))
    c1, c2, c3, c4 = st.columns(4)
    s["cv_method"] = c1.selectbox("CV method", ["kfold_shuffle", "timeseries"], index=["kfold_shuffle", "timeseries"].index(s["cv_method"]))
    s["cv_scope"] = c2.selectbox("CV data", ["full", "train"], index=["full", "train"].index(s.get("cv_scope", "full")))
    s["shap_max_samples"] = int(c3.number_input("Max SHAP samples", 100, 100000, int(s["shap_max_samples"]), 100))
    st.markdown("**XGBoost hyperparameters**")
    c1, c2, c3, c4 = st.columns(4)
    s["xgb_n_estimators"] = int(c1.number_input("n_estimators", 10, 5000, int(s["xgb_n_estimators"]), 10))
    s["xgb_max_depth"] = int(c2.number_input("max_depth", 1, 16, int(s["xgb_max_depth"])))
    s["xgb_learning_rate"] = c3.number_input("learning_rate", 0.001, 1.0, float(s["xgb_learning_rate"]), 0.01, format="%.3f")
    s["xgb_min_child_weight"] = c4.number_input("min_child_weight", 0.0, 100.0, float(s["xgb_min_child_weight"]), 0.5)
    c1, c2, c3, c4 = st.columns(4)
    s["xgb_subsample"] = c1.slider("subsample", 0.3, 1.0, float(s["xgb_subsample"]), 0.05)
    s["xgb_colsample_bytree"] = c2.slider("colsample_bytree", 0.3, 1.0, float(s["xgb_colsample_bytree"]), 0.05)
    s["xgb_reg_lambda"] = c3.number_input("reg_lambda", 0.0, 100.0, float(s["xgb_reg_lambda"]), 0.5)
    s["xgb_reg_alpha"] = c4.number_input("reg_alpha", 0.0, 100.0, float(s["xgb_reg_alpha"]), 0.1)
with t[5]:
    c = st.columns(6)
    s["match_soc_tol"] = c[0].number_input("SOC ± [%]", 0.0, 100.0, float(s["match_soc_tol"]))
    s["match_pv_tol"] = c[1].number_input("PV ± [kW]", 0.0, 1000.0, float(s["match_pv_tol"]))
    s["match_load_tol"] = c[2].number_input("Load ± [kW]", 0.0, 1000.0, float(s["match_load_tol"]))
    s["match_irr_tol"] = c[3].number_input("Irradiance ± [W/m²]", 0.0, 1500.0, float(s["match_irr_tol"]))
    s["match_temp_tol"] = c[4].number_input("Temperature ± [°C]", 0.0, 50.0, float(s["match_temp_tol"]))
    s["match_hour_tol"] = c[5].number_input("Hour ± [h]", 0.0, 12.0, float(s["match_hour_tol"]))
    s["show_winner"] = st.checkbox("Allow user-defined preference tally on the scorecard", bool(s["show_winner"]),
                                   help="Off by default: the scorecard shows separate metrics without declaring a winner.")
with t[6]:
    c1, c2, c3 = st.columns(3)
    s["gap_factor"] = c1.number_input("Gap = interval > k × median", 1.5, 100.0, float(s["gap_factor"]), 0.5)
    s["frozen_min_samples"] = int(c2.number_input("Frozen sensor: identical samples ≥", 3, 1000, int(s["frozen_min_samples"])))
    s["jump_mad_k"] = c3.number_input("Jump threshold (× robust spread)", 2.0, 100.0, float(s["jump_mad_k"]), 0.5)
    c1, c2, c3 = st.columns(3)
    s["figure_export_format"] = c1.selectbox("Chart toolbar export format", ["png", "svg"],
                                             index=["png", "svg"].index(s["figure_export_format"]))
    s["figure_scale"] = int(c2.number_input("Export scale (PNG)", 1, 6, int(s["figure_scale"])))
    s["live_refresh_s"] = int(c3.number_input("Live monitor refresh [s]", 2, 3600, int(s["live_refresh_s"])))

model_keys = [k for k in s if k.startswith(("xgb_", "test_size", "split_method", "cv_", "random_seed"))]
if any(before.get(k) != s.get(k) for k in model_keys):
    st.caption("ML settings changed – retrain models on the Machine Learning page to apply them.")

# ----------------------------------------------------------------------------- save / load
st.markdown("#### Configuration file")
c1, c2, c3 = st.columns(3)
c1.download_button("Download current configuration (JSON)", settings_json(s, bats), file_name="pv_bess_settings.json")
up = c2.file_uploader("Load configuration (JSON)", type=["json"], key="st_cfg")
if up is not None and st.session_state.get("st_cfg_loaded") != up.file_id:
    try:
        cfg = json.loads(up.getvalue())
        s.update({k: v for k, v in cfg.get("settings", {}).items() if k in DEFAULT_SETTINGS or k == "cv_scope"})
        if cfg.get("batteries"):
            st.session_state["batteries"] = [BatterySpec.from_dict(b).to_dict() for b in cfg["batteries"]]
        st.session_state["st_cfg_loaded"] = up.file_id
        state.invalidate_models()
        st.success("Configuration loaded.")
        st.rerun()
    except Exception as e:
        st.error(f"Invalid configuration file: {e}")
if c3.button("Reset all settings to paper defaults"):
    st.session_state["settings"] = copy.deepcopy(DEFAULT_SETTINGS)
    st.session_state["batteries"] = [b.to_dict() for b in DEFAULT_BATTERIES]
    state.invalidate_models()
    for k in list(st.session_state):
        if k.startswith("st_"):
            del st.session_state[k]
    st.rerun()
st.caption(f"An example configuration is provided in `{EXAMPLE_SETTINGS_FILE.name}`.")
