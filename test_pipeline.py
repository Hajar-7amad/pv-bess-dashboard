"""End-to-end pipeline test on SYNTHETIC data (no Streamlit needed).

Run:  python -m pytest -q tests/   or   python tests/test_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from config import DEFAULT_BATTERIES, DEFAULT_SETTINGS  # noqa: E402
from modules import battery_comparison as bc  # noqa: E402
from modules.data_loader import auto_mapping, read_csv_flexible, suggest_mapping  # noqa: E402
from modules.ml_model import build_dataset, leakage_check, train_model  # noqa: E402
from modules.shap_analysis import compare_rankings, compute_shap, global_importance  # noqa: E402
from modules.state import SHARED, process_battery  # noqa: E402
from modules.synthetic import generate  # noqa: E402
import io  # noqa: E402


def _run():
    files, _ = generate()
    s = dict(DEFAULT_SETTINGS)
    mappings = {}
    for g, fs in files.items():
        for n, data in fs.items():
            df, _ = read_csv_flexible(io.BytesIO(data))
            mappings[f"{g}::{n}"] = auto_mapping(n, df)
    out = {}
    for b in DEFAULT_BATTERIES:
        out[b.key] = process_battery(b, s, {b.key: files[b.key], SHARED: files[SHARED]}, mappings)
    return out, mappings, s


def test_mapping_suggestions():
    m = suggest_mapping(["State of Charge (%)", "Charging Power (kW)", "Discharging Power (kW)",
                         "Irradiance W/m2", "PV Energy (kWh)", "PV Power (kW)", "Load (kW)",
                         "Ambient Temp (°C)", "PV Module Temperature (°C)", "Grid Power (kW)"])
    got = {c: k for c, (k, _) in m.items()}
    assert got["State of Charge (%)"] == "battery_soc"
    assert got["Charging Power (kW)"] == "battery_charge_power"
    assert got["Discharging Power (kW)"] == "battery_discharge_power"
    assert got["PV Energy (kWh)"] == "pv_energy"
    assert got["PV Power (kW)"] == "pv_power"
    assert got["Load (kW)"] == "load_power"
    assert got["Irradiance W/m2"] == "solar_irradiance"
    assert got["PV Module Temperature (°C)"] == "pv_cell_temperature"
    assert got["Ambient Temp (°C)"] == "ambient_temperature"


def test_pipeline_and_rcn():
    data, _, s = _run()
    for key, d in data.items():
        assert not d.errors, d.errors
        df = d.df
        assert len(df) > 100
        cap = d.spec.nominal_kwh
        ok = df["battery_charge_power"].notna()
        assert np.allclose(df.loc[ok, "normalized_charging_rate"], df.loc[ok, "battery_charge_power"] / cap)
        ch = d.charging
        assert (ch["battery_charge_power"] > 0).all() and (ch["battery_soc"] < s["soc_threshold"]).all()
        assert d.charging_steps["remaining"].iloc[-1] == len(ch)
        assert len(d.events) > 0
    la = data["lead_acid"]
    assert la.processed.invalid["invalid_values"].sum() >= 1  # injected SOC 104.5 %


def test_ml_shap_and_comparison():
    data, _, s = _run()
    shap_res = {}
    for key, d in data.items():
        feats = ["battery_soc", "pv_power", "load_power", "grid_power", "solar_irradiance",
                 "ambient_temperature", "pv_cell_temperature", "wind_speed", "hour", "battery_charge_power"]
        blocked, _ = leakage_check(feats, d.charging)
        assert "battery_charge_power" in blocked
        feats = [f for f in feats if f not in blocked and f in d.charging]
        X, y, ts, info = build_dataset(d.charging, feats)
        r = train_model(X, y, ts, s, key, d.spec.name, d.spec.nominal_kwh)
        assert np.isfinite(r.metrics["test_r2"]) and len(r.cv) == 5
        sr = compute_shap(r, max_samples=300)
        gi = global_importance(sr)
        assert len(gi) == len(feats)
        shap_res[key] = sr
    tab = compare_rankings(shap_res)
    assert "Top 5 in all batteries" in tab.columns
    kp = {k: d.kpis for k, d in data.items()}
    card = bc.scorecard(kp, "lead_acid", "lithium_ion", {"lead_acid": "Lead-Acid", "lithium_ion": "Lithium-Ion"})
    assert len(card) > 5
    ma, mb, info = bc.matched_conditions(data["lead_acid"].charging, data["lithium_ion"].charging,
                                         {"battery_soc": 5, "pv_power": 10})
    assert info["n_a_matched"] <= info["n_a_complete"]


if __name__ == "__main__":
    test_mapping_suggestions(); print("mapping ok")
    test_pipeline_and_rcn(); print("pipeline ok")
    test_ml_shap_and_comparison(); print("ml/shap/comparison ok")
