"""
Central configuration for the PV-BESS Explainable Analytics & Battery Performance Platform.

Everything that describes the *physical system* or the *standardized data schema*
lives here, so that new batteries, new EMS variables or new default settings can be
added without touching analysis or UI code.

Scientific definitions preserved from the paper
(Albutuwaybah, Bassi, Alghamdi, "An Explainable Machine Learning Tool for
Capacity-Normalized Battery Charging Analysis in Photovoltaic Energy Storage Systems"):

    R_cn = P_charge / E_nominal      [kW / kWh  ==  h^-1]

    Charging observations:  P_charge > 0  AND  SOC < 80 %   (threshold configurable)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

APP_TITLE = "PV-BESS Explainable Analytics & Battery Performance Platform"
APP_SHORT = "PV-BESS Analytics"
APP_VERSION = "1.0.0"

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
EXAMPLE_SETTINGS_FILE = ROOT_DIR / "settings_example.json"

# ---------------------------------------------------------------------------
# Physical system (from the paper)
# ---------------------------------------------------------------------------
SYSTEM_INFO = {
    "pv_array_kwp": 260.0,
    "dc_bus": "1000 VDC distribution bus",
    "ems": "EPower Monitor",
    "load": "Common AC load centre (PV-powered reverse-osmosis plant)",
    "dispatch": "Sequential: the two battery banks are not assumed to charge simultaneously",
}


@dataclass
class BatterySpec:
    """Description of one battery energy storage system.

    Adding a new technology (e.g. a Vanadium Redox Flow Battery) only requires a new
    BatterySpec: every page, model and comparison iterates over the registry.
    """

    key: str
    name: str
    technology: str
    chemistry: str
    nominal_kwh: float
    manufacturer: str = ""
    color: str = "#5B6B7F"
    notes: str = ""
    builtin: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "BatterySpec":
        allowed = {f for f in BatterySpec.__dataclass_fields__}
        return BatterySpec(**{k: v for k, v in d.items() if k in allowed})


# Categorical colours are assigned per battery (never by rank), muted engineering tones.
BATTERY_COLORS = ["#2F5D8A", "#C2693E", "#4C8C6B", "#7A5C99", "#8C7A3E", "#3E7F8C"]

DEFAULT_BATTERIES: List[BatterySpec] = [
    BatterySpec(
        key="lead_acid",
        name="Lead-Acid",
        technology="OPzV VRLA Gel Deep Cycle",
        chemistry="Lead-acid (VRLA gel)",
        nominal_kwh=864.0,
        manufacturer="Eternity Technologies",
        color=BATTERY_COLORS[0],
        builtin=True,
    ),
    BatterySpec(
        key="lithium_ion",
        name="Lithium-Ion",
        technology="Huawei LUNA2000-200",
        chemistry="Lithium Iron Phosphate (LFP)",
        nominal_kwh=510.0,
        manufacturer="Huawei",
        color=BATTERY_COLORS[1],
        builtin=True,
    ),
]

# Template offered in Settings for future technologies (capacity is user-defined).
FUTURE_BATTERY_TEMPLATES = {
    "Vanadium Redox Flow Battery": dict(
        technology="Vanadium Redox Flow Battery (VRFB)", chemistry="Vanadium redox flow"
    ),
    "Sodium-Ion": dict(technology="Sodium-ion", chemistry="Sodium-ion"),
    "Other": dict(technology="User-defined", chemistry="User-defined"),
}


# ---------------------------------------------------------------------------
# Standardized internal data schema
# ---------------------------------------------------------------------------
@dataclass
class Variable:
    """One standardized EMS variable.

    agg        : how to aggregate when resampling
                 mean | sum | last | max | circular_mean | counter
                 ('counter' = cumulative energy register: take last value, never average)
    nonneg     : physically non-negative (negative values are flagged)
    bounds     : (min, max) physically plausible range used for validity checks
    synonyms   : lowercase tokens/phrases used by the automatic column matcher
    """

    key: str
    label: str
    unit: str
    agg: str
    group: str
    synonyms: List[str] = field(default_factory=list)
    nonneg: bool = False
    bounds: Optional[tuple] = None
    description: str = ""

    @property
    def display(self) -> str:
        return f"{self.label} [{self.unit}]" if self.unit else self.label


TIMESTAMP_SYNONYMS = [
    "timestamp", "datetime", "date time", "date_time", "time stamp", "time",
    "date", "logtime", "log time", "recorded", "record time", "sample time", "ts",
]

VARIABLES: Dict[str, Variable] = {
    v.key: v
    for v in [
        Variable("solar_irradiance", "Solar Irradiance", "W/m²", "mean", "Environment",
                 ["irradiance", "irradiation", "ghi", "poa", "solar radiation", "radiation",
                  "pyranometer", "insolation", "solar irr", "irr"],
                 nonneg=True, bounds=(0, 1500)),
        Variable("ambient_temperature", "Ambient Temperature", "°C", "mean", "Environment",
                 ["ambient temperature", "ambient temp", "air temperature", "air temp",
                  "amb temp", "outdoor temp", "t_amb", "tamb", "ambient"],
                 bounds=(-30, 65)),
        Variable("pv_cell_temperature", "PV Cell Temperature", "°C", "mean", "Environment",
                 ["pv cell temperature", "cell temperature", "module temperature", "panel temperature",
                  "pv temp", "cell temp", "module temp", "t_cell", "tmod", "back sheet"],
                 bounds=(-30, 100)),
        Variable("wind_speed", "Wind Speed", "m/s", "mean", "Environment",
                 ["wind speed", "windspeed", "wind velocity", "ws"],
                 nonneg=True, bounds=(0, 60)),
        Variable("wind_direction", "Wind Direction", "°", "circular_mean", "Environment",
                 ["wind direction", "wind dir", "winddirection", "wd"],
                 nonneg=True, bounds=(0, 360)),
        Variable("battery_soc", "Battery SOC", "%", "mean", "Battery",
                 ["state of charge", "soc", "battery soc", "bat soc", "charge state", "soc %"],
                 nonneg=True, bounds=(0, 100)),
        Variable("battery_charge_power", "Battery Charge Power", "kW", "mean", "Battery",
                 ["charge power", "charging power", "battery charge", "bat charge", "charge kw",
                  "p charge", "p_charge", "charging"],
                 nonneg=True, bounds=(0, 2000)),
        Variable("battery_discharge_power", "Battery Discharge Power", "kW", "mean", "Battery",
                 ["discharge power", "discharging power", "battery discharge", "bat discharge",
                  "discharge kw", "p discharge", "p_discharge", "discharging"],
                 nonneg=True, bounds=(0, 2000)),
        Variable("battery_power", "Battery Power (signed)", "kW", "mean", "Battery",
                 ["battery power", "bat power", "bess power", "battery active power", "p_bat"],
                 bounds=(-2000, 2000),
                 description="Optional signed battery power; used to derive charge/discharge power "
                             "only when those are not provided directly."),
        Variable("pv_power", "PV Power", "kW", "mean", "PV",
                 ["pv power", "solar power", "pv output", "pv generation", "pv active power",
                  "array power", "dc power", "pv kw", "pv"],
                 nonneg=True, bounds=(0, 400)),
        Variable("pv_energy", "PV Energy", "kWh", "counter", "PV",
                 ["pv energy", "solar energy", "pv yield", "energy yield", "pv kwh", "e_pv"],
                 nonneg=True),
        Variable("total_energy", "Total Energy", "kWh", "counter", "System",
                 ["total energy", "energy total", "total kwh", "cumulative energy", "e_total"],
                 nonneg=True),
        Variable("load_power", "Load Power", "kW", "mean", "System",
                 ["load power", "load", "consumption", "demand", "load kw", "ac load", "p_load"],
                 nonneg=True, bounds=(0, 2000)),
        Variable("grid_power", "Grid Power", "kW", "mean", "System",
                 ["grid power", "grid", "grid interaction", "grid exchange", "import export",
                  "p_grid", "utility power", "meter power"],
                 bounds=(-2000, 2000),
                 description="Positive = import from grid, negative = export (sign configurable)."),
        Variable("inverter_temperature", "Inverter Temperature", "°C", "mean", "System",
                 ["inverter temperature", "inverter temp", "inv temp", "pcs temperature", "t_inv"],
                 bounds=(-30, 120)),
    ]
}

# Variables computed by the application (never mapped from CSV).
DERIVED_VARIABLES: Dict[str, Variable] = {
    v.key: v
    for v in [
        Variable("normalized_charging_rate", "Capacity-Normalized Charging Rate (R_cn)", "h⁻¹",
                 "mean", "Derived", nonneg=True,
                 description="R_cn = P_charge / E_nominal  [kW/kWh = h⁻¹]"),
        Variable("normalized_discharge_rate", "Capacity-Normalized Discharge Rate", "h⁻¹",
                 "mean", "Derived", nonneg=True,
                 description="P_discharge / E_nominal  [kW/kWh = h⁻¹]"),
        Variable("charge_energy", "Battery Charge Energy (interval)", "kWh", "sum", "Derived",
                 nonneg=True, description="∫ P_charge dt over each sampling interval"),
        Variable("discharge_energy", "Battery Discharge Energy (interval)", "kWh", "sum", "Derived",
                 nonneg=True, description="∫ P_discharge dt over each sampling interval"),
        Variable("hour", "Hour of Day", "h", "mean", "Temporal"),
        Variable("day", "Day of Month", "", "mean", "Temporal"),
        Variable("month", "Month", "", "mean", "Temporal"),
        Variable("day_of_week", "Day of Week (0=Mon)", "", "mean", "Temporal"),
    ]
}

ALL_VARIABLES: Dict[str, Variable] = {**VARIABLES, **DERIVED_VARIABLES}

TARGET = "normalized_charging_rate"

# Columns that are mathematically derived from the target signal -> never allowed as predictors.
LEAKAGE_HARD = {
    "battery_charge_power", "normalized_charging_rate", "battery_power", "charge_energy",
}
# Columns that share the same measurement chain as the target -> allowed only with a warning.
LEAKAGE_SOFT = {
    "battery_discharge_power", "normalized_discharge_rate", "discharge_energy",
}

# Paper-aligned default predictors (used only if present in the data).
DEFAULT_PREDICTORS = [
    "battery_soc", "pv_power", "load_power", "grid_power", "solar_irradiance",
    "ambient_temperature", "pv_cell_temperature", "wind_speed", "wind_direction",
    "inverter_temperature", "hour", "day", "month", "day_of_week",
]

OPERATING_MODES = ["Charging", "Discharging", "Standby"]

AGGREGATION_INTERVALS = {
    "Raw": None,
    "5 minutes": "5min",
    "15 minutes": "15min",
    "30 minutes": "30min",
    "Hourly": "1h",
    "Daily": "1D",
}

SOC_BANDS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]


# ---------------------------------------------------------------------------
# Default analysis settings (all editable on the Settings page)
# ---------------------------------------------------------------------------
DEFAULT_SETTINGS = {
    # Charging-event selection (paper: P_charge > 0 and SOC < 80 %)
    "soc_threshold": 80.0,
    "charge_power_min_kw": 0.0,        # observation is "charging" if P_charge > this
    "discharge_power_min_kw": 0.0,     # observation is "discharging" if P_discharge > this
    "battery_power_sign": "positive_is_discharge",  # for a signed battery_power column
    "grid_sign": "positive_is_import",
    # Synchronization / resampling (paper used hourly observations)
    "sync_method": "resample",          # resample | nearest | exact
    "resample_interval": "1h",
    "merge_tolerance_min": 5.0,
    # Preprocessing
    "duplicate_policy": "keep_first",   # keep_first | mean
    "invalid_policy": "set_nan",        # set_nan | drop_row | keep
    "outlier_method": "iqr",            # iqr | zscore | none
    "outlier_iqr_k": 3.0,
    "outlier_z": 4.0,
    "outlier_policy": "retain",         # retain | remove
    "drop_missing_for_model": True,     # paper: missing values excluded
    # Charging events
    "event_max_gap_min": 90.0,
    "event_min_duration_min": 0.0,
    "energy_max_gap_min": 90.0,         # dt longer than this is not integrated
    # Machine learning (paper: XGBoost, 80/20, 5-fold CV)
    "test_size": 0.20,
    "split_method": "chronological",    # chronological | random
    "cv_folds": 5,
    "cv_method": "kfold_shuffle",       # kfold_shuffle | timeseries
    "random_seed": 42,
    "xgb_n_estimators": 400,
    "xgb_max_depth": 4,
    "xgb_learning_rate": 0.05,
    "xgb_subsample": 0.8,
    "xgb_colsample_bytree": 0.8,
    "xgb_min_child_weight": 1.0,
    "xgb_reg_lambda": 1.0,
    "xgb_reg_alpha": 0.0,
    "shap_max_samples": 2000,
    # Matched operating condition tolerances
    "match_soc_tol": 5.0,
    "match_pv_tol": 10.0,
    "match_load_tol": 10.0,
    "match_irr_tol": 50.0,
    "match_temp_tol": 2.0,
    "match_hour_tol": 1.0,
    # Data quality
    "gap_factor": 3.0,                  # gap = dt > factor × median dt
    "frozen_min_samples": 6,
    "jump_mad_k": 8.0,
    # Presentation
    "figure_export_format": "png",      # png | svg
    "figure_scale": 2,
    "show_winner": False,               # scorecard never declares a winner unless enabled
    "live_refresh_s": 10,
}
