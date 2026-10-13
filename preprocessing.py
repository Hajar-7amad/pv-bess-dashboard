"""
Preprocessing pipeline: from a synchronized master dataframe to processed battery data.

Steps (each one logged with observations before/after and values affected):

    1. Parse / sort timestamps, remove duplicate timestamps
    2. Derive charge / discharge power from a signed battery-power column (only if needed)
    3. Detect physically invalid values (SOC outside 0-100 %, negatives where impossible,
       out-of-range readings) -> handled according to the user's policy
    4. Detect outliers (IQR or robust z-score) -> retained (flagged) or removed per policy
    5. Identify missing values (reported here; excluded at model-dataset stage, as in the paper)
    6. Compute derived variables: R_cn, normalized discharge rate, interval energies,
       temporal features, operating mode

The rule throughout: never delete data silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from config import VARIABLES, BatterySpec
from modules.normalized_rate import add_operating_mode, capacity_normalized_rate

# Variables excluded from outlier screening (cumulative counters, angles, signed values).
_NO_OUTLIER = {"pv_energy", "total_energy", "wind_direction"}


@dataclass
class ProcessedData:
    battery_key: str
    df: pd.DataFrame
    log: pd.DataFrame
    invalid: pd.DataFrame
    outliers: pd.DataFrame
    missing: pd.DataFrame
    notes: List[str] = field(default_factory=list)


def _log(rows, step, before, after, affected=0, action="", note=""):
    rows.append(dict(step=step, observations_before=before, observations_after=after,
                     values_affected=int(affected), action=action, note=note))


def interval_hours(ts: pd.Series, max_gap_min: float) -> pd.Series:
    """Duration represented by each sample (forward difference), capped at a max gap.

    Samples followed by a gap longer than ``max_gap_min`` are assigned the median interval
    so that a data outage is never integrated as if the battery ran through it.
    """
    dt = ts.shift(-1) - ts
    h = dt.dt.total_seconds() / 3600.0
    med = h[(h > 0) & (h <= max_gap_min / 60)].median()
    med = 0.0 if pd.isna(med) else med
    h = h.where((h > 0) & (h <= max_gap_min / 60), med)
    return h.fillna(med)


def detect_outliers(s: pd.Series, method: str = "iqr", k: float = 3.0, z: float = 4.0,
                    positive_only: bool = False) -> pd.Series:
    """Boolean mask of outliers. For non-negative power-like signals the statistics are
    computed on strictly positive values so that night/standby zeros do not make every
    operating value look abnormal."""
    x = s.dropna()
    if positive_only:
        x = x[x > 0]
    mask = pd.Series(False, index=s.index)
    if len(x) < 10 or method == "none":
        return mask
    if method == "iqr":
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            return mask
        lo, hi = q1 - k * iqr, q3 + k * iqr
    else:  # robust z-score (median / MAD)
        med = x.median()
        mad = (x - med).abs().median() * 1.4826
        if mad <= 0:
            return mask
        lo, hi = med - z * mad, med + z * mad
    cand = s.notna() & ((s < lo) | (s > hi))
    if positive_only:
        cand &= s > 0
    return cand


def screen_invalid(df: pd.DataFrame, policy: str = "set_nan"):
    """Flag physically invalid values per variable (bounds, sign, inf).

    Returns (frame, table of invalid counts, row mask). With policy 'set_nan' the invalid
    values are replaced by NaN; 'keep' and 'drop_row' leave values untouched here.
    """
    df = df.copy()
    rows, mask_any = [], pd.Series(False, index=df.index)
    for key, var in VARIABLES.items():
        if key not in df:
            continue
        col = df[key]
        m_neg = (col < 0) if var.nonneg else pd.Series(False, index=df.index)
        m_rng = pd.Series(False, index=df.index)
        if var.bounds:
            lo, hi = var.bounds
            m_rng = (col < lo) | (col > hi)
        m_inf = pd.Series(np.isinf(col.to_numpy(dtype=float)), index=df.index)
        m = (m_neg | m_rng | m_inf) & col.notna()
        if m.any():
            rows.append(dict(variable=key, invalid_values=int(m.sum()),
                             negative=int((m_neg & col.notna()).sum()),
                             out_of_range=int((m_rng & col.notna()).sum()),
                             bounds=str(var.bounds) if var.bounds else "—",
                             min_seen=float(col[m].min()), max_seen=float(col[m].max())))
            mask_any |= m
            if policy == "set_nan":
                df.loc[m, key] = np.nan
    return df, pd.DataFrame(rows), mask_any


def run_pipeline(master: pd.DataFrame, battery: BatterySpec, settings: dict,
                 sync_summary: Optional[dict] = None,
                 pre_invalid: Optional[pd.DataFrame] = None) -> ProcessedData:
    s = settings
    log: List[dict] = []
    notes: List[str] = []
    df = master.copy()

    # 1. timestamp, sort, duplicates ------------------------------------------------
    n_raw = len(df)
    pre_dups = (sync_summary or {}).get("duplicates_removed", 0)
    clipped = (sync_summary or {}).get("clipped_to_battery_period", 0)
    _log(log, "Original synchronized observations", n_raw + clipped, n_raw, clipped,
         "shared-file rows outside this battery's monitoring period excluded" if clipped else "",
         f"{pre_dups} duplicate timestamps collapsed inside source files during synchronization"
         if pre_dups else "")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    n_bad = int(df["timestamp"].isna().sum())
    df = df[df["timestamp"].notna()].sort_values("timestamp", kind="stable")
    _log(log, "Invalid timestamps removed", n_raw, len(df), n_bad,
         "removed" if n_bad else "none found")
    n0 = len(df)
    dups = int(df["timestamp"].duplicated().sum())
    if s.get("duplicate_policy") == "mean" and dups:
        num = [c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])]
        df = df.groupby("timestamp", as_index=False)[num].mean()
    else:
        df = df.drop_duplicates(subset="timestamp", keep="first")
    df = df.reset_index(drop=True)
    _log(log, "Duplicate timestamps removed", n0, len(df), dups,
         f"policy: {s.get('duplicate_policy', 'keep_first')}" if dups else "none found")

    # 2. signed battery power -> charge / discharge ----------------------------------
    if "battery_power" in df and df["battery_power"].notna().any():
        sign = 1.0 if s.get("battery_power_sign") == "positive_is_discharge" else -1.0
        bp = df["battery_power"] * sign  # >0 discharge
        derived = []
        if "battery_charge_power" not in df or df["battery_charge_power"].isna().all():
            df["battery_charge_power"] = (-bp).clip(lower=0).where(bp.notna())
            derived.append("battery_charge_power")
        if "battery_discharge_power" not in df or df["battery_discharge_power"].isna().all():
            df["battery_discharge_power"] = bp.clip(lower=0).where(bp.notna())
            derived.append("battery_discharge_power")
        if derived:
            notes.append(f"Derived {', '.join(derived)} from signed battery_power "
                         f"({s.get('battery_power_sign')}).")
            _log(log, "Charge/discharge derived from signed battery power", len(df), len(df),
                 note=", ".join(derived))

    # 3. invalid values -------------------------------------------------------------------
    # Raw-resolution screening already happened per source file (before resampling, so
    # that impossible readings cannot be averaged into plausible-looking values). The
    # synchronized data are screened again here.
    pol = s.get("invalid_policy", "set_nan")
    n_before = len(df)
    df, inv_post, inv_mask_any = screen_invalid(df, pol)
    pre = pre_invalid if pre_invalid is not None else pd.DataFrame()
    invalid = pd.concat([pre.assign(stage="raw files (before sync)") if len(pre) else pre,
                         inv_post.assign(stage="synchronized data") if len(inv_post) else inv_post],
                        ignore_index=True)
    n_pre = int(pre["invalid_values"].sum()) if len(pre) else 0
    n_post = int(inv_post["invalid_values"].sum()) if len(inv_post) else 0
    df["flag_invalid"] = inv_mask_any
    if pol == "drop_row" and inv_mask_any.any():
        df = df[~inv_mask_any].reset_index(drop=True)
    _log(log, "Invalid / physically impossible values", n_before, len(df), n_pre + n_post,
         {"set_nan": "values set to missing", "drop_row": "rows removed",
          "keep": "flagged only"}[pol] if (n_pre + n_post) else "none found",
         f"{n_pre} in raw files before synchronization, {n_post} after; "
         "SOC outside 0–100 %, negative where impossible, outside plausible range")

    # 4. outliers -------------------------------------------------------------------------
    out_rows = []
    out_mask_any = pd.Series(False, index=df.index)
    method = s.get("outlier_method", "iqr")
    for key, var in VARIABLES.items():
        if key not in df or key in _NO_OUTLIER or method == "none":
            continue
        positive_only = var.nonneg and var.unit in ("kW", "W/m²", "m/s")
        m = detect_outliers(df[key], method, s.get("outlier_iqr_k", 3.0), s.get("outlier_z", 4.0),
                            positive_only)
        if m.any():
            out_rows.append(dict(variable=key, outliers=int(m.sum()),
                                 pct=round(100 * m.sum() / max(df[key].notna().sum(), 1), 2),
                                 min_outlier=float(df.loc[m, key].min()),
                                 max_outlier=float(df.loc[m, key].max())))
            out_mask_any |= m
            if s.get("outlier_policy") == "remove":
                df.loc[m, key] = np.nan
    outliers = pd.DataFrame(out_rows)
    df["flag_outlier"] = out_mask_any
    n_out = int(outliers["outliers"].sum()) if not outliers.empty else 0
    _log(log, f"Outliers detected ({method.upper()})", len(df), len(df), n_out,
         ("values set to missing" if s.get("outlier_policy") == "remove" else "retained & flagged")
         if n_out else "none found")

    # 5. missing values -------------------------------------------------------------------
    present = [k for k in VARIABLES if k in df]
    missing = pd.DataFrame([
        dict(variable=k, missing=int(df[k].isna().sum()),
             missing_pct=round(100 * df[k].isna().mean(), 2)) for k in present
    ])
    rows_any_missing = int(df[present].isna().any(axis=1).sum()) if present else 0
    _log(log, "Missing values identified", len(df), len(df), int(missing["missing"].sum()) if len(missing) else 0,
         "reported; excluded per-analysis (model dataset drops rows with missing inputs)",
         f"{rows_any_missing} rows have ≥1 missing variable")

    # 6. derived variables ----------------------------------------------------------------
    if "battery_charge_power" in df:
        df["normalized_charging_rate"] = capacity_normalized_rate(df["battery_charge_power"],
                                                                  battery.nominal_kwh)
    if "battery_discharge_power" in df:
        df["normalized_discharge_rate"] = df["battery_discharge_power"] / float(battery.nominal_kwh)
    h = interval_hours(df["timestamp"], s.get("energy_max_gap_min", 90.0))
    df["interval_h"] = h
    if "battery_charge_power" in df:
        df["charge_energy"] = df["battery_charge_power"] * h
    if "battery_discharge_power" in df:
        df["discharge_energy"] = df["battery_discharge_power"] * h
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df = add_operating_mode(df, s.get("charge_power_min_kw", 0.0), s.get("discharge_power_min_kw", 0.0))
    df["battery"] = battery.name
    _log(log, "Final usable (processed) observations", len(df), len(df),
         note="R_cn computed with E_nominal = %g kWh" % battery.nominal_kwh)

    return ProcessedData(battery_key=battery.key, df=df, log=pd.DataFrame(log),
                         invalid=invalid, outliers=outliers, missing=missing, notes=notes)
