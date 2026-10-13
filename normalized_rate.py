"""
Capacity-normalized charging rate and charging-observation selection (paper, Sec. III-B/C).

    R_cn = P_charge / E_nominal          P_charge [kW], E_nominal [kWh]  ->  R_cn [h⁻¹]

Charging observations used for the charging-rate analysis:

    P_charge > 0   AND   SOC < SOC_threshold (default 80 %)

The scientific definition is fixed here and used by every page; only E_nominal and the
SOC threshold are user-configurable.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


def capacity_normalized_rate(charge_power_kw: pd.Series, nominal_kwh: float) -> pd.Series:
    """R_cn = P_charge / E_nominal  [h⁻¹]. Raises on a non-positive capacity."""
    if nominal_kwh is None or float(nominal_kwh) <= 0:
        raise ValueError("Nominal energy capacity must be a positive number of kWh.")
    return charge_power_kw.astype(float) / float(nominal_kwh)


def add_operating_mode(df: pd.DataFrame, charge_min: float = 0.0,
                       discharge_min: float = 0.0) -> pd.DataFrame:
    """Label each observation Charging / Discharging / Standby / Unknown."""
    df = df.copy()
    ch = df["battery_charge_power"] if "battery_charge_power" in df else pd.Series(np.nan, index=df.index)
    dis = df["battery_discharge_power"] if "battery_discharge_power" in df else pd.Series(np.nan, index=df.index)
    mode = pd.Series("Standby", index=df.index, dtype=object)
    is_ch = ch.fillna(0) > charge_min
    is_dis = (dis.fillna(0) > discharge_min) & ~(is_ch & (ch.fillna(0) >= dis.fillna(0)))
    mode[is_dis] = "Discharging"
    mode[is_ch & ~is_dis] = "Charging"
    mode[ch.isna() & dis.isna()] = "Unknown"
    df["operating_mode"] = mode
    return df


def select_modes(df: pd.DataFrame, modes: Iterable[str]) -> pd.DataFrame:
    modes = list(modes)
    if not modes or "All" in modes or "operating_mode" not in df:
        return df
    return df[df["operating_mode"].isin(modes)]


def charging_filter(df: pd.DataFrame, soc_threshold: float = 80.0,
                    charge_min: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the paper's charging-observation rule and log every step.

    Returns (filtered dataframe, step table with observations remaining after each step).
    Nothing is removed silently: each exclusion is a row in the step table.
    """
    steps: List[dict] = []
    n0 = len(df)
    steps.append(dict(step="Processed observations (input)", remaining=n0, removed=0,
                      rule="—"))
    if "battery_charge_power" not in df:
        steps.append(dict(step="Battery charge power available", remaining=0, removed=n0,
                          rule="column battery_charge_power not mapped"))
        return df.iloc[0:0], pd.DataFrame(steps)

    cur = df[df["battery_charge_power"].notna()]
    steps.append(dict(step="Charge power value present", remaining=len(cur),
                      removed=n0 - len(cur), rule="P_charge not missing"))
    prev = len(cur)
    cur = cur[cur["battery_charge_power"] > charge_min]
    steps.append(dict(step="Charging only (discharge & standby removed)", remaining=len(cur),
                      removed=prev - len(cur), rule=f"P_charge > {charge_min:g} kW"))
    prev = len(cur)
    if "battery_soc" not in cur:
        steps.append(dict(step="SOC available", remaining=0, removed=prev,
                          rule="column battery_soc not mapped"))
        return cur.iloc[0:0], pd.DataFrame(steps)
    cur = cur[cur["battery_soc"].notna()]
    steps.append(dict(step="SOC value present", remaining=len(cur), removed=prev - len(cur),
                      rule="SOC not missing"))
    prev = len(cur)
    cur = cur[cur["battery_soc"] < soc_threshold]
    steps.append(dict(step="SOC below charging cutoff", remaining=len(cur),
                      removed=prev - len(cur), rule=f"SOC < {soc_threshold:g} %"))
    return cur.copy(), pd.DataFrame(steps)


def rcn_summary(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return dict(n=0, mean=np.nan, median=np.nan, std=np.nan, p95=np.nan, max=np.nan)
    return dict(n=int(len(s)), mean=float(s.mean()), median=float(s.median()),
                std=float(s.std()), p95=float(s.quantile(0.95)), max=float(s.max()))
