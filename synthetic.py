"""
SYNTHETIC demonstration data generator.

!!! These values are artificially generated. They are NOT measurements from the
!!! EPower Monitor EMS or the paper's PV-BESS and must never be reported as results.

Purpose: let a user explore every page of the platform, and let the test-suite exercise
the full pipeline, before real EMS exports are loaded. The files deliberately use
different column names, delimiters, timestamp formats and sampling intervals so that
the column mapper and synchronization are exercised.

Simple physics: PV ∝ irradiance with a cell-temperature derating; an RO-plant-like load;
the battery absorbs PV surplus (tapering near full SOC) and covers deficits; the grid
balances the rest. Lead-Acid and Lithium-Ion operate in *different* periods (sequential
dispatch), mirroring the plant described in the paper.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

PREFIX = "SYNTHETIC_"


def _weather(idx: pd.DatetimeIndex, rng: np.random.RandomState) -> pd.DataFrame:
    hour = idx.hour + idx.minute / 60
    doy = np.asarray(idx.dayofyear)
    daylight = np.clip(np.sin(np.pi * (hour - 6) / 12.5), 0, None)
    day_ids = np.asarray((idx.normalize() - idx.normalize().min()).days)
    n_days = int(day_ids.max()) + 1
    cloud_day = rng.uniform(0.55, 1.0, n_days)[day_ids]
    cloud_fast = np.clip(1 - 0.25 * np.abs(pd.Series(rng.normal(0, 1, len(idx))).rolling(6, min_periods=1).mean().to_numpy()), 0.4, 1)
    irr = 1000 * daylight ** 1.2 * cloud_day * cloud_fast * (0.9 + 0.1 * np.cos(2 * np.pi * (doy - 172) / 365))
    amb = 18 + 6 * np.sin(2 * np.pi * (doy - 100) / 365) + 7 * np.sin(np.pi * (hour - 9) / 12) + rng.normal(0, 0.6, len(idx))
    wind = np.clip(3 + 2 * np.sin(np.pi * hour / 12) + rng.normal(0, 1.0, len(idx)), 0, None)
    cell = amb + irr / 800 * (45 - 20) * (1 - 0.05 * wind / 3)
    wdir = (300 + 40 * np.sin(2 * np.pi * day_ids / 5) + rng.normal(0, 25, len(idx))) % 360
    return pd.DataFrame({"irr": irr, "amb": amb, "cell": cell, "wind": wind, "wdir": wdir}, index=idx)


def _battery_period(idx, w, rng, nominal, pmax_c, eff, soc0, taper_start) -> pd.DataFrame:
    pv = 260 * w["irr"].values / 1000 * (1 - 0.004 * (w["cell"].values - 25)) * 0.86
    pv = np.clip(pv + rng.normal(0, 2, len(idx)), 0, None)
    hour = np.asarray(idx.hour + idx.minute / 60)
    load = 55 + 30 * np.sin(np.pi * (hour - 7) / 14).clip(0) + rng.normal(0, 6, len(idx))
    load = np.clip(load, 40, None)
    dt_h = (idx[1] - idx[0]).total_seconds() / 3600
    soc = np.empty(len(idx))
    ch = np.zeros(len(idx))
    dis = np.zeros(len(idx))
    s = soc0
    pmax = pmax_c * nominal
    for i in range(len(idx)):
        surplus = pv[i] - load[i]
        if surplus > 0 and s < 98:
            taper = 1.0 if s < taper_start else max(0.05, (100 - s) / (100 - taper_start))
            p = min(surplus, pmax * taper) * rng.uniform(0.85, 1.0)
            ch[i] = p
            s += p * eff * dt_h / nominal * 100
        elif surplus < 0 and s > 30 and (hour[i] > 17 or hour[i] < 6):
            p = min(-surplus, pmax * 0.6) * rng.uniform(0.5, 1.0)
            dis[i] = p
            s -= p * dt_h / nominal * 100
        s = float(np.clip(s, 0, 100))
        soc[i] = s
    grid = load - pv + ch - dis
    inv_t = w["amb"].values + 8 + 0.06 * (ch + dis) + rng.normal(0, 0.5, len(idx))
    return pd.DataFrame({"soc": soc, "ch": ch, "dis": dis, "pv": pv, "load": load, "grid": grid,
                         "inv": inv_t}, index=idx)


def generate(seed: int = 7) -> Tuple[Dict[str, Dict[str, bytes]], Dict[str, pd.DataFrame]]:
    """Return ({group: {file_name: csv_bytes}}, raw frames for tests)."""
    rng = np.random.RandomState(seed)
    freq = "10min"
    la_idx = pd.date_range("2025-01-06", "2025-01-31 23:50", freq=freq)
    li_idx = pd.date_range("2025-02-10", "2025-03-16 23:50", freq=freq)
    full = la_idx.append(li_idx)
    w = _weather(full, rng)
    la = _battery_period(la_idx, w.loc[la_idx], rng, 864, 0.12, 0.85, 45, 70)
    li = _battery_period(li_idx, w.loc[li_idx], rng, 510, 0.45, 0.95, 35, 90)
    plant = pd.concat([la, li])

    # --- files with deliberately heterogeneous formats --------------------------------
    la_file = pd.DataFrame({
        "Date Time": la_idx.strftime("%d/%m/%Y %H:%M"),
        "State of Charge (%)": la["soc"].round(2),
        "Charging Power (kW)": la["ch"].round(2),
        "Discharging Power (kW)": la["dis"].round(2),
    })
    # inject a few duplicates and SOC glitches so data-quality checks have something to show
    la_file = pd.concat([la_file, la_file.iloc[[100, 2000]]]).sort_index(kind="stable")
    la_file.loc[la_file.index[500], "State of Charge (%)"] = 104.5

    li_file = pd.DataFrame({
        "timestamp": li_idx.strftime("%Y-%m-%d %H:%M:%S"),
        "Battery SOC": li["soc"].round(2),
        "Battery Charge Power [kW]": li["ch"].round(2),
        "Battery Discharge Power [kW]": li["dis"].round(2),
        "Inverter Temp (°C)": li["inv"].round(1),
    })

    # weather station at 5-min resolution with ~1-2 min offset timestamps
    widx = pd.date_range(full.min(), full.max(), freq="5min") + pd.Timedelta(seconds=70)
    wf = _weather(widx, np.random.RandomState(seed + 1))
    weather = pd.DataFrame({
        "Time": widx.strftime("%Y-%m-%dT%H:%M:%S"),
        "Irradiance W/m2": wf["irr"].round(1),
        "Ambient Temp (°C)": wf["amb"].round(2),
        "PV Module Temperature (°C)": wf["cell"].round(2),
        "Wind Speed (m/s)": wf["wind"].round(2),
        "Wind Direction (deg)": wf["wdir"].round(0),
    })
    weather = weather.drop(weather.sample(frac=0.01, random_state=seed).index)  # dropouts

    pv_counter = np.cumsum(plant["pv"].values * (10 / 60))
    tot_counter = np.cumsum(plant["load"].values * (10 / 60))
    mix = pd.DataFrame({
        "Date": plant.index.strftime("%Y-%m-%d"),
        "Hour": plant.index.strftime("%H:%M"),
        "PV Power (kW)": plant["pv"].round(2),
        "PV Energy (kWh)": pv_counter.round(1),
        "Load (kW)": plant["load"].round(2),
        "Grid Power (kW)": plant["grid"].round(2),
        "Total Energy (kWh)": tot_counter.round(1),
    })

    def b(df, sep=","):
        return df.to_csv(index=False, sep=sep).encode("utf-8")

    files = {
        "lead_acid": {f"{PREFIX}lead_acid_battery.csv": b(la_file, ";")},
        "lithium_ion": {f"{PREFIX}lithium_ion_battery.csv": b(li_file)},
        "__shared__": {f"{PREFIX}weather_station.csv": b(weather),
                       f"{PREFIX}energy_mix.csv": b(mix)},
    }
    return files, {"lead_acid": la, "lithium_ion": li, "weather": wf}


if __name__ == "__main__":  # write the files to data/sample/
    from pathlib import Path

    out = Path(__file__).resolve().parents[1] / "data" / "sample"
    out.mkdir(parents=True, exist_ok=True)
    files, _ = generate()
    for group in files.values():
        for name, data in group.items():
            (out / name).write_bytes(data)
            print("wrote", out / name)
