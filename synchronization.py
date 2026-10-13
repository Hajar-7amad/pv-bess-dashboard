"""
Timestamp synchronization of several standardized EMS tables into one master dataframe.

EMS parameters are often exported as separate files (SOC.csv, irradiance.csv,
temperature.csv...) with different sampling intervals and non-identical timestamps.
Three alignment strategies are offered:

    resample  every file is aggregated onto a common regular grid (e.g. hourly, as in the
              paper) with physically appropriate aggregation per variable, then joined
    nearest   rows of a base file are matched to the nearest record of every other file
              within a configurable tolerance (pandas.merge_asof)
    exact     outer join on identical timestamps only

Aggregation rules (``resample_frame``) never average cumulative energy registers: those
take the last reading of each interval; wind direction uses a circular mean.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import ALL_VARIABLES


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _circular_mean(x: pd.Series) -> float:
    x = x.dropna()
    if x.empty:
        return np.nan
    r = np.deg2rad(x.values)
    ang = np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))
    return float(ang % 360)


def agg_method(col: str) -> str:
    v = ALL_VARIABLES.get(col)
    return v.agg if v else "mean"


def resample_frame(df: pd.DataFrame, rule: Optional[str], ts_col: str = "timestamp",
                   cols: Optional[List[str]] = None, keep_empty: bool = False) -> pd.DataFrame:
    """Resample numeric columns of ``df`` on ``rule`` using per-variable aggregation.

    mean -> mean, sum -> sum (interval energies), counter -> last (cumulative registers),
    max -> max, circular_mean -> vector mean. Empty bins are dropped unless keep_empty.
    """
    if rule is None or df.empty:
        return df.copy()
    cols = cols or [c for c in df.columns if c != ts_col and pd.api.types.is_numeric_dtype(df[c])]
    g = df.set_index(ts_col)[cols].resample(rule)
    parts = {}
    for c in cols:
        m = agg_method(c)
        if m == "sum":
            parts[c] = g[c].sum(min_count=1)
        elif m in ("last", "counter"):
            parts[c] = g[c].last()
        elif m == "max":
            parts[c] = g[c].max()
        elif m == "circular_mean":
            parts[c] = g[c].apply(_circular_mean)
        else:
            parts[c] = g[c].mean()
    out = pd.DataFrame(parts)
    out["n_samples"] = g.size() if cols else 0
    if not keep_empty:
        out = out[out[cols].notna().any(axis=1)] if cols else out
    return out.reset_index().rename(columns={"index": ts_col})


# ---------------------------------------------------------------------------
# Synchronization
# ---------------------------------------------------------------------------
def collapse_duplicates(df: pd.DataFrame, policy: str = "keep_first") -> Tuple[pd.DataFrame, int]:
    """Collapse repeated timestamps. Returns (frame, number of rows removed)."""
    n0 = len(df)
    df = df.sort_values("timestamp", kind="stable")
    if policy == "mean":
        num = [c for c in df.columns if c != "timestamp"]
        df = df.groupby("timestamp", as_index=False)[num].mean()
    else:
        df = df.drop_duplicates(subset="timestamp", keep="first")
    return df.reset_index(drop=True), n0 - len(df)


def synchronize(frames: Dict[str, pd.DataFrame], method: str = "resample",
                interval: str = "1h", tolerance_min: float = 5.0,
                base: Optional[str] = None, duplicate_policy: str = "keep_first"
                ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Merge standardized frames into one master dataframe.

    Returns (master, per-file statistics, summary dict). Variables present in several
    files are combined in file order (first non-missing wins) and the overlap is reported.
    """
    frames = {k: v for k, v in frames.items() if v is not None and not v.empty}
    if not frames:
        return pd.DataFrame(columns=["timestamp"]), pd.DataFrame(), {}

    stats, dedup = [], {}
    clean: Dict[str, pd.DataFrame] = {}
    for name, f in frames.items():
        f2, ndup = collapse_duplicates(f, duplicate_policy)
        dedup[name] = ndup
        clean[name] = f2

    overlaps: Dict[str, List[str]] = {}
    seen: Dict[str, str] = {}
    for name, f in clean.items():
        for c in f.columns:
            if c == "timestamp":
                continue
            if c in seen:
                overlaps.setdefault(c, [seen[c]]).append(name)
            else:
                seen[c] = name

    def _combine(master: pd.DataFrame, f: pd.DataFrame, how: str) -> pd.DataFrame:
        f = f.rename(columns={c: f"{c}__new" for c in f.columns if c != "timestamp" and c in master.columns})
        m = master.merge(f, on="timestamp", how=how)
        for c in [c for c in m.columns if c.endswith("__new")]:
            base_c = c[:-5]
            m[base_c] = m[base_c].combine_first(m[c])
            m = m.drop(columns=c)
        return m

    if method == "resample":
        res = {n: resample_frame(f, interval).drop(columns="n_samples", errors="ignore")
               for n, f in clean.items()}
        master = None
        for n, f in res.items():
            master = f if master is None else _combine(master, f, "outer")
    elif method == "exact":
        master = None
        for n, f in clean.items():
            master = f if master is None else _combine(master, f, "outer")
    elif method == "nearest":
        names = list(clean)
        base = base if base in clean else max(names, key=lambda n: len(clean[n]))
        master = clean[base].sort_values("timestamp")
        tol = pd.Timedelta(minutes=float(tolerance_min))
        for n in names:
            if n == base:
                continue
            f = clean[n].sort_values("timestamp")
            f = f.rename(columns={c: f"{c}__new" for c in f.columns if c != "timestamp" and c in master.columns})
            master = pd.merge_asof(master, f, on="timestamp", direction="nearest", tolerance=tol)
            for c in [c for c in master.columns if c.endswith("__new")]:
                master[c[:-5]] = master[c[:-5]].combine_first(master[c])
                master = master.drop(columns=c)
    else:
        raise ValueError(f"Unknown synchronization method: {method}")

    master = master.sort_values("timestamp").reset_index(drop=True)

    for name, f in clean.items():
        vars_ = [c for c in f.columns if c != "timestamp"]
        present = master[vars_].notna().any(axis=1).sum() if vars_ else 0
        dt = f["timestamp"].diff().dt.total_seconds()
        stats.append(dict(
            file=name,
            variables=", ".join(vars_),
            rows=len(frames[name]),
            duplicate_timestamps=dedup[name],
            first=f["timestamp"].min(),
            last=f["timestamp"].max(),
            median_interval_min=round(float(dt.median()) / 60, 2) if len(dt.dropna()) else np.nan,
            master_rows_with_data=int(present),
            master_coverage_pct=round(100 * present / max(len(master), 1), 1),
        ))
    summary = dict(method=method, interval=interval if method == "resample" else None,
                   tolerance_min=tolerance_min if method == "nearest" else None,
                   base=base if method == "nearest" else None,
                   master_rows=len(master), overlapping_variables=overlaps,
                   duplicates_removed=int(sum(dedup.values())))
    return master, pd.DataFrame(stats), summary
