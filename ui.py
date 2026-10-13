"""Reusable Streamlit UI components (KPI cards, notices, pickers, figure display)."""
from __future__ import annotations

import datetime as dt
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from config import ALL_VARIABLES
from modules import state
from modules.export import csv_bytes, fig_html_bytes
from modules.plotting import plotly_config
from modules.synchronization import resample_frame

CSS = """
<style>
:root { --card-border: rgba(120,130,145,.28); --muted: rgba(110,120,135,1); }
.block-container { padding-top: 3.2rem; max-width: 1500px; }
h1, h2, h3 { letter-spacing: -.01em; }
.app-header { border-bottom: 1px solid var(--card-border); padding-bottom: .5rem; margin-bottom: 1rem; }
.app-header .t { font-size: 1.55rem; font-weight: 650; margin: 0; }
.app-header .s { color: var(--muted); font-size: .92rem; margin-top: .15rem; }
.kpi { border: 1px solid var(--card-border); border-radius: 6px; padding: .6rem .8rem .55rem;
       margin-bottom: .6rem; min-height: 84px; }
.kpi .l { font-size: .74rem; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
.kpi .v { font-size: 1.32rem; font-weight: 620; margin-top: .15rem; font-variant-numeric: tabular-nums; }
.kpi .u { font-size: .8rem; color: var(--muted); margin-left: .25rem; font-weight: 400; }
.kpi .h { font-size: .72rem; color: var(--muted); margin-top: .1rem; }
.kpi.na .v { color: var(--muted); font-size: 1rem; font-weight: 450; }
.kpi .bar { height: 3px; border-radius: 2px; margin: -.6rem -.8rem .45rem; }
.syn-banner { border: 1px solid #C2693E; border-left: 6px solid #C2693E; padding: .5rem .8rem;
              border-radius: 4px; margin-bottom: .8rem; font-size: .9rem; }
.small-muted { color: var(--muted); font-size: .82rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="app-header"><p class="t">{title}</p>'
                f'<p class="s">{subtitle}</p></div>', unsafe_allow_html=True)
    if st.session_state.get("synthetic_demo"):
        synthetic_banner()


def synthetic_banner() -> None:
    st.markdown('<div class="syn-banner"><b>SYNTHETIC DEMO DATA.</b> The loaded files are artificially '
                'generated for exploring the software. They are not EMS measurements and must not be '
                'reported as results. Load real EMS exports on the <i>Data Import</i> page.</div>',
                unsafe_allow_html=True)


def fmt(v, nd: int = 3, unit: str = "") -> str:
    if v is None:
        return "unavailable"
    if isinstance(v, (pd.Timestamp, dt.datetime)):
        return pd.Timestamp(v).strftime("%Y-%m-%d %H:%M")
    try:
        if pd.isna(v):
            return "unavailable"
    except (TypeError, ValueError):
        pass
    if isinstance(v, (int, np.integer)):
        s = f"{int(v):,}"
    elif isinstance(v, (float, np.floating)):
        a = abs(v)
        if a >= 100 and float(v).is_integer():
            s = f"{v:,.0f}"
        elif a == 0:
            s = "0"
        elif a >= 1000:
            s = f"{v:,.0f}"
        elif a >= 100:
            s = f"{v:,.1f}"
        elif a >= 1:
            s = f"{v:,.2f}"
        else:
            s = f"{v:.{nd}g}"
    else:
        s = str(v)
    return f"{s} {unit}".strip()


def kpi_card(label: str, value, unit: str = "", hint: str = "", color: Optional[str] = None) -> str:
    txt = fmt(value)
    na = txt == "unavailable"
    bar = f'<div class="bar" style="background:{color}"></div>' if color else ""
    u = f'<span class="u">{unit}</span>' if unit and not na else ""
    h = f'<div class="h">{hint}</div>' if hint else ""
    return (f'<div class="kpi{" na" if na else ""}">{bar}<div class="l">{label}</div>'
            f'<div class="v">{txt}{u}</div>{h}</div>')


def kpi_grid(items: Sequence[Tuple], ncols: int = 4, color: Optional[str] = None) -> None:
    """items: (label, value[, unit[, hint]])"""
    cols = st.columns(ncols)
    for i, it in enumerate(items):
        label, value = it[0], it[1]
        unit = it[2] if len(it) > 2 else ""
        hint = it[3] if len(it) > 3 else ""
        cols[i % ncols].markdown(kpi_card(label, value, unit, hint, color), unsafe_allow_html=True)


def split_unit(label: str) -> Tuple[str, str]:
    """'Average SOC [%]' -> ('Average SOC', '%')."""
    if label.endswith("]") and "[" in label:
        i = label.rfind("[")
        return label[:i].strip(), label[i + 1:-1]
    return label, ""


def kpi_from_dict(k: dict, keys: Iterable[str], ncols: int = 4, color: Optional[str] = None) -> None:
    items = []
    for key in keys:
        if key in k:
            l, u = split_unit(key)
            items.append((l, k[key], u))
    kpi_grid(items, ncols, color)


def notice(kind: str, text: str) -> None:
    {"info": st.info, "warn": st.warning, "error": st.error, "ok": st.success}[kind](text)


def require_data() -> bool:
    if not state.has_data():
        st.info("No EMS data loaded yet. Open **Data Import** to upload CSV files "
                "(or load the clearly-labelled synthetic demo set).")
        st.page_link("pages/data_import.py", label="Go to Data Import", icon="📥")
        st.stop()
    return True


def battery_picker(label: str = "Battery", key: str = "bp", only_with_data: bool = True,
                   default: Optional[str] = None) -> Optional[str]:
    specs = state.batteries_with_data() if only_with_data else state.batteries()
    if not specs:
        return None
    keys = [b.key for b in specs]
    nm = {b.key: f"{b.name} ({b.nominal_kwh:,.0f} kWh)" for b in specs}
    default = default or st.session_state.get("active_battery")
    idx = keys.index(default) if default in keys else 0
    k = st.selectbox(label, keys, index=idx, format_func=lambda x: nm[x], key=key)
    st.session_state["active_battery"] = k
    return k


def multi_battery_picker(label: str = "Batteries", key: str = "mbp") -> List[str]:
    specs = state.batteries_with_data()
    keys = [b.key for b in specs]
    nm = {b.key: b.name for b in specs}
    return st.multiselect(label, keys, default=keys, format_func=lambda x: nm[x], key=key)


def var_label(c: str) -> str:
    v = ALL_VARIABLES.get(c)
    return v.display if v else c


def numeric_vars(df: pd.DataFrame, include_derived: bool = True) -> List[str]:
    out = []
    for c in ALL_VARIABLES:
        if c in df and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
            if not include_derived and ALL_VARIABLES[c].group in ("Derived", "Temporal"):
                continue
            out.append(c)
    return out


def show_fig(fig, key: str, filename: str = "figure", export: bool = True) -> None:
    s = state.settings()
    st.plotly_chart(fig, width="stretch", key=key,
                    config=plotly_config(s.get("figure_export_format", "png"), s.get("figure_scale", 2), filename))
    if export:
        st.download_button("Interactive HTML", fig_html_bytes(fig), file_name=f"{filename}.html",
                           mime="text/html", key=f"dl_{key}", type="tertiary", icon=":material/download:")


def download_df(df: pd.DataFrame, filename: str, label: str = "Download CSV", key: Optional[str] = None) -> None:
    st.download_button(label, csv_bytes(df), file_name=filename, mime="text/csv",
                       key=key or f"dl_{filename}", icon=":material/download:")


def time_window(df: pd.DataFrame, key: str, label: str = "Period") -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Start/end date-time selector bounded by the data."""
    if df.empty:
        return None, None
    t0, t1 = df["timestamp"].min(), df["timestamp"].max()
    c1, c2, c3, c4 = st.columns(4)
    d0 = c1.date_input(f"{label} start date", t0.date(), min_value=t0.date(), max_value=t1.date(), key=f"{key}_d0")
    h0 = c2.time_input("Start time", dt.time(0, 0), key=f"{key}_h0")
    d1 = c3.date_input(f"{label} end date", t1.date(), min_value=t0.date(), max_value=t1.date(), key=f"{key}_d1")
    h1 = c4.time_input("End time", dt.time(23, 59), key=f"{key}_h1")
    start = pd.Timestamp(dt.datetime.combine(d0, h0))
    end = pd.Timestamp(dt.datetime.combine(d1, h1))
    if end < start:
        st.warning("End is before start – showing the full period.")
        return t0, t1
    return start, end


def clip_time(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if start is None:
        return df
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]


def aggregate(df: pd.DataFrame, rule: Optional[str], cols: Optional[List[str]] = None,
              by: Optional[str] = "battery") -> pd.DataFrame:
    """Resample with physically appropriate aggregation, separately per battery."""
    if rule is None or df.empty:
        return df
    cols = cols or [c for c in df.columns if c in ALL_VARIABLES and pd.api.types.is_numeric_dtype(df[c])]
    if by and by in df and df[by].nunique() > 1:
        parts = [resample_frame(g, rule, cols=cols).assign(**{by: b}) for b, g in df.groupby(by)]
        return pd.concat(parts, ignore_index=True)
    out = resample_frame(df, rule, cols=cols)
    if by and by in df and len(df):
        out[by] = df[by].iloc[0]
    return out


def unavailable(vars_: Iterable[str], df: pd.DataFrame) -> List[str]:
    return [v for v in vars_ if v not in df or df[v].isna().all()]


def note_unavailable(vars_: Iterable[str], df: pd.DataFrame) -> None:
    miss = unavailable(vars_, df)
    if miss:
        st.caption("Unavailable in the loaded data (not estimated): " + ", ".join(var_label(m) for m in miss))
