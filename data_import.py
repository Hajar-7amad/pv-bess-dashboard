"""Data Import & Column Mapping page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import VARIABLES
from modules import state, ui
from modules.data_loader import (IGNORE, TS_SINGLE, TS_SPLIT, FileMapping, TimestampSpec,
                                 build_timestamp, score_column)
from modules.data_sources import AVAILABLE_SOURCES, CSVDataSource
from modules.synthetic import generate

OPTIONS = [IGNORE] + [f"{k} — {v.display}" for k, v in VARIABLES.items()]
KEY_OF = {o: o.split(" — ")[0] for o in OPTIONS[1:]}
OPT_OF = {v: k for k, v in KEY_OF.items()}


def _file_editor(group: str, name: str, data: bytes) -> None:
    mkey = f"{group}::{name}"
    try:
        table = CSVDataSource({name: data}).read_table(name)
    except Exception as e:
        st.error(f"Could not read {name}: {e}")
        if st.button("Remove file", key=f"rm_{mkey}"):
            state.remove_file(group, name)
            st.rerun()
        return
    df, meta = table.df, table.meta
    m: FileMapping = st.session_state["mappings"].get(mkey)
    st.caption(f"{meta['n_rows']:,} rows × {meta['n_cols']} columns · encoding {meta['encoding']} · "
               f"delimiter '{meta['delimiter']}' · decimal '{meta['decimal']}'"
               + (f" · {meta['header_row']} header line(s) skipped" if meta.get("header_row") else ""))
    st.dataframe(df.head(6), width="stretch", hide_index=True)

    cols = list(df.columns)
    st.markdown("**Timestamp**")
    c1, c2, c3, c4, c5 = st.columns([1.1, 1.6, 1.6, 0.9, 1.1])
    mode = c1.radio("Format", [TS_SINGLE, TS_SPLIT], index=0 if m.ts.mode == TS_SINGLE else 1,
                    key=f"tsm_{mkey}", format_func=lambda x: "One column" if x == TS_SINGLE else "Date + time")
    if mode == TS_SINGLE:
        col = c2.selectbox("Timestamp column", cols, index=cols.index(m.ts.column) if m.ts.column in cols else 0,
                           key=f"tsc_{mkey}")
        dcol, tcol = None, None
    else:
        dcol = c2.selectbox("Date column", cols, index=cols.index(m.ts.date_column) if m.ts.date_column in cols else 0,
                            key=f"tsd_{mkey}")
        tcol = c3.selectbox("Time column", cols, index=cols.index(m.ts.time_column) if m.ts.time_column in cols else 0,
                            key=f"tst_{mkey}")
        col = None
    dayfirst = c4.checkbox("Day first", m.ts.dayfirst, key=f"tsdf_{mkey}",
                           help="Interpret 03/04/2025 as 3 April (DD/MM/YYYY).")
    fmt = c5.text_input("Explicit format (optional)", m.ts.fmt or "", key=f"tsf_{mkey}",
                        help="strftime pattern, e.g. %d/%m/%Y %H:%M. Leave blank for automatic parsing.")
    ts_spec = TimestampSpec(mode=mode, column=col, date_column=dcol, time_column=tcol, dayfirst=dayfirst,
                            fmt=fmt or None, tz_offset_h=m.ts.tz_offset_h)
    try:
        parsed = build_timestamp(df, ts_spec)
        ok = parsed.notna().mean() * 100
        st.caption(f"Parsed {ok:.1f} % of timestamps · {parsed.min()} → {parsed.max()}"
                   + ("" if ok > 99 else " · unparseable rows will be excluded and counted"))
    except Exception as e:
        st.warning(f"Timestamp could not be parsed: {e}")

    st.markdown("**Column mapping** – automatic suggestions shown; edit any row. Scale factor converts units "
                "(e.g. 0.001 for W → kW).")
    used_ts = ts_spec.used_columns()
    rows = []
    for c in cols:
        if c in used_ts:
            continue
        mapped = m.columns.get(c)
        best = max(VARIABLES, key=lambda k: score_column(c, k))
        sample = df[c].dropna().astype(str).head(3).tolist()
        rows.append({"Source column": c, "Sample values": ", ".join(sample),
                     "Mapped variable": OPT_OF.get(mapped, IGNORE),
                     "Scale factor": float(m.scale.get(c, 1.0)),
                     "Best match score": round(score_column(c, best), 2)})
    ed = st.data_editor(
        pd.DataFrame(rows), key=f"ed_{mkey}", hide_index=True, width="stretch",
        column_config={
            "Source column": st.column_config.TextColumn(disabled=True),
            "Sample values": st.column_config.TextColumn(disabled=True),
            "Mapped variable": st.column_config.SelectboxColumn(options=OPTIONS, required=True, width="large"),
            "Scale factor": st.column_config.NumberColumn(format="%.6g", min_value=-1e6, max_value=1e6),
            "Best match score": st.column_config.ProgressColumn(min_value=0, max_value=1.2, format="%.2f"),
        })
    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("Apply mapping", key=f"ap_{mkey}", type="primary"):
        new_cols, new_scale = {}, {}
        chosen = [r for r in ed.to_dict("records") if r["Mapped variable"] != IGNORE]
        dup = pd.Series([r["Mapped variable"] for r in chosen]).duplicated()
        if dup.any():
            st.warning("Several columns are mapped to the same variable; they will be combined "
                       "(first non-missing value wins).")
        for r in chosen:
            new_cols[r["Source column"]] = KEY_OF[r["Mapped variable"]]
            new_scale[r["Source column"]] = r["Scale factor"] or 1.0
        st.session_state["mappings"][mkey] = FileMapping(name, ts_spec, new_cols, new_scale)
        state.invalidate_models()
        st.success("Mapping saved.")
        st.rerun()
    if b2.button("Remove file", key=f"rm_{mkey}"):
        state.remove_file(group, name)
        state.invalidate_models()
        st.rerun()


def _group_tab(group: str, label: str, help_txt: str) -> None:
    st.caption(help_txt)
    ukey = f"upl_{group}_{st.session_state.get('upl_counter', 0)}"
    up = st.file_uploader(f"Upload CSV files – {label}", type=["csv", "txt"], accept_multiple_files=True, key=ukey)
    if up:
        state.add_files(group, {f.name: f.getvalue() for f in up})
        st.session_state["upl_counter"] = st.session_state.get("upl_counter", 0) + 1
        state.invalidate_models()
        st.rerun()
    files = state.group_files(group)
    if not files:
        st.info("No files in this group yet.")
    for name, data in files.items():
        m = st.session_state["mappings"].get(f"{group}::{name}")
        n_mapped = len([v for v in (m.columns.values() if m else []) if v])
        with st.expander(f"📄 {name}  ·  {n_mapped} variables mapped", expanded=False):
            _file_editor(group, name, data)


ui.header("Data Import & Column Mapping",
          "Upload EMS exports, confirm timestamp and column mapping, and review synchronization.")

c1, c2 = st.columns([2, 3])
with c1:
    src = st.selectbox("Data source", list(AVAILABLE_SOURCES), index=0,
                       help="CSV is fully implemented. Database, API and live EMS sources have interface "
                            "placeholders in modules/data_sources.py.")
    if src != "CSV files":
        st.info(f"'{src}' is an interface placeholder for future integration; please use CSV files for now.")
with c2:
    st.markdown("**Quick actions**")
    a1, a2 = st.columns(2)
    if a1.button("Load SYNTHETIC demo data", help="Artificial data for exploring the software. Clearly "
                                                  "labelled; never use as results."):
        state.clear_all_data()
        files, _ = generate()
        for g, fs in files.items():
            state.add_files(g, fs)
        st.session_state["synthetic_demo"] = True
        st.rerun()
    if a2.button("Clear all loaded data"):
        state.clear_all_data()
        st.rerun()

with st.expander("CSV format guidance", expanded=not state.has_data()):
    st.markdown("""
* **One timestamp** column (any common format, or separate *date* and *time* columns) plus any number of
  measurement columns. Delimiter (`,` `;` tab `|`), encoding and decimal comma are detected automatically.
* Parameters may be split across files (e.g. `SOC.csv`, `irradiance.csv`, `energy_mix.csv`); they are
  synchronized by timestamp using the method chosen in **Settings** (default: hourly resampling, as in the paper).
* Put **battery-specific** files (SOC, charge/discharge power, battery or inverter temperature) under the battery's
  tab and **plant-level** files (irradiance, temperatures, wind, PV power/energy, load, grid) under *Shared plant
  files*; shared data are clipped to each battery's own monitoring period.
* Units expected: power kW, energy kWh, irradiance W/m², temperature °C, wind m/s and degrees, SOC %.
  Use the *scale factor* to convert (W → kW = 0.001). A signed battery-power column can be mapped instead of
  separate charge/discharge columns (sign convention in Settings).
* Variables that are not present are shown as **unavailable** – nothing is estimated or invented.
""")

tabs = st.tabs([f"{b.name} ({b.nominal_kwh:,.0f} kWh)" for b in state.batteries()] + ["Shared plant files"])
for t, b in zip(tabs[:-1], state.batteries()):
    with t:
        _group_tab(b.key, b.name, f"Files that belong to the {b.name} battery ({b.technology}).")
with tabs[-1]:
    _group_tab(state.SHARED, "shared", "Plant-level files used by every battery (weather, PV, load, grid).")

if state.has_data():
    st.subheader("Synchronization & standardization summary")
    for b in state.batteries_with_data():
        d = state.get_battery_data(b.key)
        with st.container(border=True):
            st.markdown(f"**{b.name}** – {len(d.files)} file(s), method `{d.sync_summary.get('method')}`"
                        + (f" @ `{d.sync_summary.get('interval')}`" if d.sync_summary.get("interval") else ""))
            for e in d.errors:
                st.error(e)
            ui.kpi_grid([
                ("Raw rows (all files)", d.raw_rows),
                ("Synchronized observations", len(d.master)),
                ("Processed observations", len(d.df)),
                ("Charging observations (filtered)", len(d.charging)),
            ], 4, b.color)
            if d.sync_summary.get("clipped_to_battery_period"):
                t0, t1 = d.sync_summary["battery_period"]
                st.caption(f"Shared-file rows outside {b.name}'s monitoring period ({t0} → {t1}) excluded: "
                           f"{d.sync_summary['clipped_to_battery_period']:,}")
            if d.sync_summary.get("overlapping_variables"):
                st.caption("Variables provided by several files (first non-missing value wins): " +
                           "; ".join(f"{k}: {', '.join(v)}" for k, v in d.sync_summary["overlapping_variables"].items()))
            st.dataframe(d.sync_stats, width="stretch", hide_index=True)
            rep = pd.DataFrame([dict(file=r.file_name, rows_in=r.rows_in, unparseable_timestamps=r.rows_bad_timestamp,
                                     rows_out=r.rows_out, invalid_numeric=sum(r.invalid_numeric.values()),
                                     median_interval_min=(r.median_interval_s or 0) / 60,
                                     mapped=", ".join(f"{k}→{v}" for k, v in r.mapped.items()))
                                for r in d.reports])
            with st.expander("Standardization report (per file)"):
                st.dataframe(rep, width="stretch", hide_index=True)
            ui.note_unavailable(VARIABLES.keys(), d.df)
