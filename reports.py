"""Report / Export page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from modules import battery_comparison as bc
from modules import state, ui
from modules.export import csv_bytes, settings_json, timestamp_slug, zip_bundle
from modules.ml_model import metrics_table
from modules.reporting import build_report
from modules.shap_analysis import compare_rankings, global_importance

ui.header("Report / Export", "Download processed datasets, results tables, figures and an engineering report.")
ui.require_data()

data = state.all_battery_data()
models = st.session_state["models"]
shap_res = st.session_state["shap"]
nm = state.names()
s = state.settings()

tables = {}
for k, d in data.items():
    tables[f"{k}_processed_dataset"] = d.df
    tables[f"{k}_filtered_charging_dataset"] = d.charging
    tables[f"{k}_charging_events"] = d.events
    tables[f"{k}_preprocessing_log"] = d.processed.log
    tables[f"{k}_charging_filter_steps"] = d.charging_steps
    tables[f"{k}_synchronization_stats"] = d.sync_stats
for k, r in models.items():
    tables[f"{k}_predictions"] = r.predictions
    tables[f"{k}_cv_folds"] = r.cv
if models:
    tables["model_metrics"] = metrics_table(models)
for k, sr in shap_res.items():
    tables[f"{k}_shap_ranking"] = global_importance(sr)
if len(shap_res) >= 2:
    tables["shap_ranking_comparison"] = compare_rankings(shap_res)
keys = list(data)
if len(keys) >= 2:
    tables["battery_comparison_table"] = bc.scorecard({k: d.kpis for k, d in data.items()}, keys[0], keys[1], nm,
                                                      metrics=[m for m in data[keys[0]].kpis
                                                               if not isinstance(data[keys[0]].kpis[m], pd.Timestamp)])

st.markdown("#### Data and results tables (CSV)")
rows = [dict(table=n, rows=len(t), columns=t.shape[1]) for n, t in tables.items() if t is not None and len(t)]
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
c1, c2 = st.columns([1.4, 2])
pick = c1.selectbox("Download one table", [r["table"] for r in rows], key="rp_pick")
c1.download_button(f"Download {pick}.csv", csv_bytes(tables[pick]), file_name=f"{pick}.csv", key="rp_one")
c2.markdown("&nbsp;")
c2.download_button("Download ALL tables + settings (ZIP)",
                   zip_bundle(tables, {"settings.json": settings_json(s, st.session_state["batteries"])}),
                   file_name=f"pv_bess_results_{timestamp_slug()}.zip", key="rp_zip", type="primary")
if not models:
    st.caption("Prediction, model-metric and SHAP tables appear after training models (Machine Learning page) and "
               "opening the SHAP page.")

st.markdown("#### Figures")
st.markdown("Every chart has a camera icon in its toolbar for **PNG/SVG** export (format and scale in Settings) and "
            "an *Interactive HTML* download. SHAP beeswarm and waterfall plots offer PNG/SVG buttons.")

st.markdown("#### Engineering analysis report")
c1, c2, c3, c4 = st.columns(4)
ka = c1.selectbox("Battery A", keys, format_func=nm.get, key="rp_a") if keys else None
kb = c2.selectbox("Battery B", [k for k in keys if k != ka], format_func=nm.get, key="rp_b") if len(keys) > 1 else None
figs = c3.checkbox("Include interactive figures", True, key="rp_figs")
offline = c4.checkbox("Embed plotting library (offline, +4 MB)", False, key="rp_off")
if st.button("Generate report", type="primary", key="rp_gen"):
    with st.spinner("Building report…"):
        st.session_state["report_html"] = build_report(data, models, shap_res, s, ka, kb,
                                                       st.session_state.get("synthetic_demo", False), figs,
                                                       True if offline else "cdn")
if st.session_state.get("report_html"):
    rep = st.session_state["report_html"]
    st.download_button("Download report (HTML)", rep.encode("utf-8"), file_name=f"pv_bess_report_{timestamp_slug()}.html",
                       mime="text/html", key="rp_dl")
    st.caption("Open the HTML file in a browser; use the browser's Print → Save as PDF for a PDF copy.")
    with st.expander("Preview", expanded=False):
        st.components.v1.html(rep, height=800, scrolling=True)
