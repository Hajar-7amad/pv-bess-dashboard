"""
Engineering analysis report generator (self-contained HTML + Markdown).

The report summarizes dataset, monitoring period, data quality, preprocessing, charging
statistics, battery comparison, normalized comparison, model performance, SHAP
interpretation, key figures and limitations. Every statement is descriptive.
"""
from __future__ import annotations

import html
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px

from config import APP_TITLE, SYSTEM_INFO
from modules import battery_comparison as bc
from modules.normalized_rate import rcn_summary
from modules.plotting import actual_vs_predicted, importance_bar, style, time_series
from modules.shap_analysis import CAUSALITY_NOTE, compare_rankings, global_importance

CSS = """
body{font-family:Inter,Segoe UI,Helvetica,Arial,sans-serif;color:#1F2933;max-width:1100px;margin:32px auto;padding:0 20px;
     line-height:1.45;background:#fff}
h1{font-size:26px;margin-bottom:4px} h2{font-size:20px;border-bottom:1px solid #D5DBE1;padding-bottom:4px;margin-top:34px}
h3{font-size:16px;margin-top:22px} .muted{color:#6B7785;font-size:13px}
table{border-collapse:collapse;font-size:12.5px;margin:8px 0 14px;width:100%}
th,td{border:1px solid #D5DBE1;padding:4px 7px;text-align:left;vertical-align:top} th{background:#F2F5F8}
.box{border:1px solid #C9D0D8;border-left:5px solid #2F5D8A;padding:8px 12px;margin:10px 0;background:#F7F9FB;font-size:13.5px}
.warn{border-left-color:#C2693E} .syn{border:2px solid #C2693E;padding:10px;font-weight:600}
"""


def _fmt(v, nd=4):
    if v is None:
        return "unavailable"
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "unavailable"
        return f"{v:,.{nd}g}" if abs(v) < 1e5 else f"{v:,.0f}"
    return html.escape(str(v))


def _table(df: pd.DataFrame, max_rows: int = 60, index: bool = False) -> str:
    if df is None or df.empty:
        return "<p class='muted'>No data.</p>"
    d = df.head(max_rows).copy()
    if index:
        d = d.reset_index()
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in d.columns)
    body = "".join("<tr>" + "".join(f"<td>{_fmt(v)}</td>" for v in r) + "</tr>" for r in d.itertuples(index=False))
    more = f"<p class='muted'>{len(df) - max_rows} more rows not shown.</p>" if len(df) > max_rows else ""
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{more}"


def _kv(d: dict) -> str:
    return _table(pd.DataFrame({"Indicator": list(d.keys()), "Value": list(d.values())}))


def build_report(battery_data: Dict, models: Dict, shap_results: Dict, settings: dict,
                 key_a: Optional[str] = None, key_b: Optional[str] = None, synthetic: bool = False,
                 include_figures: bool = True, plotlyjs: str = "cdn") -> str:
    parts: List[str] = []
    first_fig = [True]

    def fig_html(fig):
        if not include_figures:
            return ""
        inc = plotlyjs if first_fig[0] else False
        first_fig[0] = False
        return fig.to_html(include_plotlyjs=inc, full_html=False, config={"displaylogo": False})

    names = {k: d.spec.name for k, d in battery_data.items()}
    parts.append(f"<h1>{APP_TITLE}</h1><p class='muted'>Engineering analysis report · generated "
                 f"{datetime.now():%Y-%m-%d %H:%M}</p>")
    if synthetic:
        parts.append("<div class='syn'>SYNTHETIC DEMO DATA – this report was generated from artificially generated "
                     "files and must not be used as engineering or scientific results.</div>")

    parts.append("<h2>1. System and methodology</h2>")
    parts.append(_kv({"PV array": f"{SYSTEM_INFO['pv_array_kwp']:.0f} kWp", "DC bus": SYSTEM_INFO["dc_bus"],
                      "EMS": SYSTEM_INFO["ems"], "Dispatch": SYSTEM_INFO["dispatch"],
                      **{f"Battery – {d.spec.name}": f"{d.spec.technology}, {d.spec.chemistry}, "
                                                    f"{d.spec.nominal_kwh:,.0f} kWh" for d in battery_data.values()}}))
    parts.append(f"<div class='box'>Target indicator: <b>R<sub>cn</sub> = P<sub>charge</sub> / E<sub>nominal</sub></b> "
                 f"[kW/kWh = h⁻¹]. Charging observations: P<sub>charge</sub> &gt; {settings['charge_power_min_kw']:g} kW "
                 f"and SOC &lt; {settings['soc_threshold']:g} %. Synchronization: {settings['sync_method']}"
                 f"{' at ' + settings['resample_interval'] if settings['sync_method'] == 'resample' else ''}. "
                 f"Models: battery-specific XGBoost, {settings['split_method']} {1 - settings['test_size']:.0%}/"
                 f"{settings['test_size']:.0%} split, {settings['cv_folds']}-fold cross-validation.</div>")

    for i, (k, d) in enumerate(battery_data.items(), start=2):
        parts.append(f"<h2>{i}. {html.escape(d.spec.name)}</h2>")
        kp = d.kpis
        parts.append("<h3>Dataset and monitoring period</h3>")
        parts.append(_kv({"Files": ", ".join(d.files), "Monitoring start": kp.get("Monitoring start"),
                          "Monitoring end": kp.get("Monitoring end"), "Processed observations": len(d.df),
                          "Filtered charging observations": len(d.charging)}))
        parts.append("<h3>Data quality</h3>" + _kv(d.quality))
        parts.append("<h3>Preprocessing log</h3>" + _table(d.processed.log))
        if len(d.processed.invalid):
            parts.append("<p><b>Invalid values</b></p>" + _table(d.processed.invalid))
        parts.append("<h3>Charging-observation filter</h3>" + _table(d.charging_steps))
        if "normalized_charging_rate" in d.charging:
            rs = rcn_summary(d.charging["normalized_charging_rate"])
            parts.append("<h3>Charging statistics</h3>" + _kv({f"R_cn {k_} [h⁻¹]" if k_ != "n" else "n": v
                                                                for k_, v in rs.items()}))
        sel_k = ["Average SOC [%]", "Peak charging power [kW]", "Peak discharge power [kW]",
                 "Total charging energy [kWh]", "Total discharge energy [kWh]", "Charging duration [h]",
                 "Discharging duration [h]", "Standby duration [h]", "Charging events",
                 "Avg charging-event duration [h]", "Throughput / E_nom (equivalent full cycles)"]
        parts.append(_kv({x: kp.get(x) for x in sel_k if x in kp}))
        cols = [c for c in ["battery_soc", "battery_charge_power", "normalized_charging_rate"] if c in d.df]
        if cols:
            parts.append(fig_html(time_series(d.df, cols, mode="panels", title=f"{d.spec.name} – SOC, charge power, R_cn",
                                              rangeslider=False)))
        if len(d.charging) and "normalized_charging_rate" in d.charging:
            fig = px.histogram(d.charging, x="normalized_charging_rate", nbins=40,
                               color_discrete_sequence=[d.spec.color])
            parts.append(fig_html(style(fig, "R_cn distribution (filtered charging)", 320, legend=False,
                                        xlabel="R_cn [h⁻¹]", ylabel="Count")))
        if k in models:
            r = models[k]
            m = r.metrics
            parts.append("<h3>Model performance</h3>")
            parts.append(_kv({"Test R²": m["test_r2"], "Test MAE [h⁻¹]": m["test_mae"], "Test RMSE [h⁻¹]": m["test_rmse"],
                              "CV R² (mean ± std)": f"{m['cv_r2_mean']:.3f} ± {m['cv_r2_std']:.3f}",
                              "CV MAE [h⁻¹]": m["cv_mae_mean"], "CV RMSE [h⁻¹]": m["cv_rmse_mean"],
                              "Observations (train/test)": f"{r.meta['n_obs']} ({r.meta['n_train']}/{r.meta['n_test']})",
                              "Split": r.meta["split"], "Cross-validation": r.meta["cv"],
                              "Features": ", ".join(r.features)}))
            parts.append(fig_html(actual_vs_predicted(r.predictions, f"Actual vs predicted – {d.spec.name}", d.spec.color)))
        if k in shap_results:
            g = global_importance(shap_results[k])
            parts.append("<h3>SHAP interpretation</h3>")
            top = ", ".join(g["label"].head(3))
            parts.append(f"<p>Highest mean |SHAP| contributions: <b>{html.escape(top)}</b>.</p>")
            parts.append(_table(g[["rank", "label", "mean_abs_shap", "share_pct"]], 20))
            parts.append(fig_html(importance_bar(g, "mean_abs_shap", "label", f"Mean |SHAP| – {d.spec.name}",
                                                 d.spec.color, xtitle="Mean |SHAP| [h⁻¹]")))

    n = len(battery_data) + 2
    if key_a in battery_data and key_b in battery_data:
        K = {k: dict(d.kpis) for k, d in battery_data.items()}
        for k, r in models.items():
            if k in K:
                K[k].update({"Model test R²": r.metrics["test_r2"], "Model test MAE [h⁻¹]": r.metrics["test_mae"],
                             "Model test RMSE [h⁻¹]": r.metrics["test_rmse"],
                             "Cross-validation R² (mean)": r.metrics["cv_r2_mean"]})
        parts.append(f"<h2>{n}. Battery comparison ({html.escape(names[key_a])} vs {html.escape(names[key_b])})</h2>")
        parts.append(f"<div class='box warn'>{html.escape(bc.DISCLAIMER)}</div>")
        parts.append("<h3>Scorecard</h3>" + _table(bc.scorecard(K, key_a, key_b, names)))
        parts.append(f"<h3>Normalized comparison</h3><div class='box'>{html.escape(bc.NORMALIZATION_NOTE)}</div>")
        parts.append("<h3>Descriptive interpretation</h3><ul>" + "".join(
            f"<li>{html.escape(x)}</li>" for x in bc.interpretation(K, names, key_a, key_b,
                                                                   {k: d.charging for k, d in battery_data.items()}))
                     + "</ul>")
        if key_a in models and key_b in models:
            parts.append(f"<div class='box'>{html.escape(bc.ML_NOTE)}</div>")
        sr = {k: v for k, v in shap_results.items() if k in (key_a, key_b)}
        if len(sr) == 2:
            parts.append("<h3>SHAP ranking comparison</h3>" + _table(compare_rankings(sr).drop(columns="feature")))
        n += 1

    parts.append(f"<h2>{n}. Warnings and limitations</h2><ul>")
    lim = [CAUSALITY_NOTE, bc.DISCLAIMER, bc.ML_NOTE,
           "Hold-out and cross-validation metrics are reported separately; a gap between them indicates temporal "
           "distribution change rather than a model defect to be averaged away.",
           "Variables not present in the uploaded data are reported as unavailable; nothing was estimated or "
           "invented.",
           "This platform performs monitoring and analytics only; it issues no battery control commands."]
    for d in battery_data.values():
        for e in d.errors:
            lim.append(f"{d.spec.name}: {e}")
    parts.append("".join(f"<li>{html.escape(x)}</li>" for x in lim) + "</ul>")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>PV-BESS analysis report</title>"
            f"<style>{CSS}</style></head><body>{''.join(parts)}</body></html>")
