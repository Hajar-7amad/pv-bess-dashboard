"""SHAP Explainability page (single battery and cross-battery comparison)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules import state, ui
from modules.export import mpl_bytes
from modules.plotting import SEQ_SCALE, importance_bar, style
from modules.shap_analysis import (CAUSALITY_NOTE, beeswarm_figure, compare_rankings, compute_shap,
                                   global_importance, label, local_table, waterfall_figure)

ui.header("SHAP Explainability", "Additive feature contributions of the battery-specific XGBoost models.")
ui.require_data()
st.warning("**Interpretation.** " + CAUSALITY_NOTE, icon="⚠️")

models = st.session_state["models"]
if not models:
    st.info("Train at least one model on the Machine Learning page first.")
    st.page_link("pages/machine_learning.py", label="Go to Machine Learning", icon="🧠")
    st.stop()

s = state.settings()
nm = state.names()
c1, c2, c3 = st.columns([1.2, 1.2, 2])
key = c1.selectbox("Battery model", list(models), format_func=nm.get, key="sh_bat")
subset = c2.radio("Explain", ["all", "test"], horizontal=True, key="sh_sub",
                  format_func={"all": "All model observations", "test": "Test subset only"}.get)
res = models[key]
spec = state.battery(key)


def get_shap(k: str, sub: str):
    r = models[k]
    cur = st.session_state["shap"].get(k)
    if cur is None or cur.model_fingerprint != r.fingerprint or cur.subset != sub:
        with st.spinner(f"Computing SHAP values for {r.battery_name}…"):
            cur = compute_shap(r, sub, int(s["shap_max_samples"]), int(s["random_seed"]))
        st.session_state["shap"][k] = cur
    return cur


sr = get_shap(key, subset)
c3.caption(f"TreeSHAP on {len(sr.X):,} observations · base value (expected model output) = {sr.base_value:.5f} h⁻¹ · "
           f"{len(sr.X.columns)} features")
gi = global_importance(sr)

tabs = st.tabs(["Global importance", "Summary (beeswarm)", "Dependence", "Individual prediction",
                "Battery comparison"])

with tabs[0]:
    c1, c2 = st.columns([1.4, 1])
    with c1:
        ui.show_fig(importance_bar(gi, "mean_abs_shap", "label", f"Mean |SHAP| – {res.battery_name}",
                                   spec.color if spec else "#2F5D8A", xtitle="Mean |SHAP value| [h⁻¹]"),
                    "sh_gi", f"{key}_shap_importance")
    with c2:
        st.dataframe(gi[["rank", "label", "mean_abs_shap", "share_pct"]], hide_index=True, width="stretch",
                     column_config={"mean_abs_shap": st.column_config.NumberColumn("mean |SHAP|", format="%.3g"),
                                    "share_pct": st.column_config.NumberColumn("share [%]", format="%.1f")})
        ui.download_df(gi, f"{key}_shap_ranking.csv", "Download SHAP feature ranking (CSV)", key="sh_dl_gi")

with tabs[1]:
    maxd = st.slider("Features shown", 3, max(3, len(sr.X.columns)), min(15, len(sr.X.columns)), key="sh_md")
    fig = beeswarm_figure(sr, maxd)
    pc, _ = st.columns([2.2, 1])
    pc.pyplot(fig, width="stretch")
    c1, c2 = st.columns(2)
    c1.download_button("Download PNG", mpl_bytes(fig, "png"), file_name=f"{key}_shap_beeswarm.png", key="sh_bs_png")
    c2.download_button("Download SVG", mpl_bytes(fig, "svg"), file_name=f"{key}_shap_beeswarm.svg", key="sh_bs_svg")
    st.caption("Each point is one observation; horizontal position = contribution to the predicted R_cn; colour = "
               "feature value (red high, blue low).")

with tabs[2]:
    feats = list(gi["feature"])
    c1, c2 = st.columns(2)
    f = c1.selectbox("Feature", feats, format_func=label, key="sh_dep_f")
    cby = c2.selectbox("Colour by", ["(none)"] + [x for x in feats if x != f], key="sh_dep_c",
                       format_func=lambda x: x if x == "(none)" else label(x))
    j = list(sr.X.columns).index(f)
    fig = go.Figure(go.Scattergl(
        x=sr.X[f], y=sr.values[:, j], mode="markers",
        marker=dict(size=6, opacity=0.75, color=None if cby == "(none)" else sr.X[cby],
                    colorscale=SEQ_SCALE if cby != "(none)" else None,
                    showscale=cby != "(none)", colorbar=dict(title=label(cby)) if cby != "(none)" else None),
        hovertemplate=f"{label(f)} = %{{x:.4g}}<br>SHAP = %{{y:.4g}}<extra></extra>"))
    if cby == "(none)":
        fig.update_traces(marker_color=spec.color if spec else "#2F5D8A")
    fig.add_hline(y=0, line=dict(color="#6B7785", dash="dash", width=1))
    ui.show_fig(style(fig, f"SHAP dependence – {label(f)}", 420, legend=False, xlabel=ui.var_label(f),
                      ylabel=f"SHAP value for {label(f)} [h⁻¹]"), "sh_dep", f"{key}_dependence_{f}")

with tabs[3]:
    order = pd.Series(sr.timestamps).reset_index(drop=True)
    idx = st.selectbox("Timestamp", list(range(len(order))), key="sh_row",
                       format_func=lambda i: f"{pd.Timestamp(order.iloc[i]):%Y-%m-%d %H:%M}  ·  actual "
                                             f"{sr.actual[i]:.4f}  ·  predicted {sr.predicted[i]:.4f}")
    ui.kpi_grid([("Timestamp", pd.Timestamp(order.iloc[idx])), ("Actual R_cn", sr.actual[idx], "h⁻¹"),
                 ("Predicted R_cn", sr.predicted[idx], "h⁻¹"), ("Base value", sr.base_value, "h⁻¹")], 4)
    c1, c2 = st.columns([1.3, 1])
    with c1:
        wf = waterfall_figure(sr, idx)
        st.pyplot(wf, width="stretch")
        st.download_button("Download waterfall PNG", mpl_bytes(wf, "png"), file_name=f"{key}_waterfall.png",
                           key="sh_wf_png")
    with c2:
        lt = local_table(sr, idx)
        st.dataframe(lt[["Feature", "value", "shap"]], hide_index=True, width="stretch",
                     column_config={"shap": st.column_config.NumberColumn("SHAP [h⁻¹]", format="%.4g"),
                                    "value": st.column_config.NumberColumn("feature value", format="%.4g")})
        top = lt.iloc[0]
        st.caption(f"The prediction is {sr.predicted[idx] - sr.base_value:+.4f} h⁻¹ from the base value; the largest "
                   f"single contribution comes from {top['Feature']} ({top['shap']:+.4f} h⁻¹). This describes how the "
                   "model arrived at this prediction, not a physical cause.")

with tabs[4]:
    if len(models) < 2:
        st.info("Train models for at least two batteries to compare SHAP rankings.")
    else:
        all_sr = {k: get_shap(k, subset) for k in models}
        topn = st.slider("Top-N for common-feature highlight", 3, 10, 5, key="sh_topn")
        cols = st.columns(len(all_sr))
        for c, (k, r) in zip(cols, all_sr.items()):
            with c:
                g = global_importance(r)
                b = state.battery(k)
                ui.show_fig(importance_bar(g, "mean_abs_shap", "label", r.battery_name, b.color if b else "#2F5D8A",
                                           xtitle="Mean |SHAP| [h⁻¹]"), f"sh_cmp_{k}", f"{k}_shap_bar", export=False)
        tab = compare_rankings(all_sr, topn)
        flag = f"Top {topn} in all batteries"
        st.dataframe(tab.drop(columns="feature"), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3g") for c in tab.columns if "SHAP" in c})
        common = tab.loc[tab[flag], "Feature"].tolist()
        rank_cols = [c for c in tab.columns if c.endswith(" rank")]
        specific = {c.replace(" rank", ""): tab.loc[(tab[c] <= topn) & ~tab[flag], "Feature"].tolist() for c in rank_cols}
        st.markdown(f"**Common dominant features (Top {topn} for every battery):** " + (", ".join(common) or "none"))
        for b, fl in specific.items():
            st.markdown(f"**{b}-specific Top {topn} features:** " + (", ".join(fl) or "none"))
        ui.download_df(tab, "shap_ranking_comparison.csv", "Download SHAP comparison (CSV)", key="sh_dl_cmp")
        st.caption("Differences in SHAP rankings describe how each fitted model uses its inputs on its own dataset "
                   "(different periods, sample sizes and operating ranges). They do not prove chemistry-specific "
                   "physical causation.")
