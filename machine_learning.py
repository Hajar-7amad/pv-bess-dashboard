"""Machine Learning page – battery-specific XGBoost models of R_cn."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DEFAULT_PREDICTORS, MODELS_DIR, TARGET
from modules import state, ui
from modules.ml_model import (available_features, build_dataset, fingerprint, leakage_check, metrics_table,
                              save_model, train_model)
from modules.plotting import actual_vs_predicted, importance_bar, residual_plot, style

ui.header("Machine Learning", "XGBoost regression of the capacity-normalized charging rate R_cn – one model per "
          "battery technology, trained on paper-filtered charging observations.")
ui.require_data()
s = state.settings()

c1, c2 = st.columns([1, 2.4])
with c1:
    key = ui.battery_picker(key="ml_bat")
d = state.get_battery_data(key)
spec = d.spec
data = d.charging
with c2:
    st.markdown(f"**Target:** `{TARGET}` = P_charge / {spec.nominal_kwh:,.0f} kWh [h⁻¹] · **Data:** "
                f"{len(data):,} filtered charging observations (P_charge > {s['charge_power_min_kw']:g} kW, "
                f"SOC < {s['soc_threshold']:g} %)")
    st.caption("Models are never pooled across chemistries. Feature scaling is not applied (tree models are "
               "scale-invariant).")

if len(data) < 20:
    st.warning("Fewer than 20 filtered charging observations – a model cannot be trained and tested reliably.")
    st.stop()

# ----------------------------------------------------------------------------- features
avail = available_features(data)
remembered = st.session_state["model_features"].get(key)
default = [f for f in (remembered or DEFAULT_PREDICTORS) if f in avail]
with st.container(border=True):
    st.markdown("**Predictor variables**")
    feats = st.multiselect("Select predictors", avail, default=default, format_func=ui.var_label, key=f"ml_feats_{key}")
    miss = [f for f in DEFAULT_PREDICTORS if f not in avail]
    if miss:
        st.caption("Paper-default predictors unavailable in this dataset (not estimated): " +
                   ", ".join(ui.var_label(m) for m in miss))
    blocked, warns = leakage_check(feats, data)
    st.caption("Blocked to prevent target leakage (computed from charge power): Battery Charge Power, signed Battery "
               "Power, interval charge energy, R_cn itself.")
    for w in warns:
        st.warning("Leakage check: " + w)
    feats = [f for f in feats if f not in blocked]

with st.expander("Validation design (also in Settings)", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    s["split_method"] = c1.selectbox("Hold-out split", ["chronological", "random"],
                                     index=["chronological", "random"].index(s["split_method"]), key="ml_split",
                                     help="Chronological: last 20 % of time is the test set (recommended; the paper "
                                          "attributes the test/CV gap to temporal distribution change).")
    s["test_size"] = c2.slider("Test fraction", 0.1, 0.4, float(s["test_size"]), 0.05, key="ml_ts")
    s["cv_method"] = c3.selectbox("Cross-validation", ["kfold_shuffle", "timeseries"],
                                  index=["kfold_shuffle", "timeseries"].index(s["cv_method"]), key="ml_cv",
                                  format_func={"kfold_shuffle": "Shuffled K-fold", "timeseries": "Time-series split"}.get)
    s["cv_scope"] = c4.selectbox("CV data", ["full", "train"], index=["full", "train"].index(s.get("cv_scope", "full")),
                                 key="ml_cvs", format_func={"full": "All model observations", "train": "Training subset"}.get)
    st.caption(f"{s['cv_folds']}-fold CV · seed {s['random_seed']} · XGBoost: {s['xgb_n_estimators']} trees, depth "
               f"{s['xgb_max_depth']}, learning rate {s['xgb_learning_rate']} (edit in Settings).")


def _train(k: str, features: list) -> None:
    dd = state.get_battery_data(k)
    blocked_k, _ = leakage_check(features, dd.charging)
    fts = [f for f in features if f in dd.charging and f not in blocked_k]
    X, y, ts, info = build_dataset(dd.charging, fts)
    fp = fingerprint(X, y, s)
    cur = st.session_state["models"].get(k)
    if cur is not None and cur.fingerprint == fp:
        st.toast(f"{dd.spec.name}: model already up to date – not retrained.")
        return
    bar = st.progress(0.0, text=f"Training {dd.spec.name} model…")
    try:
        res = train_model(X, y, ts, s, k, dd.spec.name, dd.spec.nominal_kwh,
                          progress=lambda p, t: bar.progress(p, text=f"{dd.spec.name}: {t}"))
        res.meta.update(info)
        st.session_state["models"][k] = res
        st.session_state["model_features"][k] = fts
        st.session_state["shap"].pop(k, None)
    except Exception as e:
        st.error(f"{dd.spec.name}: {e}")
    finally:
        bar.empty()


b1, b2, _ = st.columns([1, 1.3, 3])
if b1.button(f"Train {spec.name} model", type="primary", disabled=not feats):
    _train(key, feats)
if b2.button("Train all batteries (same predictors)", disabled=not feats):
    for b in state.batteries_with_data():
        _train(b.key, feats)

res = st.session_state["models"].get(key)
if res is None:
    st.info("No model trained for this battery yet.")
    st.stop()
X, y, ts, _ = build_dataset(data, res.features)
if fingerprint(X, y, s) != res.fingerprint:
    st.warning("Data, predictors or settings changed since this model was trained – results below are from the "
               "previous model. Retrain to update.")

m = res.metrics
st.markdown(f"#### {spec.name} model performance")
ui.kpi_grid([
    ("Test R²", m["test_r2"]), ("Test MAE", m["test_mae"], "h⁻¹"), ("Test RMSE", m["test_rmse"], "h⁻¹"),
    ("CV R²", m["cv_r2_mean"], "", f"± {m['cv_r2_std']:.3f} (std, {len(res.cv)} folds)"),
    ("Train R²", m["train_r2"]), ("CV MAE", m["cv_mae_mean"], "h⁻¹", f"± {m['cv_mae_std']:.2g}"),
    ("CV RMSE", m["cv_rmse_mean"], "h⁻¹", f"± {m['cv_rmse_std']:.2g}"),
    ("Train / test obs.", f"{res.meta['n_train']} / {res.meta['n_test']}"),
], 4, spec.color)
if m["train_r2"] - m["test_r2"] > 0.25:
    st.caption("Large train–test gap: the model may be over-fitting or the test period may differ from the training "
               "period (temporal distribution change).")

with st.expander("Model metadata", expanded=False):
    meta = res.meta
    st.json({k_: v for k_, v in meta.items()}, expanded=True)

t = st.tabs(["Actual vs predicted", "Residuals", "Error over time", "Cross-validation", "Feature importance",
             "Train vs test", "All models"])
with t[0]:
    ui.show_fig(actual_vs_predicted(res.predictions, f"Actual vs predicted R_cn – {spec.name}", spec.color),
                "ml_avp", f"{key}_actual_vs_predicted")
with t[1]:
    c1, c2 = st.columns(2)
    with c1:
        ui.show_fig(residual_plot(res.predictions, spec.color), "ml_res", f"{key}_residuals", export=False)
    with c2:
        te = res.predictions[res.predictions["set"] == "test"]
        fig = px.histogram(te, x="residual", nbins=40, color_discrete_sequence=[spec.color])
        ui.show_fig(style(fig, "Residual distribution (test)", 340, legend=False, xlabel="Residual [h⁻¹]",
                          ylabel="Count"), "ml_resh", f"{key}_residual_hist", export=False)
with t[2]:
    p = res.predictions.sort_values("timestamp")
    fig = go.Figure()
    for sset, col in (("train", "#AAB4C0"), ("test", spec.color)):
        g = p[p["set"] == sset]
        fig.add_trace(go.Scattergl(x=g["timestamp"], y=g["residual"], mode="markers", name=sset.capitalize(),
                                   marker=dict(size=5, color=col)))
    fig.add_hline(y=0, line=dict(color="#6B7785", dash="dash", width=1))
    ui.show_fig(style(fig, "Prediction error over time", 360, ylabel="Residual (actual − predicted) [h⁻¹]"), "ml_eot",
                f"{key}_error_over_time")
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=p["timestamp"], y=p["actual"], mode="markers", name="Actual",
                               marker=dict(size=5, color="#6B7785")))
    fig.add_trace(go.Scattergl(x=p["timestamp"], y=p["predicted"], mode="markers", name="Predicted",
                               marker=dict(size=5, color=spec.color, symbol="x")))
    ui.show_fig(style(fig, "Actual and predicted R_cn over time", 360, ylabel="R_cn [h⁻¹]"), "ml_apot",
                f"{key}_pred_over_time", export=False)
with t[3]:
    c1, c2 = st.columns([1, 1.3])
    c1.dataframe(res.cv, hide_index=True, width="stretch")
    fig = px.strip(res.cv, y="r2", color_discrete_sequence=[spec.color])
    fig.add_hline(y=m["cv_r2_mean"], line=dict(dash="dash", color="#6B7785"), annotation_text="mean")
    fig.add_hline(y=m["test_r2"], line=dict(dash="dot", color="#C2693E"), annotation_text="hold-out test")
    with c2:
        ui.show_fig(style(fig.update_traces(marker=dict(size=10)), "Cross-validation R² by fold", 340, legend=False,
                          ylabel="R²"), "ml_cvfig", f"{key}_cv", export=False)
    st.caption("The paper reports the hold-out test as the primary out-of-sample indicator and CV as a stability "
               "check; a higher CV R² than test R² indicates temporal distribution change, not better accuracy.")
with t[4]:
    imp = res.importance.assign(label=res.importance["feature"].map(ui.var_label))
    ui.show_fig(importance_bar(imp, "gain_pct", "label", "XGBoost feature importance (gain share)", spec.color,
                               xtitle="Gain share [%]"), "ml_imp", f"{key}_feature_importance")
    st.caption("Gain importance is model-internal. See the SHAP page for additive, per-prediction attributions.")
with t[5]:
    pr = res.predictions
    tt = pr.groupby("set").agg(n=("actual", "size"), actual_mean=("actual", "mean"), actual_std=("actual", "std"),
                               actual_max=("actual", "max"), predicted_mean=("predicted", "mean"),
                               mae=("residual", lambda r: r.abs().mean()), start=("timestamp", "min"),
                               end=("timestamp", "max"))
    st.dataframe(tt, width="stretch")
    fig = px.histogram(pr, x="actual", color="set", barmode="overlay", opacity=0.6, nbins=40,
                       color_discrete_map={"train": "#AAB4C0", "test": spec.color}, histnorm="probability density")
    ui.show_fig(style(fig, "Target distribution: training vs testing", 320, xlabel="R_cn [h⁻¹]", ylabel="Density"),
                "ml_tt", f"{key}_train_test_dist", export=False)
with t[6]:
    mt = metrics_table(st.session_state["models"])
    st.dataframe(mt, hide_index=True, width="stretch")
    ui.download_df(mt, "model_metrics.csv", "Download model metrics (CSV)", key="ml_dl_m")

c1, c2, c3 = st.columns(3)
c1.download_button("Download predictions (CSV)", res.predictions.to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"{key}_predictions.csv", key="ml_dl_p")
c2.download_button("Download CV folds (CSV)", res.cv.to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"{key}_cv_folds.csv", key="ml_dl_cv")
if c3.button("Save model to models/"):
    path = save_model(res, MODELS_DIR)
    st.success(f"Saved {path.name} (+ metadata JSON) in {MODELS_DIR}")
