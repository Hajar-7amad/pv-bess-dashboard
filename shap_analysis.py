"""
SHAP explainability for the battery-specific XGBoost models (paper, Sec. III-E).

SHAP values describe how the fitted model uses its inputs: they are additive feature
contributions relative to the model's expected output. They describe MODEL ASSOCIATIONS,
not physical causality – a large contribution does not prove that intervening on that
variable would change the charging rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from config import ALL_VARIABLES  # noqa: E402
from modules.ml_model import ModelResult  # noqa: E402

CAUSALITY_NOTE = (
    "SHAP describes model associations and feature contributions learned from the observed "
    "EMS data. It does NOT prove physical causality: a high SHAP contribution means the "
    "variable is influential to the model prediction, not that changing it would cause the "
    "predicted charging response."
)


@dataclass
class ShapResult:
    battery_key: str
    battery_name: str
    values: np.ndarray
    base_value: float
    X: pd.DataFrame
    timestamps: pd.Series
    predicted: np.ndarray
    actual: np.ndarray
    subset: str
    model_fingerprint: str

    def explanation(self) -> shap.Explanation:
        return shap.Explanation(values=self.values, base_values=np.full(len(self.X), self.base_value),
                                data=self.X.to_numpy(), feature_names=[label(c) for c in self.X.columns])


def label(col: str) -> str:
    v = ALL_VARIABLES.get(col)
    return v.label if v else col


def compute_shap(result: ModelResult, subset: str = "all", max_samples: int = 2000,
                 seed: int = 42) -> ShapResult:
    """TreeSHAP on the model dataset ('all' = train + test, or 'test' only)."""
    preds = result.predictions.reset_index(drop=True)
    X = pd.concat([result.X_train, result.X_test]).reset_index(drop=True)
    if subset == "test":
        mask = (preds["set"] == "test").to_numpy()
        X, preds = X[mask].reset_index(drop=True), preds[mask].reset_index(drop=True)
    if len(X) > max_samples:
        rs = np.random.RandomState(seed).choice(len(X), max_samples, replace=False)
        rs.sort()
        X, preds = X.iloc[rs].reset_index(drop=True), preds.iloc[rs].reset_index(drop=True)
    explainer = shap.TreeExplainer(result.model)
    vals = explainer.shap_values(X)
    base = explainer.expected_value
    base = float(np.ravel(base)[0])
    return ShapResult(battery_key=result.battery_key, battery_name=result.battery_name,
                      values=np.asarray(vals), base_value=base, X=X,
                      timestamps=preds["timestamp"], predicted=preds["predicted"].to_numpy(),
                      actual=preds["actual"].to_numpy(), subset=subset,
                      model_fingerprint=result.fingerprint)


def global_importance(sr: ShapResult) -> pd.DataFrame:
    imp = np.abs(sr.values).mean(axis=0)
    df = pd.DataFrame({"feature": sr.X.columns, "label": [label(c) for c in sr.X.columns],
                       "mean_abs_shap": imp})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    tot = df["mean_abs_shap"].sum()
    df["share_pct"] = 100 * df["mean_abs_shap"] / tot if tot else 0
    return df


def compare_rankings(results: Dict[str, ShapResult], top_n: int = 5) -> pd.DataFrame:
    """Feature × battery table of mean |SHAP| and rank, with 'Top-N in all' flag."""
    table = None
    tops = []
    for sr in results.values():
        g = global_importance(sr)[["feature", "mean_abs_shap", "rank"]]
        g = g.rename(columns={"mean_abs_shap": f"{sr.battery_name} mean |SHAP|",
                              "rank": f"{sr.battery_name} rank"})
        tops.append(set(g.loc[g[f"{sr.battery_name} rank"] <= top_n, "feature"]))
        table = g if table is None else table.merge(g, on="feature", how="outer")
    if table is None:
        return pd.DataFrame()
    common = set.intersection(*tops) if tops else set()
    table.insert(1, "Feature", table["feature"].map(label))
    table[f"Top {top_n} in all batteries"] = table["feature"].isin(common)
    rank_cols = [c for c in table.columns if c.endswith(" rank")]
    table["_best"] = table[rank_cols].min(axis=1)
    return table.sort_values("_best").drop(columns="_best").reset_index(drop=True)


def beeswarm_figure(sr: ShapResult, max_display: int = 15):
    plt.close("all")
    shap.plots.beeswarm(sr.explanation(), max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, max(3.5, 0.42 * min(max_display, sr.X.shape[1]) + 1.2))
    ax = plt.gca()
    ax.set_xlabel("SHAP value (contribution to predicted R_cn) [h⁻¹]")
    ax.set_title(f"SHAP summary – {sr.battery_name}", fontsize=11)
    fig.tight_layout()
    return fig


def waterfall_figure(sr: ShapResult, row: int, max_display: int = 12):
    plt.close("all")
    shap.plots.waterfall(sr.explanation()[row], max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 0.42 * max_display + 1.5)
    plt.title(f"{sr.battery_name} – {pd.Timestamp(sr.timestamps.iloc[row])}", fontsize=10)
    fig.tight_layout()
    return fig


def local_table(sr: ShapResult, row: int) -> pd.DataFrame:
    d = pd.DataFrame({"feature": sr.X.columns, "Feature": [label(c) for c in sr.X.columns],
                      "value": sr.X.iloc[row].values, "shap": sr.values[row]})
    d["abs"] = d["shap"].abs()
    return d.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
