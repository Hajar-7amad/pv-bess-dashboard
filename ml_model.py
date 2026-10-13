"""
XGBoost modelling of the capacity-normalized charging rate (paper, Sec. III-D, III-F).

* One model per battery technology (chemistries are never pooled automatically).
* Target: R_cn = P_charge / E_nominal, on paper-filtered charging observations.
* 80/20 hold-out split – chronological (default, guards against temporal leakage) or random.
* 5-fold cross-validation – shuffled K-fold (paper-style stability check) or
  TimeSeriesSplit (forward-chaining).
* Metrics: R², MAE, RMSE (MAE/RMSE in h⁻¹ = kW/kWh).
* No feature scaling (tree splits are scale-invariant, as stated in the paper).
* Target-leakage guard: variables computed from charge power are blocked as predictors.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from xgboost import XGBRegressor

from config import LEAKAGE_HARD, LEAKAGE_SOFT, TARGET, ALL_VARIABLES


@dataclass
class ModelResult:
    battery_key: str
    battery_name: str
    nominal_kwh: float
    features: List[str]
    model: XGBRegressor
    metrics: Dict[str, float]
    cv: pd.DataFrame
    predictions: pd.DataFrame
    importance: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    meta: Dict = field(default_factory=dict)
    fingerprint: str = ""


def leakage_check(features: List[str], df: Optional[pd.DataFrame] = None,
                  target: str = TARGET, corr_limit: float = 0.98) -> Tuple[List[str], List[str]]:
    """Return (blocked features, warnings).

    Blocked: the target itself or variables it is computed from (charge power, signed
    battery power, interval charge energy). Warned: same measurement chain (discharge
    power) or near-perfect correlation with the target.
    """
    blocked, warnings = [], []
    for f in features:
        if f in LEAKAGE_HARD or f == target:
            blocked.append(f)
        elif f in LEAKAGE_SOFT:
            warnings.append(f"'{f}' shares the battery power measurement chain with the target; "
                            "on charging-only data it is usually constant and may leak sign information.")
    balance = {"pv_power", "load_power", "grid_power"}
    if balance <= set(features):
        msg = ("PV power, load power and grid power are all selected. Through the plant energy balance "
               "(P_charge ≈ P_PV − P_load + P_grid + P_discharge) they can jointly reconstruct the charge power "
               "the target is computed from (indirect leakage).")
        if df is not None and target in df:
            x = df[list(balance) + [target]].dropna()
            if len(x) > 20:
                A = np.c_[x[list(balance)].to_numpy(float), np.ones(len(x))]
                coef, *_ = np.linalg.lstsq(A, x[target].to_numpy(float), rcond=None)
                pred = A @ coef
                ss = ((x[target] - x[target].mean()) ** 2).sum()
                r2 = 1 - ((x[target] - pred) ** 2).sum() / ss if ss > 0 else np.nan
                msg += f" A linear fit of the target on these three variables gives R² = {r2:.3f}"
                msg += " – leakage is likely; consider removing one of them." if r2 > 0.9 else \
                    " – the identity does not hold closely in this dataset (other loads/losses), so the risk is limited."
        warnings.append(msg)
    if df is not None and target in df:
        for f in features:
            if f in blocked or f not in df:
                continue
            x = df[[f, target]].dropna()
            if len(x) > 10 and x[f].std() > 0:
                r = x[f].corr(x[target])
                if abs(r) >= corr_limit:
                    warnings.append(f"'{f}' has |r| = {abs(r):.3f} with the target – check for "
                                    "indirect leakage.")
    return blocked, warnings


def available_features(df: pd.DataFrame, target: str = TARGET) -> List[str]:
    """Numeric candidate predictors present in the data (leakage columns excluded)."""
    skip = LEAKAGE_HARD | {target, "interval_h", "flag_invalid", "flag_outlier", "charging_event_id"}
    out = []
    for c in df.columns:
        if c in skip or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].notna().sum() == 0:
            continue
        if c in ALL_VARIABLES or c in ("hour", "day", "month", "day_of_week"):
            out.append(c)
    return out


def build_dataset(df: pd.DataFrame, features: List[str], target: str = TARGET
                  ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """Model dataset: rows with all selected features and the target present
    (paper: missing values are excluded). Constant features are reported."""
    cols = [c for c in features if c in df]
    d = df[["timestamp"] + cols + [target]].copy()
    n0 = len(d)
    d = d.dropna(subset=cols + [target]).sort_values("timestamp").reset_index(drop=True)
    const = [c for c in cols if d[c].nunique() <= 1]
    info = dict(rows_in=n0, rows_missing_removed=n0 - len(d), rows_model=len(d),
                constant_features=const)
    return d[cols], d[target], d["timestamp"], info


def _xgb(settings: dict) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=int(settings.get("xgb_n_estimators", 400)),
        max_depth=int(settings.get("xgb_max_depth", 4)),
        learning_rate=float(settings.get("xgb_learning_rate", 0.05)),
        subsample=float(settings.get("xgb_subsample", 0.8)),
        colsample_bytree=float(settings.get("xgb_colsample_bytree", 0.8)),
        min_child_weight=float(settings.get("xgb_min_child_weight", 1.0)),
        reg_lambda=float(settings.get("xgb_reg_lambda", 1.0)),
        reg_alpha=float(settings.get("xgb_reg_alpha", 0.0)),
        random_state=int(settings.get("random_seed", 42)),
        objective="reg:squarederror", n_jobs=-1, tree_method="hist",
    )


def _metrics(y, p) -> Dict[str, float]:
    return dict(r2=float(r2_score(y, p)) if len(y) > 1 else np.nan,
                mae=float(mean_absolute_error(y, p)),
                rmse=float(np.sqrt(mean_squared_error(y, p))))


def fingerprint(X: pd.DataFrame, y: pd.Series, settings: dict) -> str:
    keys = sorted(k for k in settings if k.startswith(("xgb_", "test_size", "split_method",
                                                        "cv_", "random_seed")))
    h = hashlib.sha1()
    h.update(pd.util.hash_pandas_object(X, index=False).values.tobytes())
    h.update(pd.util.hash_pandas_object(y, index=False).values.tobytes())
    h.update(json.dumps({k: settings[k] for k in keys}, sort_keys=True, default=str).encode())
    h.update(",".join(X.columns).encode())
    return h.hexdigest()[:16]


def train_model(X: pd.DataFrame, y: pd.Series, ts: pd.Series, settings: dict,
                battery_key: str, battery_name: str, nominal_kwh: float,
                progress=None) -> ModelResult:
    """Train, evaluate and cross-validate one battery-specific XGBoost model."""
    if len(X) < 20:
        raise ValueError(f"Only {len(X)} complete observations – at least 20 are needed to train "
                         "and test a model.")
    seed = int(settings.get("random_seed", 42))
    test_size = float(settings.get("test_size", 0.2))
    split = settings.get("split_method", "chronological")

    idx = np.arange(len(X))
    if split == "chronological":
        cut = int(round(len(X) * (1 - test_size)))
        tr, te = idx[:cut], idx[cut:]
    else:
        tr, te = train_test_split(idx, test_size=test_size, random_state=seed)
        tr, te = np.sort(tr), np.sort(te)

    if progress:
        progress(0.15, "Fitting XGBoost on the training subset…")
    model = _xgb(settings)
    model.fit(X.iloc[tr], y.iloc[tr])
    p_tr, p_te = model.predict(X.iloc[tr]), model.predict(X.iloc[te])
    m_tr, m_te = _metrics(y.iloc[tr], p_tr), _metrics(y.iloc[te], p_te)

    # cross-validation
    k = int(settings.get("cv_folds", 5))
    scope = settings.get("cv_scope", "full")
    cv_idx = idx if scope == "full" else tr
    if settings.get("cv_method", "kfold_shuffle") == "timeseries":
        splitter = TimeSeriesSplit(n_splits=k)
    else:
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for i, (a, b) in enumerate(splitter.split(cv_idx), start=1):
        if progress:
            progress(0.15 + 0.7 * i / k, f"Cross-validation fold {i}/{k}…")
        m = _xgb(settings)
        m.fit(X.iloc[cv_idx[a]], y.iloc[cv_idx[a]])
        pr = m.predict(X.iloc[cv_idx[b]])
        folds.append(dict(fold=i, n_train=len(a), n_test=len(b), **_metrics(y.iloc[cv_idx[b]], pr)))
    cv = pd.DataFrame(folds)

    preds = pd.DataFrame({
        "timestamp": pd.concat([ts.iloc[tr], ts.iloc[te]]).values,
        "actual": np.r_[y.iloc[tr].values, y.iloc[te].values],
        "predicted": np.r_[p_tr, p_te],
        "set": ["train"] * len(tr) + ["test"] * len(te),
    })
    preds["residual"] = preds["actual"] - preds["predicted"]
    preds["battery"] = battery_name

    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    imp = pd.DataFrame({"feature": X.columns,
                        "gain": [gain.get(f, 0.0) for f in X.columns]})
    imp["gain_pct"] = 100 * imp["gain"] / imp["gain"].sum() if imp["gain"].sum() else 0
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)

    metrics = {
        "test_r2": m_te["r2"], "test_mae": m_te["mae"], "test_rmse": m_te["rmse"],
        "train_r2": m_tr["r2"], "train_mae": m_tr["mae"], "train_rmse": m_tr["rmse"],
        "cv_r2_mean": float(cv["r2"].mean()), "cv_r2_std": float(cv["r2"].std()),
        "cv_mae_mean": float(cv["mae"].mean()), "cv_mae_std": float(cv["mae"].std()),
        "cv_rmse_mean": float(cv["rmse"].mean()), "cv_rmse_std": float(cv["rmse"].std()),
    }
    meta = dict(
        battery=battery_name, nominal_kwh=nominal_kwh, target=TARGET,
        target_definition="R_cn = P_charge / E_nominal [h⁻¹]",
        date_start=str(ts.min()), date_end=str(ts.max()), n_obs=len(X),
        n_train=len(tr), n_test=len(te), split=split, test_size=test_size,
        train_period=f"{ts.iloc[tr].min()} → {ts.iloc[tr].max()}",
        test_period=f"{ts.iloc[te].min()} → {ts.iloc[te].max()}",
        cv=f"{k}-fold {'TimeSeriesSplit' if settings.get('cv_method') == 'timeseries' else 'shuffled K-fold'} "
           f"on {'all model observations' if scope == 'full' else 'training subset'}",
        features=list(X.columns), seed=seed,
        hyperparameters={k_: v for k_, v in settings.items() if k_.startswith("xgb_")},
    )
    if progress:
        progress(1.0, "Done")
    return ModelResult(battery_key=battery_key, battery_name=battery_name, nominal_kwh=nominal_kwh,
                       features=list(X.columns), model=model, metrics=metrics, cv=cv,
                       predictions=preds, importance=imp, X_train=X.iloc[tr], X_test=X.iloc[te],
                       meta=meta, fingerprint=fingerprint(X, y, settings))


def metrics_table(results: Dict[str, ModelResult]) -> pd.DataFrame:
    rows = []
    for r in results.values():
        m = r.metrics
        rows.append({
            "Battery": r.battery_name, "Observations": r.meta["n_obs"],
            "Train": r.meta["n_train"], "Test": r.meta["n_test"], "Split": r.meta["split"],
            "Test R²": m["test_r2"], "Test MAE [h⁻¹]": m["test_mae"], "Test RMSE [h⁻¹]": m["test_rmse"],
            "Train R²": m["train_r2"],
            "CV R² mean": m["cv_r2_mean"], "CV R² std": m["cv_r2_std"],
            "CV MAE mean [h⁻¹]": m["cv_mae_mean"], "CV RMSE mean [h⁻¹]": m["cv_rmse_mean"],
            "Features": len(r.features),
        })
    return pd.DataFrame(rows)


def save_model(result: ModelResult, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f"xgb_{result.battery_key}_{result.fingerprint}"
    result.model.save_model(str(stem) + ".json")
    Path(str(stem) + "_meta.json").write_text(json.dumps(
        {**result.meta, "metrics": result.metrics}, indent=2, default=str))
    return Path(str(stem) + ".json")
