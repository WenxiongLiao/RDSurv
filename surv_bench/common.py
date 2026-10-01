# -*- coding: utf-8 -*-
"""
Common utilities for SurvSet static benchmark.

Shared by the RDSurv experiment pipeline:
- Dataset loading (SurvSet)
- Train/val/test split (same as your reference)
- Feature encoding (fit on TRAIN only) using fac_*/num_* convention
- IPCW time-specific Brier Scores and IBS using censoring KM fitted on TRAIN only
- Result aggregation and consistent printing

The sole survival model is implemented in rd_surv.py.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from typing import Dict, List, Tuple, Any, Optional


import os

# -----------------------------
# Guards
# -----------------------------
def require_lifelines():
    try:
        import lifelines  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "lifelines is required. Install via: pip install lifelines\n"
            f"Original error: {e}"
        )



def require_sksurv():
    try:
        import sksurv  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "scikit-survival is required for the time-dependent AUC metric. "
            "Install via: pip install scikit-survival\n"
            f"Original error: {e}"
        )


def require_survset():
    try:
        from SurvSet.data import SurvLoader  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "SurvSet is required. Please install SurvSet per its repository instructions.\n"
            f"Original error: {e}"
        )


# -----------------------------
# Reproducibility (numpy/random only)
# -----------------------------
def set_seed_np(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def set_seed_every(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Robust OneHotEncoder
# -----------------------------
def make_ohe():
    try:
        return OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(drop=None, sparse=False, handle_unknown="ignore")


# -----------------------------
# Split (same as your reference)
# -----------------------------
def safe_train_test_split(idx: np.ndarray, e: np.ndarray, test_size: float, seed: int):
    try:
        return train_test_split(idx, test_size=test_size, random_state=seed, stratify=e)
    except ValueError:
        return train_test_split(idx, test_size=test_size, random_state=seed, stratify=None)


# -----------------------------
# SurvSet dataset list
# -----------------------------
def default_static_list_like_reference() -> List[str]:
    # exactly your reference script list


    

    return [  'NSBCD', 'hdfail', 'nki70', 'support2','UnempDur']






def resolve_survset_datasets(env_datasets: str = "", use_all_static: bool = False) -> List[str]:
    """
    Priority:
      1) env_datasets if provided (comma-separated)
      2) if use_all_static: SurvLoader.df_ds where is_td==False
      3) default_static_list_like_reference()
    """
    require_survset()
    from SurvSet.data import SurvLoader

    if env_datasets.strip():
        return [x.strip() for x in env_datasets.split(",") if x.strip()]

    if use_all_static:
        loader = SurvLoader()
        df_ds = loader.df_ds.copy()
        name_col = "ds_name" if "ds_name" in df_ds.columns else ("ds" if "ds" in df_ds.columns else None)
        if name_col is None or "is_td" not in df_ds.columns:
            raise ValueError("SurvSet loader.df_ds must contain columns ['ds_name' or 'ds', 'is_td'].")
        # return df_ds.loc[df_ds["is_td"] == False, name_col].astype(str).tolist()  # noqa: E712
        exclude = {'AML_Bull','DBCD','DLBCL','chop','gse1992','gse3143','gse4335','vdv'}
        result = [
            x for x in df_ds.loc[df_ds["is_td"] == False, name_col].astype(str).tolist()
            if x not in exclude] 
        return result      

    return default_static_list_like_reference()


# -----------------------------
# Feature encoding (fit on TRAIN only)
# -----------------------------
def fit_transform_survset_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_categories: int = 0,  # 0 means do not drop any categoricals
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    SurvSet naming convention:
      - num_* : continuous
      - fac_* : categorical
    """
    fac_cols = [c for c in train_df.columns if c.startswith("fac_")]
    num_cols = [c for c in train_df.columns if c.startswith("num_")]

    if max_categories and max_categories > 0:
        fac_cols = [c for c in fac_cols if train_df[c].nunique(dropna=True) <= max_categories]

    transformers = []
    if fac_cols:
        enc_fac = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", make_ohe()),
        ])
        transformers.append(("fac", enc_fac, fac_cols))

    if num_cols:
        enc_num = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("num", enc_num, num_cols))

    if not transformers:
        raise ValueError("No feature columns found with prefixes fac_* or num_*.")

    enc = ColumnTransformer(transformers=transformers, remainder="drop")
    enc.fit(train_df)

    X_tr = np.asarray(enc.transform(train_df), dtype=np.float32)
    X_va = np.asarray(enc.transform(val_df), dtype=np.float32)
    X_te = np.asarray(enc.transform(test_df), dtype=np.float32)

    try:
        feat_names = enc.get_feature_names_out().tolist()
    except Exception:
        feat_names = [f"x{i}" for i in range(X_tr.shape[1])]

    return X_tr, X_va, X_te, feat_names


# -----------------------------
# IPCW IBS (same as your reference)
# -----------------------------
def km_predict_1d(kmf, times: np.ndarray) -> np.ndarray:
    try:
        v = kmf.predict(times)
        return np.asarray(v).reshape(-1)
    except Exception:
        sf = kmf.survival_function_at_times(times)
        return np.asarray(sf).reshape(-1)


def compute_ipcw_brier_curve(
    times: np.ndarray,
    y_test: np.ndarray,
    e_test: np.ndarray,
    surv_preds: np.ndarray,   # [N,T]
    censor_kmf,
    eps: float = 1e-6,
    tau_g_min: float = 0.05,   # ✅ 新增：只积分到 G(t) >= tau_g_min
    w_max: float = 50.0,       # ✅ 新增：截断权重，防止爆炸
) -> Tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    y_test = np.asarray(y_test, dtype=np.float64).reshape(-1)
    e_test = np.asarray(e_test, dtype=np.int64).reshape(-1)
    surv_preds = np.asarray(surv_preds, dtype=np.float64)

    if surv_preds.ndim != 2:
        raise ValueError(
            f"surv_preds must have shape [n_samples, n_times], got {surv_preds.shape}"
        )
    if surv_preds.shape != (y_test.shape[0], times.shape[0]):
        raise ValueError(
            "surv_preds shape must match len(y_test) and len(times): "
            f"got {surv_preds.shape}, expected {(y_test.shape[0], times.shape[0])}"
        )

    finite_times = np.isfinite(times)
    times = times[finite_times]
    surv_preds = surv_preds[:, finite_times]
    if times.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    order = np.argsort(times)
    times = times[order]
    surv_preds = surv_preds[:, order]

    unique_times, unique_indices = np.unique(times, return_index=True)
    times = unique_times
    surv_preds = surv_preds[:, unique_indices]

    # Brier-score evaluation times must be inside the TEST follow-up range.
    y_test_finite = y_test[np.isfinite(y_test)]
    if y_test_finite.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    followup_mask = (
        (times >= float(np.min(y_test_finite)))
        & (times < float(np.max(y_test_finite)))
    )
    times = times[followup_mask]
    surv_preds = surv_preds[:, followup_mask]
    if times.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    # raw censor survival on grid
    G_t_raw = km_predict_1d(censor_kmf, times)  # [T]
    if tau_g_min is not None:
        mask = (G_t_raw >= float(tau_g_min))
        times = times[mask]
        surv_preds = surv_preds[:, mask]
        G_t_raw = G_t_raw[mask]
    if times.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    G_t = np.clip(G_t_raw, eps, None)                 # [T]
    G_y = np.clip(km_predict_1d(censor_kmf, y_test), eps, None)  # [N]

    N, Tn = surv_preds.shape
    bs = np.zeros(Tn, dtype=np.float64)

    for j, t in enumerate(times):
        ind_surv = (y_test > t).astype(np.float64)

        w = np.zeros(N, dtype=np.float64)
        w[y_test > t] = 1.0 / G_t[j]

        mask_event = (y_test <= t) & (e_test == 1)
        w[mask_event] = 1.0 / G_y[mask_event]

        if w_max is not None:
            w = np.minimum(w, float(w_max))  # ✅ 截断权重

        err2 = (ind_surv - surv_preds[:, j]) ** 2
        bs[j] = np.mean(w * err2)

    return times, bs


def compute_ipcw_ibs(
    times: np.ndarray,
    y_test: np.ndarray,
    e_test: np.ndarray,
    surv_preds: np.ndarray,
    censor_kmf,
    eps: float = 1e-6,
    tau_g_min: float = 0.05,
    w_max: float = 50.0,
) -> float:
    eval_times, brier_scores = compute_ipcw_brier_curve(
        times=times,
        y_test=y_test,
        e_test=e_test,
        surv_preds=surv_preds,
        censor_kmf=censor_kmf,
        eps=eps,
        tau_g_min=tau_g_min,
        w_max=w_max,
    )
    if eval_times.size == 0:
        return float("nan")
    if eval_times.size == 1 or eval_times[-1] <= eval_times[0]:
        return float(brier_scores[0])
    return float(
        np.trapz(brier_scores, eval_times)
        / (eval_times[-1] - eval_times[0])
    )


def _survival_at_times(
    prediction_times: np.ndarray,
    surv_preds: np.ndarray,
    evaluation_times: np.ndarray,
) -> np.ndarray:
    prediction_times = np.asarray(prediction_times, dtype=np.float64).reshape(-1)
    surv_preds = np.asarray(surv_preds, dtype=np.float64)
    evaluation_times = np.asarray(evaluation_times, dtype=np.float64).reshape(-1)

    if surv_preds.ndim != 2 or surv_preds.shape[1] != prediction_times.shape[0]:
        raise ValueError(
            "surv_preds must have shape [n_samples, len(prediction_times)]."
        )

    order = np.argsort(prediction_times)
    prediction_times = prediction_times[order]
    surv_preds = surv_preds[:, order]
    return np.vstack(
        [
            np.interp(
                evaluation_times,
                prediction_times,
                row,
                left=1.0,
                right=row[-1],
            )
            for row in surv_preds
        ]
    )


def compute_ipcw_brier_metrics(
    prediction_times: np.ndarray,
    y_train: np.ndarray,
    e_train: np.ndarray,
    y_test: np.ndarray,
    e_test: np.ndarray,
    surv_preds: np.ndarray,
    censor_kmf,
    tau_g_min: float = 0.05,
    w_max: float = 50.0,
) -> Dict[str, float]:
    """Compute BS at training event-time quartiles and IBS on a common grid."""
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    e_train = np.asarray(e_train, dtype=np.int64).reshape(-1)
    prediction_times = np.asarray(prediction_times, dtype=np.float64).reshape(-1)
    surv_preds = np.asarray(surv_preds, dtype=np.float64)

    train_event_times = y_train[(e_train == 1) & np.isfinite(y_train)]
    if train_event_times.size == 0:
        horizons = np.full(3, np.nan, dtype=np.float64)
    else:
        horizons = np.quantile(train_event_times, [0.25, 0.50, 0.75])

    result = {
        "t25": float(horizons[0]),
        "t50": float(horizons[1]),
        "t75": float(horizons[2]),
        "bs_t25": float("nan"),
        "bs_t50": float("nan"),
        "bs_t75": float("nan"),
        "ibs": float("nan"),
    }

    finite_horizons = np.isfinite(horizons)
    if finite_horizons.any():
        valid_horizons = horizons[finite_horizons]
        horizon_surv = _survival_at_times(
            prediction_times, surv_preds, valid_horizons
        )
        evaluated_times, scores = compute_ipcw_brier_curve(
            times=valid_horizons,
            y_test=y_test,
            e_test=e_test,
            surv_preds=horizon_surv,
            censor_kmf=censor_kmf,
            tau_g_min=tau_g_min,
            w_max=w_max,
        )
        score_by_time = {
            float(t): float(score)
            for t, score in zip(evaluated_times, scores)
        }
        for label, horizon in zip(("t25", "t50", "t75"), horizons):
            if np.isfinite(horizon):
                result[f"bs_{label}"] = score_by_time.get(
                    float(horizon), float("nan")
                )

    result["ibs"] = compute_ipcw_ibs(
        times=prediction_times,
        y_test=y_test,
        e_test=e_test,
        surv_preds=surv_preds,
        censor_kmf=censor_kmf,
        tau_g_min=tau_g_min,
        w_max=w_max,
    )
    return result



# -----------------------------
# Benchmark data container
# -----------------------------
@dataclass
class SplitData:
    X_tr: np.ndarray
    X_va: np.ndarray
    X_te: np.ndarray
    feat_names: List[str]

    y_tr: np.ndarray
    e_tr: np.ndarray
    y_va: np.ndarray
    e_va: np.ndarray
    y_te: np.ndarray
    e_te: np.ndarray

    times: np.ndarray
    km_cens: Any  # lifelines.KaplanMeierFitter
    n_samples: int
    n_features_raw: int
    event_rate: float

    # how to choose the time horizon when a single time is needed to build a risk score
    # options: 'median' (default), 'max', or a float string like '12.0'
    risk_time: str = "median"


@dataclass
class BenchConfig:
    seed0: int = 42
    n_runs: int = 10
    test_size: float = 0.2
    val_size: float = 0.25
    n_time_grid: int = 100
    max_categories: int = 0

    # Output control
    results_dir: str = "results"
    save_risk_curves: bool = True
    risk_curves_subdir: str = "risk_curves"
    save_calibration_predictions: bool = True
    calibration_predictions_subdir: str = "calibration_predictions"
    # how to choose the evaluation time horizon when a single time is needed to build a risk score
    # options: 'median' (default), 'max', or a float string like '12.0'
    risk_time: str = "median"



@dataclass
class MethodResult:
    cindex: float
    ibs: float
    note: str = ""
    # training wall-clock time in seconds (TRAIN only; excludes prediction)
    train_time_sec: Optional[float] = None
    # risk scores on TEST (higher => higher risk)
    risk_scores_te: Optional[np.ndarray] = None
    # survival curve array on TEST, shape (n_test, n_times)
    surv_te: Optional[np.ndarray] = None
    # Filled by the shared benchmark evaluator from surv_te.
    bs_t25: Optional[float] = None
    bs_t50: Optional[float] = None
    bs_t75: Optional[float] = None



def _median_train_event_time(y_tr: np.ndarray, e_tr: np.ndarray) -> float:
    """Median of observed event times in TRAIN (e_tr==1). Falls back to median of y_tr if no events."""
    y_tr = np.asarray(y_tr, dtype=float).reshape(-1)
    e_tr = np.asarray(e_tr, dtype=int).reshape(-1)
    ev = y_tr[e_tr == 1]
    if ev.size == 0:
        return float(np.nanmedian(y_tr))
    return float(np.nanmedian(ev))



def compute_time_dependent_auc_at_t0(
    y_tr: np.ndarray,
    e_tr: np.ndarray,
    y_te: np.ndarray,
    e_te: np.ndarray,
    risk_scores_te: np.ndarray,
    t0: float,
) -> float:
    """Time-dependent AUC at single t0. Never raises, never returns NaN.

    Priority:
      1) IPCW cumulative_dynamic_auc (with finite-cleaning + safe clipping)
      2) retry IPCW with pooled censoring (train+test) if (1) fails
      3) naive dynamic AUC at t0 (exclude censored <= t0), else 0.5
    """
    require_sksurv()
    import numpy as _np
    from sksurv.metrics import cumulative_dynamic_auc
    from sksurv.nonparametric import CensoringDistributionEstimator

    y_tr = _np.asarray(y_tr, dtype=float).reshape(-1)
    e_tr = _np.asarray(e_tr, dtype=int).reshape(-1)
    y_te = _np.asarray(y_te, dtype=float).reshape(-1)
    e_te = _np.asarray(e_te, dtype=int).reshape(-1)
    rs = _np.asarray(risk_scores_te, dtype=float).reshape(-1)
    t0 = float(t0)

    # -----------------------
    # helpers
    # -----------------------
    def _finite_clean(y, e):
        """Drop non-finite times; also clip absurdly large values."""
        y = _np.asarray(y, float)
        e = _np.asarray(e, int)

        finite = _np.isfinite(y)
        y2, e2 = y[finite], e[finite]

        if y2.size == 0:
            return y2, e2

        # Clip very large values to a robust maximum (avoid overflow issues)
        # robust_max = min(max(y), p99*10) to keep ordering but prevent extreme sentinel values.
        p99 = _np.percentile(y2, 99)
        robust_max = min(float(_np.max(y2)), float(p99 * 10.0))
        if not _np.isfinite(robust_max) or robust_max <= 0:
            robust_max = float(_np.max(y2[_np.isfinite(y2)]))
        y2 = _np.clip(y2, 0.0, robust_max)
        return y2, e2

    def _to_struct(e, y):
        return _np.array([(bool(ev), float(tt)) for ev, tt in zip(e, y)],
                         dtype=[("event", bool), ("time", float)])

    def _auc_rank(pos_scores, neg_scores) -> float:
        """AUC via Mann–Whitney U with tie-handled average ranks."""
        pos_scores = _np.asarray(pos_scores, dtype=float).reshape(-1)
        neg_scores = _np.asarray(neg_scores, dtype=float).reshape(-1)
        n_pos, n_neg = pos_scores.size, neg_scores.size
        if n_pos == 0 or n_neg == 0:
            return 0.5

        scores = _np.concatenate([pos_scores, neg_scores], axis=0)
        labels = _np.concatenate([_np.ones(n_pos, dtype=int), _np.zeros(n_neg, dtype=int)], axis=0)

        order = _np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        ranks = _np.empty_like(scores, dtype=float)

        i = 0
        rank = 1.0
        while i < len(sorted_scores):
            j = i + 1
            while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
                j += 1
            avg = (rank + (rank + (j - i) - 1.0)) / 2.0
            ranks[order[i:j]] = avg
            rank += (j - i)
            i = j

        sum_ranks_pos = ranks[labels == 1].sum()
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(_np.clip(auc, 0.0, 1.0))

    def _naive_dynamic_auc_at_t0(y, e, rs, t0) -> float:
        """Dynamic/cumulative AUC at t0 without IPCW:
           cases: event by t0; controls: time > t0; exclude censored <= t0.
        """
        y = _np.asarray(y, float)
        e = _np.asarray(e, int)
        rs = _np.asarray(rs, float)

        finite = _np.isfinite(y)
        y, e, rs = y[finite], e[finite], rs[finite]

        exclude = (e == 0) & (y <= t0)
        keep = ~exclude
        y, e, rs = y[keep], e[keep], rs[keep]

        cases = (e == 1) & (y <= t0)
        controls = (y > t0)

        return _auc_rank(rs[cases], rs[controls])

    def _safe_return(x):
        if not _np.isfinite(x):
            return 0.5
        return float(_np.clip(float(x), 0.0, 1.0))

    # -----------------------
    # finite cleaning
    # -----------------------
    y_tr_c, e_tr_c = _finite_clean(y_tr, e_tr)
    y_te_c, e_te_c = _finite_clean(y_te, e_te)

    # if too few samples after cleaning, fallback
    if y_tr_c.size < 5 or y_te_c.size < 5:
        return _safe_return(_naive_dynamic_auc_at_t0(y_te_c, e_te_c, rs[: y_te_c.size], t0))

    # make sure rs aligns with cleaned test set length (if some test times were dropped)
    # safest: drop rs where y_te was non-finite originally
    finite_te = _np.isfinite(_np.asarray(y_te, float).reshape(-1))
    rs_c = rs[finite_te]
    if rs_c.size != y_te_c.size:
        # last resort align by truncation
        n = min(rs_c.size, y_te_c.size)
        rs_c, y_te_c, e_te_c = rs_c[:n], y_te_c[:n], e_te_c[:n]

    # adjust t0 into finite range
    t0 = float(t0)
    if not _np.isfinite(t0):
        # default to median of event times in training (finite-cleaned)
        ev = y_tr_c[e_tr_c == 1]
        t0 = float(_np.median(ev)) if ev.size else float(_np.median(y_tr_c))
    t0 = max(0.0, t0)

    # if training has no censoring, IPCW is unstable/unnecessary -> naive
    if _np.sum(e_tr_c == 0) == 0:
        return _safe_return(_naive_dynamic_auc_at_t0(y_te_c, e_te_c, rs_c, t0))

    surv_tr = _to_struct(e_tr_c, y_tr_c)

    # -----------------------
    # IPCW attempt with safe clipping
    # -----------------------
    def _try_ipcw_auc(surv_train_struct, y_test, e_test, rs_test, t0_in) -> float:
        cens = CensoringDistributionEstimator()
        cens.fit(surv_train_struct)

        unique_t = getattr(cens, "unique_time_", None)
        if unique_t is None or len(unique_t) == 0:
            raise ValueError("no censoring support")

        unique_t = _np.asarray(unique_t, dtype=float)
        unique_t = unique_t[_np.isfinite(unique_t)]
        if unique_t.size == 0:
            raise ValueError("unique_time_ not finite")

        # ensure sorted
        unique_t = _np.sort(unique_t)

        # predict G(t) safely
        G = cens.predict_proba(unique_t)
        pos = _np.where(G > 0.0)[0]
        if pos.size == 0:
            raise ValueError("G(t)=0 everywhere")

        t_max_ok = float(unique_t[pos[-1]])
        eps = 1e-8

        t0_safe = min(float(t0_in), t_max_ok - eps)
        if not _np.isfinite(t0_safe) or t0_safe <= 0:
            raise ValueError("invalid t0 after clipping")

        y_test = _np.asarray(y_test, float)
        y_test = y_test[_np.isfinite(y_test)]
        # clip test times into support
        y_test_safe = _np.minimum(y_test, t_max_ok - eps)

        # align e_test and rs_test with y_test_safe length (since we filtered finite)
        # (y_test is already cleaned before calling; keep simple)
        surv_test_struct = _to_struct(e_test, y_test_safe)

        out = cumulative_dynamic_auc(
            surv_train_struct,
            surv_test_struct,
            rs_test,
            _np.array([t0_safe], dtype=float),
        )

        # parse outputs across versions
        if isinstance(out, tuple) and len(out) >= 2:
            a0, a1 = _np.asarray(out[0]), _np.asarray(out[1])
            if a0.shape == (1,) and _np.allclose(a0, _np.array([t0_safe]), equal_nan=False):
                aucs = a1
            elif a0.shape == (1,) and a1.shape == ():
                aucs = a0
            elif a1.shape == (1,):
                aucs = a1
            else:
                aucs = a0
            return float(_np.asarray(aucs).reshape(-1)[0])

        return float(_np.asarray(out).reshape(-1)[0])

    # attempt 1: training censoring
    try:
        auc = _try_ipcw_auc(surv_tr, y_te_c, e_te_c, rs_c, t0)
        return _safe_return(auc)
    except Exception:
        pass

    # attempt 2: pooled censoring (train+test) for stability
    try:
        y_pool = _np.concatenate([y_tr_c, y_te_c], axis=0)
        e_pool = _np.concatenate([e_tr_c, e_te_c], axis=0)
        surv_pool = _to_struct(e_pool, y_pool)
        auc = _try_ipcw_auc(surv_pool, y_te_c, e_te_c, rs_c, t0)
        return _safe_return(auc)
    except Exception:
        pass

    # final fallback: naive AUC (always defined)
    auc = _naive_dynamic_auc_at_t0(y_te_c, e_te_c, rs_c, t0)
    return _safe_return(auc)




def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_calibration_predictions(
    *,
    prediction_times: np.ndarray,
    surv_preds: np.ndarray,
    y_train: np.ndarray,
    e_train: np.ndarray,
    y_test: np.ndarray,
    e_test: np.ndarray,
    sample_ids: np.ndarray,
    dataset: str,
    method: str,
    run_id: int,
    seed: int,
    out_dir: str,
) -> str:
    """Save TEST predictions at training event-time quartiles for calibration plots."""
    prediction_times = np.asarray(prediction_times, dtype=np.float64).reshape(-1)
    surv_preds = np.asarray(surv_preds, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    e_train = np.asarray(e_train, dtype=np.int64).reshape(-1)
    y_test = np.asarray(y_test, dtype=np.float64).reshape(-1)
    e_test = np.asarray(e_test, dtype=np.int64).reshape(-1)
    sample_ids = np.asarray(sample_ids).reshape(-1)

    n_test = y_test.shape[0]
    if surv_preds.shape != (n_test, prediction_times.shape[0]):
        raise ValueError(
            "surv_preds shape must match TEST samples and prediction_times: "
            f"got {surv_preds.shape}, expected {(n_test, prediction_times.shape[0])}"
        )
    if e_test.shape[0] != n_test or sample_ids.shape[0] != n_test:
        raise ValueError("y_test, e_test, and sample_ids must have equal length.")

    train_event_times = y_train[(e_train == 1) & np.isfinite(y_train)]
    if train_event_times.size == 0:
        horizons = np.full(3, np.nan, dtype=np.float64)
    else:
        horizons = np.quantile(train_event_times, [0.25, 0.50, 0.75])

    data = {
        "dataset": np.repeat(dataset, n_test),
        "method": np.repeat(method, n_test),
        "run_id": np.repeat(int(run_id), n_test),
        "seed": np.repeat(int(seed), n_test),
        "sample_id": sample_ids,
        "observed_time": y_test,
        "event": e_test,
    }

    finite_prediction_times = prediction_times[np.isfinite(prediction_times)]
    prediction_min = (
        float(np.min(finite_prediction_times))
        if finite_prediction_times.size
        else float("nan")
    )
    prediction_max = (
        float(np.max(finite_prediction_times))
        if finite_prediction_times.size
        else float("nan")
    )

    for label, horizon in zip(("t25", "t50", "t75"), horizons):
        data[f"horizon_{label}"] = np.repeat(float(horizon), n_test)
        supported = (
            np.isfinite(horizon)
            and np.isfinite(prediction_min)
            and prediction_min <= horizon <= prediction_max
        )
        if supported:
            survival = _survival_at_times(
                prediction_times, surv_preds, np.asarray([horizon])
            )[:, 0]
            data[f"pred_survival_{label}"] = survival
            data[f"pred_risk_{label}"] = 1.0 - survival
        else:
            data[f"pred_survival_{label}"] = np.full(n_test, np.nan)
            data[f"pred_risk_{label}"] = np.full(n_test, np.nan)

    out_dir = _ensure_dir(out_dir)
    dataset_dir = _ensure_dir(os.path.join(out_dir, dataset))
    method_dir = _ensure_dir(os.path.join(dataset_dir, method))
    output_path = os.path.join(
        method_dir, f"run_{int(run_id):02d}_calibration.csv"
    )
    pd.DataFrame(data).to_csv(output_path, index=False)
    return output_path


def save_high_low_risk_km_curves(
    *,
    y_te: np.ndarray,
    e_te: np.ndarray,
    risk_scores_te: np.ndarray,
    dataset: str,
    method: str,
    run_id: int,
    out_dir: str,
) -> Dict[str, str]:
    """Split TEST into high/low risk groups by median risk and save KM curves (png + csv)."""
    require_lifelines()
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import pandas as pd

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    out_dir = _ensure_dir(out_dir)
    ds_dir = _ensure_dir(os.path.join(out_dir, dataset))
    m_dir = _ensure_dir(os.path.join(ds_dir, method))
    prefix = os.path.join(m_dir, f"run_{run_id:02d}")

    # ---- 强制数值化 + 清洗 ----
    y = pd.to_numeric(pd.Series(np.asarray(y_te).reshape(-1)), errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(pd.Series(np.asarray(e_te).reshape(-1)), errors="coerce").fillna(0).to_numpy(dtype=int)
    r = pd.to_numeric(pd.Series(np.asarray(risk_scores_te).reshape(-1)), errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(y) & np.isfinite(r) & np.isfinite(e) & (y > 0)
    y, e, r = y[mask], e[mask], r[mask]

    if y.size < 3:
        csv_path = prefix + "_km.csv"
        pd.DataFrame({"time": [], "S_low": [], "S_high": []}).to_csv(csv_path, index=False)
        return {"csv": csv_path, "png": ""}

    # ---- 分组（中位数）----
    thr = float(np.nanmedian(r))
    hi = r >= thr
    lo = ~hi

    # Guard: if one group empty -> half split by sorted risk
    if hi.sum() == 0 or lo.sum() == 0:
        idx = np.argsort(r)
        mid = len(idx) // 2
        lo = np.zeros_like(r, dtype=bool)
        hi = np.zeros_like(r, dtype=bool)
        lo[idx[:mid]] = True
        hi[idx[mid:]] = True

    km_hi = KaplanMeierFitter()
    km_lo = KaplanMeierFitter()
    km_hi.fit(y[hi], event_observed=e[hi], label="High risk")
    km_lo.fit(y[lo], event_observed=e[lo], label="Low risk")

    # ---- 保存 survival table ----
    df_hi = (
        km_hi.survival_function_
        .rename(columns={"High risk": "S_high"})
        .reset_index()
        .rename(columns={"timeline": "time"})
    )
    df_lo = (
        km_lo.survival_function_
        .rename(columns={"Low risk": "S_low"})
        .reset_index()
        .rename(columns={"timeline": "time"})
    )
    df = pd.merge(df_lo, df_hi, on="time", how="outer").sort_values("time")
    csv_path = prefix + "_km.csv"
    df.to_csv(csv_path, index=False)

    # ---- log-rank test ----
    p_str, chi2_str = "nan", "nan"
    try:
        lr = logrank_test(
            durations_A=y[lo],
            durations_B=y[hi],
            event_observed_A=e[lo],
            event_observed_B=e[hi],
        )
        p = float(lr.p_value)
        chi2 = float(lr.test_statistic)
        p_str = f"{p:.2e}"
        chi2_str = f"{chi2:.2e}"
    except Exception:
        pass

    # ---- 绘图 ----
    png_path = prefix + "_km.png"
    if plt is not None:
        fig, ax = plt.subplots(figsize=(6, 4))

        # Low risk: green
        t_lo = df_lo["time"].to_numpy(dtype=float)
        s_lo = df_lo["S_low"].to_numpy(dtype=float)
        ax.step(t_lo, s_lo, where="post", label="Low risk", color='green')

        # High risk: red
        t_hi = df_hi["time"].to_numpy(dtype=float)
        s_hi = df_hi["S_high"].to_numpy(dtype=float)
        ax.step(t_hi, s_hi, where="post", label="High risk", color='Firebrick')

        dataset_ = dataset.replace('survset:', "")
        dataset_ = dataset_[0].upper() + dataset_[1:]

        ax.set_xlabel("Time", fontsize=15)
        ax.set_ylabel("Survival probability", fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=13)

        # Display log-rank p-value in bottom-right corner
        ax.text(
            0.98, 0.02,
            f"p={p_str}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            color="red",
            fontsize=14,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", linewidth=0)
        )

        fig.tight_layout()
        fig.savefig(png_path, dpi=300)
        plt.close(fig)

    return {"csv": csv_path, "png": png_path}





def build_split_data(
    df_feat: pd.DataFrame,
    y_all: np.ndarray,
    e_all: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    cfg: BenchConfig,
    add_censor: float = 0.0,
) -> SplitData:
    """
    Build SplitData with an option to INCREASE censoring ONLY in the training set.

    add_censor:
        - float in [0, 1]. Fraction of *currently event* samples (E==1) in TRAIN that will be
          converted into censored samples.
        - Validation/Test censoring ratio remains unchanged.
        - IMPORTANT: when an event is converted to censored, we also change its time y to a
          censoring time t_cens <= original y (as requested).

    How y is changed for newly-censored samples:
        - For selected event samples, draw u ~ Uniform(0, 1) and set
              y_new = max(eps_time, u * y_old)
          so that y_new <= y_old always, and is strictly positive.

    Notes:
        - This does NOT change the data split indices; it only perturbs (y_tr, e_tr) after split.
        - km_cens is fit on the modified training labels (consistent with training censoring).
    """
    require_lifelines()
    from lifelines import KaplanMeierFitter

    # --- slice labels ---
    y_tr = np.asarray(y_all[idx_tr], dtype=float).reshape(-1)
    e_tr = np.asarray(e_all[idx_tr], dtype=int).reshape(-1)

    y_va = np.asarray(y_all[idx_va], dtype=float).reshape(-1)
    e_va = np.asarray(e_all[idx_va], dtype=int).reshape(-1)

    y_te = np.asarray(y_all[idx_te], dtype=float).reshape(-1)
    e_te = np.asarray(e_all[idx_te], dtype=int).reshape(-1)

    # --- apply additional censoring to TRAIN only ---
    add_censor = float(add_censor)
    if add_censor < 0.0 or add_censor > 1.0:
        raise ValueError(f"add_censor must be in [0, 1], got {add_censor}")

    if add_censor > 0.0:
        event_idx = np.flatnonzero(e_tr == 1)
        if event_idx.size > 0:
            n_flip = int(np.round(add_censor * e_tr.size))
            if n_flip > 0:
                # deterministic if cfg.seed exists
                rng = np.random.default_rng(getattr(cfg, "seed", 24))
                flip_idx = rng.choice(event_idx, size=min(n_flip, event_idx.size), replace=False)

                y_tr = y_tr.copy()
                e_tr = e_tr.copy()

                # flip to censored
                e_tr[flip_idx] = 0

                # change time to censoring time <= original
                # u in (0,1), so y_new <= y_old always
                eps_time = 1e-8
                u = rng.random(size=flip_idx.size)
                y_new = u * y_tr[flip_idx]
                y_new = np.maximum(y_new, eps_time)
                # ensure y_new <= y_old (numerical safety)
                y_tr[flip_idx] = np.minimum(y_new, y_tr[flip_idx])

    # --- slice features ---
    df_tr = df_feat.iloc[idx_tr].reset_index(drop=True)
    df_va = df_feat.iloc[idx_va].reset_index(drop=True)
    df_te = df_feat.iloc[idx_te].reset_index(drop=True)

    X_tr, X_va, X_te, feat_names = fit_transform_survset_features(
        df_tr, df_va, df_te, max_categories=cfg.max_categories
    )

    # Fit censoring KM on (possibly modified) training labels
    km_cens = KaplanMeierFitter().fit(y_tr, event_observed=(1 - e_tr))

    # Time grid remains based on TEST follow-up and calibration quartiles.
    t_min = float(np.min(y_te))
    t_max = float(np.max(y_te))
    if t_max <= t_min:
        times = np.linspace(0.0, 1.0, cfg.n_time_grid).astype(np.float64)
    else:
        times = np.linspace(t_min, t_max, cfg.n_time_grid).astype(np.float64)
        train_event_times = y_tr[(e_tr == 1) & np.isfinite(y_tr)]
        if train_event_times.size:
            calibration_horizons = np.quantile(
                train_event_times, [0.25, 0.50, 0.75]
            ).astype(np.float64)
            supported_horizons = calibration_horizons[
                (calibration_horizons >= t_min) & (calibration_horizons < t_max)
            ]
            times = np.unique(np.concatenate([times, supported_horizons]))

    return SplitData(
        X_tr=X_tr, X_va=X_va, X_te=X_te, feat_names=feat_names,
        y_tr=y_tr, e_tr=e_tr, y_va=y_va, e_va=e_va, y_te=y_te, e_te=e_te,
        times=times, km_cens=km_cens,
        risk_time=getattr(cfg, "risk_time", "median"),
        n_samples=int(len(y_all)),
        n_features_raw=int(df_feat.shape[1]),
        # event_rate=float(np.mean(e_all)),
        event_rate=float(np.mean(np.concatenate([e_tr, e_va, e_te]))),
        
    )



# -----------------------------
# Printing helpers
# -----------------------------
def fmt_float(x: float) -> str:
    if x is None or (isinstance(x, float) and (not np.isfinite(x))):
        return "nan"
    return f"{x:.4f}"


def print_run_line(dataset: str, method: str, run_i: int, n_runs: int, seed: int,
                   cindex: float, ibs: float, auc_td: float, extra: str = "",
                   train_time_sec: float = None, bs_t25: float = None,
                   bs_t50: float = None, bs_t75: float = None) -> None:
    s = (
        f"[{dataset}] Run {run_i:02d}/{n_runs} seed={seed} | "
        f"{method}: C-index={fmt_float(cindex)}, "
        f"BS(t25)={fmt_float(bs_t25)}, BS(t50)={fmt_float(bs_t50)}, "
        f"BS(t75)={fmt_float(bs_t75)}, IBS={fmt_float(ibs)}, "
        f"tdAUC@t0={fmt_float(auc_td)}"
        + (f", Train(s)={fmt_float(train_time_sec)}" if train_time_sec is not None else "")
    )
    if extra:
        s += f" | {extra}"
    print(s)


def summarize_metrics(xs: List[float]) -> Tuple[float, float]:
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan"), float("nan")
    if len(a) == 1:
        return float(a.mean()), 0.0
    return float(a.mean()), float(a.std(ddof=1))





from decimal import Decimal, ROUND_HALF_UP
from typing import List

def _round_half_up(x: float, ndigits: int) -> float:
    q = Decimal("1." + "0" * ndigits)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))

def _fmt_mean_std(mean: float, std: float) -> str:
    m = _round_half_up(mean, 3)
    s = _round_half_up(std, 4)
    return f"{m:.3f}±{s:.4f}"

def print_method_summary(
    dataset: str,
    method: str,
    c_list: List[float],
    i_list: List[float],
    a_list: List[float],
    t_list: Optional[List[float]] = None,
    note: str = "",
    bs_t25_list: Optional[List[float]] = None,
    bs_t50_list: Optional[List[float]] = None,
    bs_t75_list: Optional[List[float]] = None,
) -> None:
    c_m, c_s = summarize_metrics(c_list)
    i_m, i_s = summarize_metrics(i_list)
    a_m, a_s = summarize_metrics(a_list)
    b25_m, b25_s = summarize_metrics(bs_t25_list or [])
    b50_m, b50_s = summarize_metrics(bs_t50_list or [])
    b75_m, b75_s = summarize_metrics(bs_t75_list or [])
    t_m, t_s = (float('nan'), float('nan'))
    if t_list is not None:
        t_m, t_s = summarize_metrics(t_list)

    print(f"\n===== Summary ({dataset} | {method}) =====")
    if note:
        print(note)
    print(f"{method:6s} C-index: {_fmt_mean_std(c_m, c_s)}")
    print(f"{method:6s} BS(t25): {_fmt_mean_std(b25_m, b25_s)}")
    print(f"{method:6s} BS(t50): {_fmt_mean_std(b50_m, b50_s)}")
    print(f"{method:6s} BS(t75): {_fmt_mean_std(b75_m, b75_s)}")
    print(f"{method:6s} IBS    : {_fmt_mean_std(i_m, i_s)}")
    print(f"{method:6s} tdAUC@t0: {_fmt_mean_std(a_m, a_s)}")
    if t_list is not None:
        print(f"{method:6s} Train(s): {_fmt_mean_std(t_m, t_s)}")
    print("====================================\n")


def _format_overall_summary_table(df_res: pd.DataFrame) -> pd.DataFrame:
        """Return a display DataFrame with mean±SD columns while keeping df_res intact."""
        df_show = df_res.copy()

        # Build mean±SD columns.
        df_show["cindex"] = df_show.apply(lambda r: _fmt_mean_std(r["cindex_mean"], r["cindex_std"]), axis=1)
        df_show["bs_t25"] = df_show.apply(lambda r: _fmt_mean_std(r["bs_t25_mean"], r["bs_t25_std"]), axis=1)
        df_show["bs_t50"] = df_show.apply(lambda r: _fmt_mean_std(r["bs_t50_mean"], r["bs_t50_std"]), axis=1)
        df_show["bs_t75"] = df_show.apply(lambda r: _fmt_mean_std(r["bs_t75_mean"], r["bs_t75_std"]), axis=1)
        df_show["ibs"] = df_show.apply(lambda r: _fmt_mean_std(r["ibs_mean"], r["ibs_std"]), axis=1)
        df_show["td_auc_t0"] = df_show.apply(lambda r: _fmt_mean_std(r["td_auc_t0_mean"], r["td_auc_t0_std"]), axis=1)
        if "train_time_mean" in df_show.columns and "train_time_std" in df_show.columns:
            df_show["train_time"] = df_show.apply(lambda r: _fmt_mean_std(r["train_time_mean"], r["train_time_std"]), axis=1)

        # keep a clean column order (you can adjust if you want)
        keep_cols = [
            "dataset", "method", "n_samples", "n_features_raw", "event_rate",
            "cindex", "bs_t25", "bs_t50", "bs_t75", "ibs",
            "td_auc_t0", "train_time", "note"
        ]
        # only keep those exist (robust)
        keep_cols = [c for c in keep_cols if c in df_show.columns]
        return df_show[keep_cols]

# -----------------------------
# Main benchmark loop (SurvSet static)
# -----------------------------
def run_rdsurv_survset_benchmark(
    datasets: List[str],
    bench_cfg: BenchConfig,
    method_cfg: Any,
    add_censor: float = 0.0,
    time_scale: Optional[str] = None
) -> pd.DataFrame:
    """
    Run the sole supported method, RDSurv, and return a long-format DataFrame.
    """
    require_survset()
    require_lifelines()

    from SurvSet.data import SurvLoader

    from . import rd_surv as rd_surv_mod

    methods = ["RDSurv"]

    loader = SurvLoader()
    rows: List[Dict[str, Any]] = []

    for di, ds_name in enumerate(datasets, 1):
        dataset_tag = f"survset:{ds_name}"
        print(f"\n========== [{di}/{len(datasets)}] Dataset {dataset_tag} ==========")

        pack = loader.load_dataset(ds_name=ds_name)
        df = pack["df"].copy() if isinstance(pack, dict) and "df" in pack else pack.copy()

        # static only: drop time2 if present
        if "time2" in df.columns:
            df = df.drop(columns=["time2"])

        if not {"event", "time"}.issubset(df.columns):
            print(f"[WARN] {dataset_tag} missing event/time; skipped.")
            continue

        y_all = pd.to_numeric(df["time"], errors="coerce").astype(float).to_numpy().astype(np.float32)
        e_all = pd.to_numeric(df["event"], errors="coerce").fillna(0).astype(int).to_numpy().astype(np.int64)

        if time_scale == 'log':
            y_all = np.log(y_all + 1)
        elif time_scale == 'sqrt':
            y_all = np.sqrt(y_all)

        feat_cols = [c for c in df.columns if c.startswith("num_") or c.startswith("fac_")]
        if len(feat_cols) == 0:
            print(f"[WARN] {dataset_tag} has no num_/fac_ features; skipped.")
            continue

        df_feat = df[feat_cols].copy()

        per_method_c = {m: [] for m in methods}
        per_method_bs25 = {m: [] for m in methods}
        per_method_bs50 = {m: [] for m in methods}
        per_method_bs75 = {m: [] for m in methods}
        per_method_i = {m: [] for m in methods}
        per_method_a = {m: [] for m in methods}
        per_method_t = {m: [] for m in methods}  # train time (sec), TRAIN only

        for r in range(bench_cfg.n_runs):
            seed = bench_cfg.seed0 + r
            set_seed_np(seed)


            idx = np.arange(len(df))
            idx_trval, idx_te = safe_train_test_split(idx, e_all, test_size=bench_cfg.test_size, seed=seed)
            e_trval = e_all[idx_trval]
            idx_tr, idx_va = safe_train_test_split(idx_trval, e_trval, test_size=bench_cfg.val_size, seed=seed)

            split = build_split_data(df_feat, y_all, e_all, idx_tr, idx_va, idx_te, bench_cfg,add_censor)

            t0 = _median_train_event_time(split.y_tr, split.e_tr)

            for m in methods:
                set_seed_every(seed)
                res = rd_surv_mod.fit_predict(split, method_cfg)

                cidx, ibs, extra = res.cindex, res.ibs, res.note
                bs_t25 = float("nan")
                bs_t50 = float("nan")
                bs_t75 = float("nan")
                if res.surv_te is not None:
                    try:
                        brier_metrics = compute_ipcw_brier_metrics(
                            prediction_times=split.times,
                            y_train=split.y_tr,
                            e_train=split.e_tr,
                            y_test=split.y_te,
                            e_test=split.e_te,
                            surv_preds=res.surv_te,
                            censor_kmf=split.km_cens,
                        )
                        bs_t25 = brier_metrics["bs_t25"]
                        bs_t50 = brier_metrics["bs_t50"]
                        bs_t75 = brier_metrics["bs_t75"]
                        ibs = brier_metrics["ibs"]
                        res.bs_t25 = bs_t25
                        res.bs_t50 = bs_t50
                        res.bs_t75 = bs_t75
                        res.ibs = ibs
                    except Exception as exc:
                        print(
                            f"[WARN] {dataset_tag} {m}: shared Brier evaluation "
                            f"failed: {exc!r}"
                        )

                auc_td = float("nan")
                if res.risk_scores_te is not None:
                    auc_td = compute_time_dependent_auc_at_t0(
                        split.y_tr, split.e_tr,
                        split.y_te, split.e_te,
                        res.risk_scores_te,
                        t0,
                    )
                if (
                    bench_cfg.save_calibration_predictions
                    and res.surv_te is not None
                ):
                    calibration_root = _ensure_dir(
                        os.path.join(
                            bench_cfg.results_dir,
                            bench_cfg.calibration_predictions_subdir,
                        )
                    )
                    save_calibration_predictions(
                        prediction_times=split.times,
                        surv_preds=res.surv_te,
                        y_train=split.y_tr,
                        e_train=split.e_tr,
                        y_test=split.y_te,
                        e_test=split.e_te,
                        sample_ids=idx_te,
                        dataset=dataset_tag,
                        method=m,
                        run_id=r + 1,
                        seed=seed,
                        out_dir=calibration_root,
                    )
                # Save high/low-risk KM curves on TEST for each run
                if bench_cfg.save_risk_curves and res.risk_scores_te is not None:
                    curve_root = _ensure_dir(os.path.join(bench_cfg.results_dir, bench_cfg.risk_curves_subdir))
                    save_high_low_risk_km_curves(
                        y_te=split.y_te,
                        e_te=split.e_te,
                        risk_scores_te=res.risk_scores_te,
                        dataset=dataset_tag,
                        method=m,
                        run_id=r + 1,
                        out_dir=curve_root,
                    )

                per_method_c[m].append(float(cidx) if cidx is not None else float("nan"))
                per_method_bs25[m].append(float(bs_t25))
                per_method_bs50[m].append(float(bs_t50))
                per_method_bs75[m].append(float(bs_t75))
                per_method_i[m].append(float(ibs) if ibs is not None else float("nan"))
                per_method_a[m].append(float(auc_td) if auc_td is not None else float("nan"))
                per_method_t[m].append(float(res.train_time_sec) if getattr(res, 'train_time_sec', None) is not None else float('nan'))
                print_run_line(
                    dataset_tag, m, r + 1, bench_cfg.n_runs, seed,
                    cidx, ibs, auc_td, extra,
                    train_time_sec=getattr(res, 'train_time_sec', None),
                    bs_t25=bs_t25, bs_t50=bs_t50, bs_t75=bs_t75,
                )

        for m in methods:
            c_m, c_s = summarize_metrics(per_method_c[m])
            b25_m, b25_s = summarize_metrics(per_method_bs25[m])
            b50_m, b50_s = summarize_metrics(per_method_bs50[m])
            b75_m, b75_s = summarize_metrics(per_method_bs75[m])
            i_m, i_s = summarize_metrics(per_method_i[m])
            a_m, a_s = summarize_metrics(per_method_a[m])
            t_m, t_s = summarize_metrics(per_method_t[m])

            note = ""
            if m == "RDSurv":
                f0m = getattr(method_cfg, "f0_method", None)
                note = f"F0_METHOD={f0m}" if f0m else ""

            print_method_summary(
                dataset_tag, m, per_method_c[m], per_method_i[m],
                per_method_a[m], per_method_t[m], note=note,
                bs_t25_list=per_method_bs25[m],
                bs_t50_list=per_method_bs50[m],
                bs_t75_list=per_method_bs75[m],
            )

            rows.append({
                "dataset": dataset_tag,
                "method": m,
                "n_samples": split.n_samples,
                "n_features_raw": split.n_features_raw,
                "event_rate": split.event_rate,
                "cindex_mean": c_m,
                "cindex_std": c_s,
                "bs_t25_mean": b25_m,
                "bs_t25_std": b25_s,
                "bs_t50_mean": b50_m,
                "bs_t50_std": b50_s,
                "bs_t75_mean": b75_m,
                "bs_t75_std": b75_s,
                "ibs_mean": i_m,
                "ibs_std": i_s,
                "td_auc_t0_mean": a_m,
                "td_auc_t0_std": a_s,
                "train_time_mean": t_m,
                "train_time_std": t_s,
                "note": note,
            })

    if not rows:
        print("No dataset finished successfully.")
        return pd.DataFrame()

    df_res = pd.DataFrame(rows).sort_values(["dataset", "method"]).reset_index(drop=True)




    print("\n==================== Overall Summary (long format) ====================")
    df_show = _format_overall_summary_table(df_res)

    with pd.option_context("display.max_rows", 500, "display.max_columns", 50, "display.width", 180):
        # event_rate 仍然可以用原格式输出；如果你也要 event_rate 3位小数我也可以改
        print(df_show.to_string(index=False))
    print("======================================================================\n")


    # save metrics
    try:
        _ensure_dir(bench_cfg.results_dir)
        df_res.to_csv(os.path.join(bench_cfg.results_dir, "metrics_long.csv"), index=False)
    except Exception:
        pass

    return df_res
