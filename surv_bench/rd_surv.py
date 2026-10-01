# -*- coding: utf-8 -*-
"""
RDSurv (LogitNormal in percentile space) method module.

Interface:
  fit_predict(split: SplitData, cfg: RDSurvConfig) -> (cindex, ibs, extra_str)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import time
from typing import Tuple, Any, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gc
from .common import require_lifelines, compute_ipcw_ibs, SplitData, km_predict_1d,MethodResult





# -----------------------------
# F0 estimation
# -----------------------------
def na_cumhaz_1d(naf, times: np.ndarray) -> np.ndarray:
    ch = naf.cumulative_hazard_at_times(times)
    if isinstance(ch, pd.Series):
        return ch.to_numpy().reshape(-1)
    if isinstance(ch, pd.DataFrame):
        return ch.iloc[:, 0].to_numpy().reshape(-1)
    return np.asarray(ch).reshape(-1)


@dataclass
class IPWKaplanMeierEstimate:
    event_times: np.ndarray
    survival: np.ndarray
    censor_penalizer: Optional[float]
    min_censor_survival: float
    max_weight: Optional[float]
    max_observed_weight: float


def _predict_step_survival(est: IPWKaplanMeierEstimate, times: np.ndarray) -> np.ndarray:
    query_times = np.asarray(times, dtype=float).reshape(-1)
    out = np.ones(query_times.shape[0], dtype=np.float64)
    if est.event_times.size == 0:
        return out

    indices = np.searchsorted(est.event_times, query_times, side="right") - 1
    observed = indices >= 0
    out[observed] = est.survival[indices[observed]]
    return np.clip(out, 0.0, 1.0)


def _fit_censoring_cox(
    y_train: np.ndarray,
    e_train: np.ndarray,
    X_train: np.ndarray,
    penalizer: float,
):
    from lifelines import CoxPHFitter

    X = np.asarray(X_train, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or X.shape[0] != y_train.shape[0]:
        raise ValueError(
            "X_train must be a 2D array with the same number of rows as y_train."
        )

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.std(X, axis=0) > 1e-12
    X = X[:, keep]
    if X.shape[1] == 0:
        return None, None, None

    columns = [f"x{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=columns)
    train_df = X_df.copy()
    train_df["__time"] = y_train
    train_df["__censored"] = 1 - e_train

    penalizers = []
    for candidate in (float(penalizer), 0.1, 1.0):
        if candidate < 0:
            raise ValueError("ipw_censor_penalizer must be non-negative.")
        if candidate not in penalizers:
            penalizers.append(candidate)

    errors = []
    for candidate in penalizers:
        try:
            model = CoxPHFitter(penalizer=candidate)
            model.fit(
                train_df,
                duration_col="__time",
                event_col="__censored",
                show_progress=False,
            )
            return model, X_df, candidate
        except Exception as exc:
            errors.append(f"penalizer={candidate:g}: {exc!r}")

    raise RuntimeError(
        "The censoring Cox model for IPW-KM failed to converge. Tried: "
        + " | ".join(errors)
    )


def _fit_ipw_km(
    y_train: np.ndarray,
    e_train: np.ndarray,
    X_train: np.ndarray,
    censor_penalizer: float,
    min_censor_survival: float,
    max_weight: Optional[float],
) -> IPWKaplanMeierEstimate:
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    e = np.asarray(e_train, dtype=np.int64).reshape(-1)
    if y.shape[0] != e.shape[0]:
        raise ValueError("y_train and e_train must have the same length.")
    if y.size == 0:
        raise ValueError("Cannot fit IPW-KM on an empty training set.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y_train contains non-finite values.")
    if not np.all(np.isin(e, (0, 1))):
        raise ValueError("e_train must contain only 0/1 event indicators.")
    if not (0.0 < float(min_censor_survival) <= 1.0):
        raise ValueError("ipw_min_censor_survival must be in (0, 1].")
    if max_weight is not None and float(max_weight) < 1.0:
        raise ValueError("ipw_max_weight must be at least 1 or None.")

    event_times = np.unique(y[e == 1])
    event_times.sort()
    if event_times.size == 0:
        return IPWKaplanMeierEstimate(
            event_times=event_times,
            survival=np.empty(0, dtype=np.float64),
            censor_penalizer=None,
            min_censor_survival=float(min_censor_survival),
            max_weight=max_weight,
            max_observed_weight=1.0,
        )

    censor_model = None
    X_df = None
    used_penalizer = None
    if np.any(e == 0):
        if X_train is None:
            raise ValueError(
                "X_train is required when method='ipw_km' so that G(t|X) "
                "can adjust for covariate-dependent censoring."
            )
        censor_model, X_df, used_penalizer = _fit_censoring_cox(
            y, e, X_train, penalizer=censor_penalizer
        )

    if censor_model is None:
        baseline_times = np.empty(0, dtype=np.float64)
        baseline_cumhaz = np.empty(0, dtype=np.float64)
        relative_hazard = np.ones(y.shape[0], dtype=np.float64)
    else:
        baseline = censor_model.baseline_cumulative_hazard_
        baseline_times = baseline.index.to_numpy(dtype=np.float64)
        baseline_cumhaz = baseline.iloc[:, 0].to_numpy(dtype=np.float64)
        relative_hazard = (
            censor_model.predict_partial_hazard(X_df)
            .to_numpy(dtype=np.float64)
            .reshape(-1)
        )
        relative_hazard = np.nan_to_num(
            relative_hazard, nan=1.0, posinf=1e12, neginf=1e-12
        )
        relative_hazard = np.clip(relative_hazard, 1e-12, 1e12)

    weight_cap = 1.0 / float(min_censor_survival)
    if max_weight is not None:
        weight_cap = min(weight_cap, float(max_weight))
    max_log_weight = math.log(weight_cap)

    survival_values = np.empty(event_times.shape[0], dtype=np.float64)
    current_survival = 1.0
    max_observed_weight = 1.0
    previous_baseline_index = None
    weights = np.ones(y.shape[0], dtype=np.float64)

    for j, event_time in enumerate(event_times):
        baseline_index = (
            np.searchsorted(baseline_times, event_time, side="left") - 1
            if baseline_times.size
            else -1
        )
        if baseline_index != previous_baseline_index:
            cumulative_hazard = (
                baseline_cumhaz[baseline_index] if baseline_index >= 0 else 0.0
            )
            log_weights = np.minimum(
                cumulative_hazard * relative_hazard, max_log_weight
            )
            weights = np.exp(log_weights)
            max_observed_weight = max(
                max_observed_weight, float(np.max(weights))
            )
            previous_baseline_index = baseline_index

        at_risk = y >= event_time
        events = (y == event_time) & (e == 1)
        weighted_risk = float(np.sum(weights[at_risk]))
        weighted_events = float(np.sum(weights[events]))
        if weighted_risk > 0.0:
            hazard = np.clip(weighted_events / weighted_risk, 0.0, 1.0)
            current_survival *= 1.0 - hazard
        survival_values[j] = current_survival

    return IPWKaplanMeierEstimate(
        event_times=event_times,
        survival=np.minimum.accumulate(survival_values),
        censor_penalizer=used_penalizer,
        min_censor_survival=float(min_censor_survival),
        max_weight=max_weight,
        max_observed_weight=max_observed_weight,
    )


def fit_f0_estimator(
    y_train: np.ndarray,
    e_train: np.ndarray,
    method: str,
    X_train: Optional[np.ndarray] = None,
    ipw_censor_penalizer: float = 0.01,
    ipw_min_censor_survival: float = 1e-3,
    ipw_max_weight: Optional[float] = 50.0,
):
    """Estimate the marginal reference chart with KM, NA, or covariate-adjusted IPW-KM."""
    require_lifelines()
    from lifelines import KaplanMeierFitter, NelsonAalenFitter

    m = (method or "km").strip().lower()
    if m in ("km", "kaplan-meier", "kaplan_meier", "kaplanmeier"):
        kmf = KaplanMeierFitter().fit(y_train, event_observed=e_train)
        return ("km", kmf)
    if m in ("na", "nelson-aalen", "nelson_aalen", "nelsonaalen"):
        naf = NelsonAalenFitter().fit(y_train, event_observed=e_train)
        return ("na", naf)
    if m in ("ipw_km", "ipw-km", "ipcw_km", "ipcw-km"):
        ipw_est = _fit_ipw_km(
            y_train,
            e_train,
            X_train,
            censor_penalizer=ipw_censor_penalizer,
            min_censor_survival=ipw_min_censor_survival,
            max_weight=ipw_max_weight,
        )
        return ("ipw_km", ipw_est)
    raise ValueError(
        f"Unknown f0_method='{method}'. Use 'km', 'na', or 'ipw_km'."
    )


def predict_S0_1d(f0_est, times: np.ndarray) -> np.ndarray:
    kind, est = f0_est
    if kind == "km":
        return km_predict_1d(est, times)
    if kind == "na":
        H = na_cumhaz_1d(est, times)
        return np.exp(-H)
    if kind == "ipw_km":
        return _predict_step_survival(est, times)
    raise ValueError(f"Unknown f0 estimator kind: {kind}")


def predict_F0_1d(f0_est, times: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    S0 = predict_S0_1d(f0_est, times)
    F0 = 1.0 - S0
    return np.clip(F0, eps, 1.0 - eps)


# -----------------------------
# Model
# -----------------------------
class RDSurvNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        if hidden_dim <= 0:
            self.backbone = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, 1),
            )
        else:
            self.backbone = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        self.log_s = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor):
        m = self.backbone(x).squeeze(-1)
        s = F.softplus(self.log_s) + 1e-4
        return m, s



class _SurvDS(Dataset):
    def __init__(self, X: np.ndarray, z: np.ndarray, u: np.ndarray, e: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.z = torch.from_numpy(z).float()
        self.u = torch.from_numpy(u).float()
        self.e = torch.from_numpy(e).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return self.X[i], self.z[i], self.u[i], self.e[i]


def normal_logpdf(z: torch.Tensor, m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return -0.5 * math.log(2 * math.pi) - torch.log(s) - 0.5 * ((z - m) / s) ** 2


def normal_logcdf(x: torch.Tensor) -> torch.Tensor:
    return torch.special.log_ndtr(x)


def rd_surv_nll(m: torch.Tensor, s: torch.Tensor, z: torch.Tensor, u: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    log_pdf_u = normal_logpdf(z, m, s) - torch.log(u) - torch.log1p(-u)
    t = (m - z) / s
    log_surv_u = normal_logcdf(t)
    ll = torch.where(e > 0, log_pdf_u, log_surv_u)
    return -ll.mean()


@torch.no_grad()
def predict_survival(model: RDSurvNet, X: np.ndarray, F0_times: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    X_t = torch.from_numpy(X).float().to(device)
    m, s = model(X_t)
    m = m.unsqueeze(1)

    eps = 1e-6
    u_t = np.clip(F0_times, eps, 1 - eps)
    z_t = torch.from_numpy(np.log(u_t) - np.log1p(-u_t)).float().to(device).unsqueeze(0)

    t = (m - z_t) / s
    S = 0.5 * (1.0 + torch.erf(t / math.sqrt(2.0)))
    return S.cpu().numpy()


@torch.no_grad()
def risk_score(model: RDSurvNet, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    X_t = torch.from_numpy(X).float().to(device)
    m, _s = model(X_t)
    return torch.sigmoid(m).cpu().numpy().reshape(-1)


@dataclass
class RDSurvConfig:
    hidden_dim: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 300
    patience: int = 30
    device: str = "cpu"
    f0_method: str = "ipw_km"  # 'km', 'na', or 'ipw_km'
    ipw_censor_penalizer: float = 0.01
    ipw_min_censor_survival: float = 1e-3
    ipw_max_weight: Optional[float] = 50.0


def train(split: SplitData, cfg: RDSurvConfig) -> Tuple[RDSurvNet, dict, Any]:
    eps = 1e-6
    f0_est = fit_f0_estimator(
        split.y_tr,
        split.e_tr,
        method=cfg.f0_method,
        X_train=split.X_tr,
        ipw_censor_penalizer=cfg.ipw_censor_penalizer,
        ipw_min_censor_survival=cfg.ipw_min_censor_survival,
        ipw_max_weight=cfg.ipw_max_weight,
    )

    u_tr = predict_F0_1d(f0_est, split.y_tr, eps=eps)
    z_tr = np.log(u_tr) - np.log1p(-u_tr)

    u_va = predict_F0_1d(f0_est, split.y_va, eps=eps)
    z_va = np.log(u_va) - np.log1p(-u_va)

    ds_tr = _SurvDS(split.X_tr, z_tr, u_tr, split.e_tr)
    ds_va = _SurvDS(split.X_va, z_va, u_va, split.e_va)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = RDSurvNet(in_dim=split.X_tr.shape[1], hidden_dim=cfg.hidden_dim).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_state = None
    bad = 0
    epoch = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for xb, zb, ub, eb in dl_tr:
            xb, zb, ub, eb = xb.to(cfg.device), zb.to(cfg.device), ub.to(cfg.device), eb.to(cfg.device)
            m, s = model(xb)
            loss = rd_surv_nll(m, s, zb, ub, eb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        va_losses = []
        for xb, zb, ub, eb in dl_va:
            xb, zb, ub, eb = xb.to(cfg.device), zb.to(cfg.device), ub.to(cfg.device), eb.to(cfg.device)
            m, s = model(xb)
            va_losses.append(float(rd_surv_nll(m, s, zb, ub, eb).item()))

        val_loss = float(np.mean(va_losses)) if va_losses else float("inf")
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    del opt
    del loss
    torch.cuda.empty_cache()
    gc.collect()
    info = {"best_val_nll": best_val, "epochs": epoch}
    return model, info, f0_est


def fit_predict(split: SplitData, cfg: Optional[RDSurvConfig]) -> MethodResult:
    require_lifelines()
    from lifelines.utils import concordance_index

    if cfg is None:
        cfg = RDSurvConfig()

    t0_train = time.perf_counter()
    model, info, f0_est = train(split, cfg)
    train_time_sec = time.perf_counter() - t0_train

    F0_times = predict_F0_1d(f0_est, split.times, eps=1e-6).astype(np.float64)
    S_te = predict_survival(model, split.X_te, F0_times, device=cfg.device)
    score_longer = risk_score(model, split.X_te, device=cfg.device)
    risk = -score_longer

    cidx = float(concordance_index(split.y_te, score_longer, split.e_te))
    ibs = float(compute_ipcw_ibs(split.times, split.y_te, split.e_te, S_te, split.km_cens))

    extra = f"F0_METHOD={cfg.f0_method}, best_val_nll={info['best_val_nll']:.4f}, epochs={info['epochs']}"
    if f0_est[0] == "ipw_km":
        ipw_est = f0_est[1]
        extra += (
            f", censor_penalizer={ipw_est.censor_penalizer}, "
            f"max_ipw={ipw_est.max_observed_weight:.3f}"
        )
    del model
    return MethodResult(cindex=cidx, ibs=ibs, note=extra, train_time_sec=train_time_sec, risk_scores_te=risk, surv_te=S_te)
