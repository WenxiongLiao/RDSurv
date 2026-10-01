# -*- coding: utf-8 -*-
"""Reproducible SurvSet experiments for RDSurv.

Run with ``python main.py``. Experiment settings can be overridden with
environment variables; the defaults reproduce the five static datasets and
sqrt time scaling used by the accompanying paper.
"""

import os

from surv_bench.common import (
    BenchConfig,
    resolve_survset_datasets,
    run_rdsurv_survset_benchmark,
)
from surv_bench.rd_surv import RDSurvConfig


def main() -> None:
    datasets = resolve_survset_datasets(
        env_datasets=os.environ.get("DATASETS", ""),
        use_all_static=os.environ.get("USE_ALL_STATIC", "0").lower() in {"1", "true"},
    )

    bench_cfg = BenchConfig(
        seed0=int(os.environ.get("SEED0", "42")),
        n_runs=int(os.environ.get("N_RUNS", "10")),
        test_size=float(os.environ.get("TEST_SIZE", "0.2")),
        val_size=float(os.environ.get("VAL_SIZE", "0.25")),
        n_time_grid=int(os.environ.get("N_TIME_GRID", "100")),
        max_categories=int(os.environ.get("MAX_CATEGORIES", "0")),
        results_dir=os.environ.get("RESULTS_DIR", "results"),
        save_risk_curves=os.environ.get("SAVE_RISK_CURVES", "1").lower() not in {"0", "false"},
        save_calibration_predictions=os.environ.get("SAVE_CALIBRATION", "1").lower() not in {"0", "false"},
    )

    method_cfg = RDSurvConfig(
        hidden_dim=int(os.environ.get("HIDDEN_DIM", "32")),
        lr=float(os.environ.get("LR", "3e-4")),
        weight_decay=float(os.environ.get("WEIGHT_DECAY", "1e-4")),
        batch_size=int(os.environ.get("BATCH_SIZE", "128")),
        max_epochs=int(os.environ.get("MAX_EPOCHS", "512")),
        patience=int(os.environ.get("PATIENCE", "50")),
        device=os.environ.get("DEVICE", "cpu"),
        f0_method=os.environ.get("F0_METHOD", "ipw_km"),
        ipw_censor_penalizer=float(os.environ.get("IPW_CENSOR_PENALIZER", "0.01")),
        ipw_min_censor_survival=float(os.environ.get("IPW_MIN_G", "0.001")),
        ipw_max_weight=float(os.environ.get("IPW_MAX_WEIGHT", "50.0")),
    )

    run_rdsurv_survset_benchmark(
        datasets=datasets,
        bench_cfg=bench_cfg,
        method_cfg=method_cfg,
        add_censor=0.0,
        time_scale=os.environ.get("TIME_SCALE", "sqrt"),
    )


if __name__ == "__main__":
    main()
