# RDSurv

Reference implementation of **Rank-Distortion Survival (RDSurv)** for
right-censored survival analysis.

This repository accompanies the paper:

> Wenxiong Liao, Qian Wu, and Fei Wang.  
> *Rank-Distortion Survival: Time-Scale Invariant Survival Modeling on
> Percentile Geometry.*

RDSurv models failure percentiles instead of absolute event times. A baseline
percentile chart maps observed time to the unit interval, a logit transform
maps percentiles to the real line, and a neural network learns a scalar
covariate-dependent distortion coordinate.



## Repository Structure

```text
.
├── main.py                  # Main benchmark entry point
├── requirements.txt         # Pinned Python dependencies
├── surv_bench/
│   ├── common.py            # Data loading, preprocessing, metrics, outputs
│   └── rd_surv.py           # RDSurv model and training code
├── KBS_RDSurv.pdf           # Accompanying paper
└── results/                 # Existing benchmark outputs
```

## Requirements

- CPython 3.10.x
- The packages and versions listed in `requirements.txt`
- A working PyTorch installation for the target CPU or GPU

The benchmark uses the static datasets provided by the `SurvSet` package.

## Installation

Create and activate a virtual environment, then install the pinned
dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick Start

Run the default RDSurv benchmark:

```bash
python main.py
```

The default configuration runs 10 repetitions on:

```text
NSBCD, hdfail, nki70, support2, UnempDur
```


To run a single dataset:

```bash
DATASETS=nki70 python main.py
```

To preserve outputs from separate experiments, use a different results
directory:

```bash
RESULTS_DIR=results/nki70_sqrt DATASETS=nki70 python main.py
```

## Time-Scale Experiments

The paper evaluates the original, logarithmic, and square-root time scales.
Run them separately:

```bash
# Original time scale
TIME_SCALE=original python main.py

# log(T + 1)
TIME_SCALE=log python main.py

# sqrt(T)
TIME_SCALE=sqrt python main.py
```

Only the values `log` and `sqrt` trigger transformations in the current
implementation. Any other value, including `original`, keeps the original
time scale.

## Configuration

All configuration is controlled through environment variables. The most
important options are listed below.

### Dataset and benchmark options

| Variable | Default | Description |
|---|---:|---|
| `DATASETS` | empty | Comma-separated dataset names, for example `nki70,support2` |
| `USE_ALL_STATIC` | `0` | Discover static SurvSet datasets when set to `1` |
| `SEED0` | `42` | Seed of the first run |
| `N_RUNS` | `10` | Number of repeated runs |
| `TEST_SIZE` | `0.2` | Test fraction |
| `VAL_SIZE` | `0.25` | Validation fraction within the train-validation split |
| `N_TIME_GRID` | `100` | Number of prediction grid points |
| `MAX_CATEGORIES` | `0` | Maximum allowed categorical cardinality; `0` keeps all |
| `TIME_SCALE` | `sqrt` | `original`, `log`, or `sqrt` |
| `RESULTS_DIR` | `results` | Output directory |
| `SAVE_RISK_CURVES` | `1` | Save high/low-risk Kaplan-Meier curves |
| `SAVE_CALIBRATION` | `1` | Save calibration prediction files |

### Model options

| Variable | Default | Description |
|---|---:|---|
| `HIDDEN_DIM` | `32` | Hidden width of the RDSurv MLP |
| `LR` | `3e-4` | AdamW learning rate |
| `WEIGHT_DECAY` | `1e-4` | AdamW weight decay |
| `BATCH_SIZE` | `128` | Training batch size |
| `MAX_EPOCHS` | `512` | Maximum training epochs |
| `PATIENCE` | `50` | Validation early-stopping patience |
| `DEVICE` | `cpu` | PyTorch device, for example `cuda` |
| `F0_METHOD` | `ipw_km` | `ipw_km`, `km`, or `na` |
| `IPW_CENSOR_PENALIZER` | `0.01` | Cox censoring-model penalizer |
| `IPW_MIN_G` | `0.001` | Lower bound for censoring survival |
| `IPW_MAX_WEIGHT` | `50.0` | Maximum IPW weight |

Example:

```bash
DATASETS=NSBCD,nki70 \
N_RUNS=10 \
SEED0=42 \
TIME_SCALE=original \
F0_METHOD=ipw_km \
python main.py
```

## Data Processing

The loader expects the naming convention used by `SurvSet`:

- `num_*` columns are treated as numerical features;
- `fac_*` columns are treated as categorical features;
- categorical variables are imputed and one-hot encoded;
- numerical variables are median-imputed and standardized;
- preprocessing objects are fitted on the training split only;
- `time2` is removed when present because this benchmark uses static data.

The baseline percentile chart and the censoring distribution used for
evaluation are also estimated from training data.

## Evaluation

The benchmark reports:

- C-index for global risk ordering;
- time-dependent AUC at the median event time of the training split;
- IPCW Brier Score at the 25th, 50th, and 75th percentiles of training event
  times;
- IBS over the available evaluation grid;
- training time.

The paper emphasizes `BS(t75)`. The implementation also records `BS(t25)`,
`BS(t50)`, and IBS for additional diagnostics.

Risk-group Kaplan-Meier curves are created by splitting the test set at the
median predicted risk score. Calibration prediction files are saved at the
training event-time quartiles.

## Output Files

With the default `RESULTS_DIR=results`, the main outputs are:

```text
results/
├── metrics_long.csv
├── risk_curves/
│   └── survset:<dataset>/RDSurv/
│       ├── run_01_km.csv
│       └── run_01_km.png
└── calibration_predictions/
    └── survset:<dataset>/RDSurv/
        └── run_01_calibration.csv
```

`metrics_long.csv` contains one summary row per dataset and method. Individual
run outputs include the seed and run identifier.

## Reproducibility Notes

The benchmark sets Python, NumPy, and PyTorch random seeds and enables
deterministic cuDNN settings where applicable. Results can still vary across
operating systems, CPU/GPU backends, and dependency versions.

The current repository contains the RDSurv implementation and its benchmark
pipeline. It does not contain implementations of the comparison methods
listed in the paper, such as CoxPH, Weibull AFT, DeepHit, DSM, and UniSurv.
Those methods are therefore not run by `main.py`.

## Citation

If you use this code or the RDSurv method, please cite:

```text
Wenxiong Liao, Qian Wu, and Fei Wang.
Rank-Distortion Survival: Time-Scale Invariant Survival Modeling on
Percentile Geometry.
```

## License

No license file is currently included. Please add the license selected by the
authors before distributing the repository publicly.
