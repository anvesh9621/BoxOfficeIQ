# BoxOfficeIQ — Machine Learning Methodology

> **Version:** 1.0  
> **Last Updated:** 2024

---

## Table of Contents

1. [Problem Definition](#problem-definition)
2. [Data Split Strategy](#data-split-strategy)
3. [Feature Engineering Rationale](#feature-engineering-rationale)
4. [Model Selection Rationale](#model-selection-rationale)
5. [LSTM Rationale](#lstm-rationale)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Prediction Intervals](#prediction-intervals)
8. [SHAP Explanations](#shap-explanations)
9. [MLflow Experiment Tracking](#mlflow-experiment-tracking)
10. [No Data Leakage Guarantee](#no-data-leakage-guarantee)

---

## Problem Definition

**Task:** Supervised regression — predict daily theatre-level cinema attendance count.

**Formal definition:**

Given a theatre `t` and date `d`, predict:

```
ŷ(t, d)  =  E[ audience_count(t, d) | features(t, d) ]
```

where `features(t, d)` includes temporal indicators, historical attendance lags, rolling statistics, and theatre characteristics.

**Target variable:** `audience_count` — an integer ≥ 0 representing the total number of tickets sold or persons who attended a cinema on a given day at a specific theatre.

**Forecast horizon:** 1 to 14 days ahead.

**Granularity:** One prediction per (theatre, date) pair per day.

**Key challenges:**
- **Seasonality:** Weekly (weekends vs. weekdays), monthly, and annual patterns.
- **Holiday effects:** Irregular and large attendance spikes during public holidays.
- **Movie effects:** A blockbuster release dramatically increases attendance; data about which movie is showing is sparse.
- **Theatre heterogeneity:** Multiplex, SingleScreen, IMAX, and DriveIn venues have fundamentally different attendance scales.
- **Cold start:** New theatres or movies have limited history for lag-based features.

---

## Data Split Strategy

> **Rule #1: Never shuffle time-series data.**

Shuffling a time series before splitting creates data leakage — the model sees future observations during training. All splits are strictly temporal.

| Split | Date Range | Purpose |
|---|---|---|
| **Training** | 2022-01-01 → 2024-07-01 | Model fitting |
| **Validation** | 2024-07-02 → 2024-12-31 | Hyperparameter tuning, early stopping, interval calibration |
| **Test** | Last 2 weeks of validation period (held out) | Final unbiased evaluation (reported in Model Metrics dashboard) |

```
Time ──────────────────────────────────────────────────────▶

│◀──────────── Training (2.5 years) ──────────────▶│◀── Val ──▶│
2022-01-01                                    2024-07-01   2024-12-31
```

**Why this split?**
- 30 months of training data provides sufficient history for seasonal patterns (at least 2 full annual cycles).
- 6 months of validation covers both summer and winter seasonal extremes.
- Validation is used to calibrate prediction intervals, so it must remain unseen by the model during training.

---

## Feature Engineering Rationale

Each feature group captures a distinct real-world signal:

### Temporal Features

| Feature | Why It Matters |
|---|---|
| `day_of_week` | Weekend attendance is 40–80% higher than weekday attendance — this is the single strongest predictor. |
| `is_weekend` | Binary encoding provides an explicit weekday/weekend contrast; complements `day_of_week`. |
| `month` | Summer (May–June) and winter holiday (November–January) seasons see elevated demand. |
| `week_of_year` | Captures specific weeks with cultural significance (e.g., Diwali week, school-holiday weeks). |
| `quarter` | Broad seasonal grouping useful for tree-based models' split decisions. |
| `day_of_month` | First and last days of month may show payday-driven attendance. |
| `is_holiday` | Public holidays (Republic Day, Diwali, Christmas) produce some of the highest attendance spikes — typically 60–120% above baseline. |
| `days_since_start` | A linear time index allows models to capture a long-term upward or downward trend in cinema attendance. |

### Lag Features

| Feature | Why It Matters |
|---|---|
| `lag_1` | Yesterday's attendance is highly correlated with today's — represents recent momentum. |
| `lag_7` | Same day last week — captures the weekly seasonal cycle directly. This is typically the most important lag feature. |
| `lag_14` | Two-week lag — provides a second reference point for weekly seasonality and smooths out noise in `lag_7`. |

### Rolling Statistics

| Feature | Why It Matters |
|---|---|
| `rolling_mean_7` | Short-term moving average — represents the recent attendance "regime" a theatre is in. |
| `rolling_mean_14` | Medium-term baseline — helps distinguish whether a recent spike is a trend or a one-off event. |
| `rolling_std_7` | Standard deviation of the past week — measures how volatile attendance has been; high volatility implies wider prediction intervals. |

### Theatre Features

| Feature | Why It Matters |
|---|---|
| `theatre_type_encoded` | Multiplex, IMAX, SingleScreen, and DriveIn have fundamentally different capacity and demand levels. |
| `ticket_price` | Higher ticket prices correlate with premium theatres and can inversely affect attendance. |
| `show_count` | More screenings in a day mean more potential audience slots, raising the ceiling on attendance. |

---

## Model Selection Rationale

### Why XGBoost and LightGBM?

Both are **gradient-boosted decision tree (GBDT)** frameworks and are the state-of-the-art choice for structured tabular time-series data for the following reasons:

1. **Handle non-linear interactions natively:** Attendance is affected by complex interactions between features (e.g., the holiday effect is stronger on weekends than weekdays). Decision trees naturally capture these without manual interaction engineering.

2. **Robust to outliers:** Attendance spikes during blockbuster releases or major holidays are extreme values. GBDTs are less sensitive to outliers than linear models.

3. **No normalization required:** Unlike neural networks, GBDTs do not require feature scaling or normalization.

4. **Built-in feature importance:** Both XGBoost and LightGBM provide gain-based and split-count feature importance out of the box.

5. **Efficient on medium-sized tabular data:** With ~16,000 training rows and 18 features, GBDTs train in seconds and do not require GPU resources.

6. **LightGBM advantage:** LightGBM uses a leaf-wise tree growth strategy (vs. level-wise in XGBoost) and histogram-based splitting, making it faster on larger datasets while achieving similar accuracy.

**Hyperparameters tuned on validation set:**

| Parameter | XGBoost | LightGBM |
|---|---|---|
| `n_estimators` | 500 | 500 |
| `max_depth` | 6 | -1 (no limit, controlled by `num_leaves`) |
| `learning_rate` | 0.05 | 0.05 |
| `subsample` | 0.8 | 0.8 |
| `colsample_bytree` | 0.8 | 0.8 |
| `reg_alpha` | 0.1 | 0.1 |
| `reg_lambda` | 1.0 | 1.0 |
| `num_leaves` | — | 31 |
| `early_stopping_rounds` | 50 | 50 |

---

## LSTM Rationale

A **Long Short-Term Memory (LSTM)** recurrent neural network is included to capture sequential temporal dependencies that tree-based models may miss.

### Why LSTM?

1. **Sequence modelling:** Attendance has autocorrelation — today's attendance depends on a sequence of past days, not just individual lag values. LSTMs are designed to learn from sequences.

2. **Long-range dependencies:** The 28-day lookback window allows the LSTM to see four full weeks of history, capturing monthly patterns that isolated lag features may not encode perfectly.

3. **Gradient gating:** LSTM's forget gate allows it to selectively remember patterns from many days ago (e.g., the same day four weeks prior) while ignoring irrelevant noise.

### Architecture

```
Input shape:  (batch_size, sequence_length=28, num_features=18)
              ↓
LSTM Layer 1: 64 hidden units, return_sequences=True
              ↓
Dropout:      p=0.2
              ↓
LSTM Layer 2: 32 hidden units, return_sequences=False
              ↓
Dropout:      p=0.2
              ↓
Dense Layer:  16 units, ReLU activation
              ↓
Output Layer: 1 unit, linear activation (attendance count)
```

### Training Details

| Parameter | Value |
|---|---|
| **Lookback window** | 28 days |
| **Optimizer** | Adam (lr=0.001) |
| **Loss function** | Huber loss (δ=50) — robust to attendance outliers |
| **Batch size** | 64 |
| **Epochs** | 100 (with early stopping, patience=10) |
| **Framework** | PyTorch |

The Huber loss is preferred over MSE because it is quadratic for small errors but linear for large errors (outlier-robust).

---

## Evaluation Metrics

### Mean Absolute Error (MAE)

```
         1   n
MAE  =  ─── Σ |yᵢ - ŷᵢ|
         n  i=1
```

**Interpretation:** The average number of attendees by which the model's prediction is off, in the same units as attendance. A MAE of 30 means the model is, on average, off by 30 people.

**Why MAE?** It is robust to outliers, easy to explain to cinema managers ("on average, our forecast is wrong by X people"), and directly actionable.

---

### Root Mean Squared Error (RMSE)

```
          ┌──────────────────┐
          │  1   n           │
RMSE  =  √│ ─── Σ (yᵢ - ŷᵢ)²│
          │  n  i=1          │
          └──────────────────┘
```

**Interpretation:** Similar to MAE but penalises large errors more heavily due to squaring. If RMSE >> MAE, the model has occasional very large errors.

---

### Weighted Absolute Percentage Error (WAPE)

```
         Σ |yᵢ - ŷᵢ|
WAPE  = ─────────────
            Σ yᵢ
```

**Interpretation:** The total absolute error divided by total actual attendance — a percentage error weighted by volume. Unlike MAPE, WAPE avoids division-by-zero when actual values are small.

**Example:** A WAPE of 0.089 (8.9%) means the model's total forecast error is 8.9% of actual total attendance — the model is off by 8.9 cents in every rupee of predicted tickets.

---

### R² (Coefficient of Determination)

```
         Σ (yᵢ - ŷᵢ)²
R²  =  1 - ─────────────────
         Σ (yᵢ - ȳ)²
```

**Interpretation:** The proportion of variance in attendance explained by the model. R² = 0.91 means the model explains 91% of attendance variation; the remaining 9% is unexplained noise.

---

## Prediction Intervals

BoxOfficeIQ provides **90% prediction intervals** `[lower_bound, upper_bound]` around each point forecast. These intervals are calibrated using **residual bootstrapping on the validation set**.

### Calibration Procedure

1. **Collect residuals:** After training, run the model on the validation set and collect all residuals:  
   `ε = y_actual - ŷ_predicted`

2. **Compute quantiles:** For a 90% interval, compute the 5th and 95th percentile of `ε` across all validation residuals.  
   `q5 = np.percentile(residuals, 5)`  
   `q95 = np.percentile(residuals, 95)`

3. **Construct intervals:** For any new prediction `ŷ`:  
   `lower_bound = max(0, ŷ + q5)`  
   `upper_bound = ŷ + q95`

4. **Per-theatre calibration:** Optionally, residuals are split by theatre type (Multiplex, IMAX, etc.) to produce tighter, more accurate intervals for each venue class.

### Interpretation

> "There is a 90% probability that actual attendance will fall between the lower and upper bounds."

- **Wide intervals** indicate high uncertainty — perhaps due to volatility in recent attendance (`rolling_std_7` is high) or a holiday whose effect is uncertain.
- **Narrow intervals** indicate the model is confident — consistent recent attendance, stable week.

**What prediction intervals are NOT:**
- Guarantees of actual attendance.
- Intervals based on a formal probability model (they are empirically calibrated, not derived from distributional assumptions).

---

## SHAP Explanations

### What is SHAP?

**SHapley Additive exPlanations (SHAP)** is a game-theory-based method for explaining the output of any machine learning model. It assigns each feature a contribution value (SHAP value) for a specific prediction.

**Core idea:** The SHAP value for feature `j` in prediction `i` is the average marginal contribution of feature `j` across all possible subsets of features, computed by analogy with the Shapley value in cooperative game theory.

### Tree SHAP

For tree-based models (XGBoost, LightGBM), **TreeSHAP** (Lundberg et al., 2020) computes exact SHAP values in polynomial time by exploiting the tree structure. This is orders of magnitude faster than model-agnostic approaches (like KernelSHAP) which require many model evaluations.

**API used:**
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_instance)
```

### Global vs. Local Explanations

| Explanation Type | What It Shows | How Computed | Displayed in |
|---|---|---|---|
| **Global** | Which features are most important overall | `mean(|SHAP value|)` across all predictions | Forecasting page — bar chart |
| **Local** | Why a specific prediction was high/low | SHAP values for a single (theatre, date) | Forecasting page — waterfall chart |

**Global SHAP example output:**

```
Feature          Mean |SHAP|
lag_7            82.4
rolling_mean_7   61.3
is_weekend       45.7
is_holiday       38.2
lag_1            31.5
month            20.1
...
```

**Interpretation of a local SHAP plot:**

- **Positive SHAP value:** This feature pushed the prediction *higher* than the baseline.
- **Negative SHAP value:** This feature pushed the prediction *lower* than the baseline.
- **Baseline:** The average model prediction across all training data (`E[ŷ]`).

> Example: "The model predicted 420 attendees. The baseline was 290. `is_holiday=1` contributed +85, `is_weekend=1` contributed +52, `rolling_mean_7=180` contributed -7 (below-average recent trend), giving a net prediction of 420."

---

## MLflow Experiment Tracking

MLflow is used for complete reproducibility of all training runs.

### What Is Tracked

For each training run, MLflow records:

| Category | Examples |
|---|---|
| **Parameters** | `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `lookback=28` |
| **Metrics** | `train_mae`, `val_mae`, `val_rmse`, `val_wape`, `val_r2` |
| **Artifacts** | Serialized model file (`.pkl` / `.pt`), `feature_names.json`, SHAP summary plot, residual plot |
| **Tags** | `model_type=xgboost`, `theatre_id=aggregate`, `run_date=2024-11-13` |
| **Source** | Git commit hash, Python version, training script path |

### How to View Experiments

```bash
# Start the MLflow UI
mlflow ui --port 5000
# Visit http://localhost:5000
```

The UI shows:
- All experiment runs in a table, sortable by any metric.
- Metric comparison plots across runs.
- Artifact browser to download model files.
- Run-to-run diff for parameters.

### Model Registry

Production models are promoted through stages in the MLflow Model Registry:

```
None  →  Staging  →  Production  →  Archived
```

The FastAPI backend loads the **Production** stage model at startup:
```python
model = mlflow.sklearn.load_model("models:/XGBoostAttendance/Production")
```

---

## No Data Leakage Guarantee

Data leakage occurs when information about the future is used to train a model, producing falsely optimistic evaluation metrics that do not generalise to real forecasting.

BoxOfficeIQ employs the following safeguards:

### 1. Temporal Train/Validation Split

Training data is strictly limited to dates ≤ 2024-07-01. The validation set (2024-07-02 to 2024-12-31) is **never seen by the model during training or hyperparameter search**.

### 2. Lag Features Use Only Past Data

```python
# Computed per theatre using groupby + shift
df['lag_1'] = df.groupby('theatre_id')['audience_count'].shift(1)
df['lag_7'] = df.groupby('theatre_id')['audience_count'].shift(7)
```

`shift(n)` moves values forward by `n` rows within each theatre group, ensuring that the lag for date `d` only uses data from date `d - n`.

### 3. Rolling Statistics Use Closed='left' Windows

```python
df['rolling_mean_7'] = (
    df.groupby('theatre_id')['audience_count']
      .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
)
```

The `.shift(1)` before `.rolling()` ensures the window for date `d` ends at `d - 1`, never including `d` itself.

### 4. No Target Encoding with Full Dataset

Target encoding (replacing a categorical variable with the mean target value for that category) can leak if computed on the full dataset. If used, it must be computed **only on the training fold** and then applied to validation and test.

### 5. Prediction Interval Calibration Uses Validation Set

The quantiles `q5` and `q95` for prediction intervals are computed on **validation residuals only** — after the model has been trained on the training set. This ensures the intervals reflect true out-of-sample uncertainty.

### 6. Verification

After every pipeline run, the following assertion is checked:

```python
assert train_df['date'].max() < val_df['date'].min(), "Leakage detected: train/val overlap"
```
