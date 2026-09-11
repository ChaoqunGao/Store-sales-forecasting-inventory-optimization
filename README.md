# Demand Forecasting & Inventory Optimization: A Statistical Deep Dive

A time series forecasting and inventory optimization pipeline built on the Kaggle Store Sales (Favorita) dataset. The core analysis is deliberately scoped to a single SKU-location pair to demonstrate PhD-level statistical rigor rather than broad model coverage, with a lightweight batch diagnostic layer (50 series) added to show scalability awareness without diluting that focus.

## Why this project is scoped the way it is

Most public forecasting portfolios optimize for breadth: many models, many store/product combinations, a leaderboard-style comparison. This project takes the opposite approach. It focuses on **one SKU-location pair** (Store 1 / GROCERY I) and asks a harder question of it: not just "which model wins," but *why*, under what conditions, and with what statistical justification.

The goal is to show reasoning depth on a forecasting problem — grounding model choice in statistical evidence and being explicit about what each result does and doesn't support — rather than to demonstrate familiarity with as many algorithms as possible. No gradient-boosted tree models (LightGBM, XGBoost) were added for this reason — they would broaden the toolkit without adding statistical depth relevant to the story this project tells.

## Pipeline overview

| Notebook | Purpose |
|---|---|
| `NB1` | Exploratory data analysis — STL decomposition, ADF stationarity testing, outlier flagging |
| `NB2` | SARIMA/SARIMAX modeling |
| `NB3` | Prophet modeling with holiday effects |
| `NB4` | Model comparison via rolling-origin cross-validation |
| `NB5` | Inventory optimization — Safety Stock, Reorder Point, EOQ |
| `NB6` | Batch diagnostics tool across 50 series (scalability / future work) |

## Methodology & key results

**EDA (NB1):** STL decomposition and ADF testing establish the series' stationarity properties and surface an early caution: ADF results are sensitive to the choice of regression specification (`'c'` vs `'ct'`), a theme that resurfaces in the batch diagnostics.

**Modeling (NB2–NB3):** SARIMAX incorporates `onpromotion` as an exogenous regressor; Prophet is fit with holiday effects layered in.

**Model comparison (NB4):** Rather than relying on a single train/test split, model performance is evaluated with rolling-origin cross-validation across four windows, each chosen to stress-test a different condition (holiday-dense, holiday-light, short training history). This matters because a single 90-day split proved insufficient to support any real ranking conclusion — the four-window design surfaces findings a single split would have hidden entirely:

- Prophet won 3 of 4 windows, **including** a holiday-light window — indicating its advantage isn't purely from holiday effects but also from its handling of yearly seasonality more broadly.
- In the shortest-history window, SARIMAX's `onpromotion` coefficient became unstable, coinciding with low promotion variance in that window.

These findings replaced an earlier, simpler heuristic ("if a holiday falls in the forecast window, use Prophet") with a more defensible framework based on training history length and promotion variance — a better reflection of *why* one model outperforms another, not just *that* it does.

**Inventory optimization (NB5):** Safety Stock, Reorder Point, and EOQ are computed with lead time treated as a scenario variable (3/7/14 days) and demand uncertainty (σ) sourced from the rolling CV mean RMSE, tying the inventory layer directly to the forecasting layer's demonstrated error rather than an assumed constant. EOQ is presented as a sensitivity table rather than a single point estimate.

A note on scope: this project intentionally does not incorporate a newsvendor framing. Newsvendor models typically combine *yield* uncertainty with *demand* uncertainty, while this pipeline only models demand uncertainty. Forcing a newsvendor lens on here would overstate what the pipeline actually captures.

**Scalability diagnostics (NB6):** To acknowledge scalability without diluting the project's depth-first identity, NB6 runs lightweight diagnostics — ADF testing, STL seasonal strength, IQR-based outlier flagging, and `auto_arima` order selection — across 50 series rather than building a full parallel batch-modeling pipeline.

Results: 48/50 series fit successfully, with 31 unique `auto_arima` parameter combinations — concrete evidence that a single model specification would not generalize well across even a modest sample of series. The diagnostics also flagged a stationarity disagreement between ADF specifications (`'c'` vs `'ct'`) on 19.4% of series (345/1,782) in the full-scale run: plain ADF under-detects trend-driven non-stationarity in series with a visible multi-year upward trend (confirmed via a Store 3/GROCERY I case study), and `auto_arima`'s own differencing choice agreed with the trend-aware specification. Rather than silently resolving these disagreements, affected series are flagged `needs_review` — treating "knowing when to flag uncertainty" as part of the pipeline's value, not a gap in it.

## Tech stack

- **Python** — pandas, matplotlib
- **Modeling** — statsmodels, pmdarima (`auto_arima`), prophet
- **Data** — [Kaggle Store Sales (Favorita) dataset](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- **Format** — Jupyter notebooks as the primary storytelling medium, with `.py` modules for reusable logic

## Setup

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

Download the Favorita dataset from Kaggle and place it in `data/`, then run the notebooks in order (`NB1` → `NB6`).

## Scalability / future work

NB6's diagnostics are a deliberate boundary, not an oversight: they demonstrate awareness that a production pipeline would need to run at scale, without expanding this project's core narrative from a focused, defensible single-series analysis into a shallower multi-series one. Natural next steps — outside the current scope — would include automating model refitting on the `needs_review` flagged series and extending the rolling-CV framework across the full batch.
