# UFC Fight Outcome Predictor

A calibrated machine-learning system for predicting UFC fight outcomes from pre-fight fighter statistics, built and evaluated using the same methodology a quantitative trading strategy would be: purged walk-forward cross-validation, probability calibration, an event-driven backtest against real historical bookmaker odds, and honest reporting of the result — including when that result is negative.

## Live Demo

🔗 **[Try the dashboard](https://ufc-fight-predictor-j2jbm8p3zzdrvappi9dbpxa.streamlit.app)** — browse real historical UFC fights, see the model's calibrated win probability, top SHAP drivers, and comparison to the bookmaker line.

---

## Results

| Metric | Value |
|---|---|
| Out-of-sample AUC (XGBoost, purged walk-forward CV) | ~0.625 |
| Sharpe ratio (annualised, underdog-selection backtest) | **-0.150** (95% CI: [-0.544, 0.228]) |
| Max drawdown | -38.2% |
| Calmar ratio | -0.122 |
| Backtest return vs. random betting | Model underperformed random betting |
| Backtest return vs. always-bet-favourite | Model underperformed favourite-betting |

**Headline finding: this project found no statistically significant edge against bookmaker odds.** The model's Sharpe ratio confidence interval includes zero, and a backtest restricted to the model's underdog-selection strategy underperformed both a random-betting baseline and a naive always-favourite baseline on the same out-of-sample fights.

This is reported as a negative/inconclusive result rather than forced into a favourable-looking headline number. An earlier version of this backtest appeared profitable (+109.9%, later +4.7% after fixing a model-reproducibility bug) — decomposing that result by year revealed it was driven almost entirely by a single anomalous year coinciding with the walk-forward CV's least-mature folds. Once that artifact was removed, the true result was negative. Full investigation, methodology, and the reasoning behind this conclusion are in [`reports/tearsheet.pdf`](reports/tearsheet.pdf) and the notebooks below.

**Why this negative result is presented prominently rather than hidden:** the ability to design a rigorous validation and backtesting pipeline, catch a subtle non-reproducibility bug, decompose a misleading aggregate result into its true driver, and report an honest negative finding instead of a flattering but wrong one is itself the core skill this project set out to demonstrate — arguably more relevant to quantitative research roles than a clean positive number would have been.

---

## What this project maps to in quantitative finance

| UFC predictor | Quant finance equivalent |
|---|---|
| Pre-fight fighter stats | Financial factors / signals |
| Historical fight outcomes | Price returns / trade outcomes |
| ELO ratings updated after each fight | Signal decay and updating |
| Purged walk-forward CV | Point-in-time backtesting with no lookahead |
| Model probability vs. bookmaker implied probability | Alpha signal vs. market consensus |
| Kelly criterion position sizing | Optimal position sizing |
| Sharpe, max drawdown, Calmar | Strategy performance metrics |
| Calibration plot | Probability model reliability for risk management |
| Edge decile chart | Signal strength vs. realised return attribution |

---

## Methodology

**Data.** 4,633 UFC fights (2010–2021) from the Kaggle UFC dataset, filtered to exclude a period (pre-2010) where corner assignment leaked the outcome. Bookmaker odds sourced separately and joined on fighter names and date (95% match rate).

**Features.** 148 pre-fight features, all audited for lookahead bias: career-average fight statistics (verified to exclude the current fight), an ELO rating system built from scratch with an adaptive K-factor, and physical/matchup differentials (age, reach, stance, layoff time).

**Validation.** Purged walk-forward cross-validation — expanding training window, non-overlapping test windows, fold boundaries snapped to event-card dates to prevent same-card leakage. 16 folds covering 2010–2020.

**Model.** XGBoost, hyperparameters tuned once via Optuna on an internal chronological split of training data only. Compared against a logistic regression baseline (near-tie on most metrics; XGBoost retained for SHAP explainability).

**Calibration.** Platt scaling and isotonic regression tested against raw model output. Raw XGBoost was already close to well-calibrated; neither method meaningfully improved it — a legitimate finding, not a failed deliverable.

**Backtest.** Event-driven, chronological, comparing model probability to bookmaker implied probability. A stratified edge-decile analysis found the model's apparent edge on underdog bets did not survive removal of the walk-forward CV's earliest (smallest-training-window) folds — see Results above.

**Position sizing.** Fractional Kelly criterion was implemented and tested against flat staking. A systematic cap sweep found any Kelly cap loose enough to meaningfully differentiate stake by edge also amplified variance faster than it captured edge; flat staking was used as the primary comparison for this reason, with the Kelly investigation retained as a documented methodological finding rather than a working feature.

**Reproducibility.** All XGBoost models are seeded (`random_state=42`). An earlier version of this pipeline was not seeded and produced materially different results (+109.9% vs. -37%) on identical code and data — caught by deliberately re-running the pipeline twice and comparing outputs. This is documented as a specific, real lesson in the notebooks, not glossed over.

---

## What was deliberately cut from scope

- **Bayesian ELO (PyMC)** — too time-intensive for the project timeline; noted as a natural extension.
- **Ensemble stacking** — a single well-validated model was prioritised over added complexity without demonstrated lift.
- **Neural networks** — XGBoost + SHAP is more defensible and explainable than a deep model would be for this dataset size and this audience.
- **Live arbitrary-matchup prediction** — the deployed dashboard browses real historical fights rather than constructing features for hypothetical future matchups, which would require a separate live feature-inference pipeline. Noted as the primary extension for future work.

---

## Repository structure