# UFC Fight Outcome Predictor

A quantitative betting-research project on UFC fights, built and evaluated using the same methodology a trading strategy would be: purged walk-forward cross-validation, probability calibration, event-driven backtests against real bookmaker odds, pre-registered out-of-sample holdouts, and closing-line-value analysis.

**Headline result: an early-line value strategy achieved +4.0% closing-line value (95% CI [+2.1%, +6.2%]) on a pre-registered out-of-sample holdout, with 83% of bets beating the closing line.** It was positive in every holdout month and on the development period (+2.3%, CI [+0.5%, +3.8%]). That makes it a statistically significant positive expected return of roughly +2% to +4% per bet. It was reached after four earlier false positives were identified and rejected (see [Results](#results)).

## Live Demo

🔗 **[Try the dashboard](https://ufc-fight-predictor-j2jbm8p3zzdrvappi9dbpxa.streamlit.app)** — browse real historical UFC fights, see the model's calibrated win probability, top SHAP drivers, and comparison to the bookmaker line.

---

## Results

### Week 10 (final result): early-line value betting against Pinnacle — positive, significant CLV

Rather than out-predict the market, this strategy exploits slower bookmakers lagging Pinnacle, the sharpest book. It bets at the second-best available price when that price beats Pinnacle's de-vigged fair line by more than 2%, and it is scored by closing-line value (CLV), the professional standard for measuring edge.

| | Development (Aug 2025–Mar 2026) | Holdout (Apr–Sep 2026) |
|---|---|---|
| Bets | 120 | 54 |
| CLV vs. closing line | +2.3%, 95% CI [+0.5%, +3.8%] | **+4.0%, 95% CI [+2.1%, +6.2%]** |
| Bets beating the close | 73% | 83% |

A first version passed its holdout but was shown by a robustness check to be an artefact of the de-vig method; it was corrected and re-tested, and this is disclosed as a second look at the holdout. The caveats are real: small sample, prices scraped every ~1.5 days, and soft bookmakers limit winning accounts. Full write-up: [`reports/methodology.md`](reports/methodology.md#week-10-early-line-value-betting-against-pinnacle).

### How the project got there

Each earlier stage below was tested and rejected. Those rejections are what make the Week 10 result credible.

#### Week 9: market-anchored model on a locked 5-year holdout

The original model (below) forecast fights from scratch and lost to the bookmaker, whose line alone scores AUC 0.716. The redesign starts from the market's probability and learns only systematic corrections to it. It was developed on 2010–Mar 2021, pre-registered, then run **once** on fights it had never seen (Apr 2021–Mar 2026).

| Holdout metric | Value |
|---|---|
| Bets | 1,026 (flat 1-unit stakes) |
| ROI | **+4.4%** (+44.8 units), 95% CI [-2.8%, +11.8%] |
| Log loss, model vs. market | **0.5898 vs. 0.5930** (beats the line) |
| Max drawdown | -31.1 units |
| Break-even price haircut | ~8% worse payouts |

**Verdict for this model: no demonstrated bettable edge.** The holdout ROI was positive but not significant. A pre-registered closing-line-value test (108 bets, Aug 2025–Mar 2026, ~35 bookmakers including Pinnacle) then failed clearly: picks were worth **-4.0% against the closing line** (95% CI [-5.6%, -2.3%]), so the holdout profit was most likely variance. Method-of-victory props and three feature upgrades were tested on development data and rejected. Full write-up: [`reports/methodology.md`](reports/methodology.md#week-9-market-anchored-model-and-a-locked-holdout).

#### Weeks 1–8: original from-scratch model

| Metric | Value |
|---|---|
| Out-of-sample AUC (XGBoost, purged walk-forward CV) | ~0.625 |
| Sharpe ratio (annualised, underdog-selection backtest) | **-0.150** (95% CI: [-0.544, 0.228]) |
| Max drawdown | -38.2% |
| Calmar ratio | -0.122 |
| Backtest return vs. random betting | Model underperformed random betting |
| Backtest return vs. always-bet-favourite | Model underperformed favourite-betting |

**Finding for this model: no statistically significant edge against bookmaker odds.** The model's Sharpe ratio confidence interval includes zero, and a backtest restricted to the model's underdog-selection strategy underperformed both a random-betting baseline and a naive always-favourite baseline on the same out-of-sample fights.

This is reported as a negative/inconclusive result rather than forced into a favourable-looking headline number. An earlier version of this backtest appeared profitable (+109.9%, later +4.7% after fixing a model-reproducibility bug) — decomposing that result by year revealed it was driven almost entirely by a single anomalous year coinciding with the walk-forward CV's least-mature folds. Once that artifact was removed, the true result was negative. Full investigation, methodology, and the reasoning behind this conclusion are in [`reports/tearsheet.pdf`](reports/tearsheet.pdf) and the notebooks below.

**Why the earlier negative results are kept:** catching a non-reproducibility bug, decomposing a misleading aggregate into its true driver, and rejecting false positives is what gives the final Week 10 result its credibility.

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

ufc-fight-predictor/
├── app/
│ └── streamlit_app.py # Deployed dashboard
├── data/
│ ├── raw/ # Kaggle UFC dataset + bookmaker odds
│ └── processed/ # Cleaned fights table, feature matrix
├── notebooks/
│ ├── 01_data_cleaning.ipynb
│ ├── 02_feature_engineering.ipynb
│ ├── 03_modelling.ipynb
│ ├── 04_calibration.ipynb
│ ├── 05_backtest.ipynb
│ ├── 06_tearsheet.ipynb
│ └── 07_dashboard_prep.ipynb
├── reports/
│ └── tearsheet.pdf # One-page results summary
├── src/ # All modelling, backtest, and utility code
└── requirements.txt


## Running this project

```bash
pip install -r requirements.txt
```

Run notebooks `01` through `07` in order — each is self-contained and re-runs cleanly from a fresh kernel. To run the dashboard locally:

```bash
streamlit run app/streamlit_app.py
```

To reproduce the Week 9–10 results, run the scripts from `src/`. The Week 10 scripts need the Kaggle "UFC Betting Odds (Daily Updated Dataset)" (`kaggle datasets download -d jerzyszocik/ufc-betting-odds-daily-dataset`):

```bash
cd src
python market_edge_holdout.py                          # Week 9 locked holdout
python market_edge_clv.py <UFC_betting_odds.csv>       # Week 9 closing-line test
python line_move.py <UFC_betting_odds.csv>             # Week 10 development research
python line_move_holdout_v2.py <UFC_betting_odds.csv>  # Week 10 final holdout test
```

---

## Extensions and future work

- Forward-test the Week 10 strategy at live, bettable prices with continuous price monitoring, and measure realised profit alongside CLV.
- Extend the Week 10 strategy to other MMA promotions for more betting volume.
- Re-validate the underdog edge decile threshold on a genuinely held-out data slice, rather than the current in-sample selection (a documented limitation — see `reports/tearsheet.pdf`).
- Test a larger initial walk-forward training window to reduce the variance in the earliest folds.
- Bayesian ELO, ensemble methods, and regime detection, as noted above.
- Live feature construction for arbitrary (non-historical) fighter matchups.