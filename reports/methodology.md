# UFC Fight Outcome Predictor — Methodology

## Motivation

This project applies the discipline of quantitative research to a domain
with abundant, clean, publicly available data and genuine structural
uncertainty: UFC fight outcomes. The parallels to quantitative trading are
exact, not approximate — pre-fight statistics are factors, fight outcomes
are trade results, ELO ratings are a signal-decay model, purged
walk-forward cross-validation is point-in-time backtesting, and the
bookmaker line is the market consensus a model's edge must be measured
against. The project was built to demonstrate the full research
lifecycle a quantitative researcher is expected to execute: data
engineering, leakage-free feature construction, model validation,
probability calibration, backtesting against a real market, and — the
part most portfolio projects skip — honest reporting of the result,
whatever it turns out to be.

## Data

4,633 UFC fights (2010-03-21 to 2021-03-20), sourced from the Kaggle UFC
dataset (Rajeev Warrier) and joined to a separate historical bookmaker
odds dataset (mdabbert's "Ultimate UFC Dataset") on fighter names and
date, with a 95.0% match rate. Fights before 2010-03-21 were excluded
after discovering that corner assignment (`R_fighter`/`B_fighter`)
perfectly predicted the outcome before that date — a genuine data
artifact, independently corroborated by the odds dataset's own
documentation excluding the same period for the same reason.

## Features

148 pre-fight features, each explicitly audited against a single test:
*could this value have been known before the fight started?* Three
families of features:

- **Career-average fight statistics** (striking accuracy, takedown rate,
  submission attempts, etc.) — sourced from pre-built columns in the
  original dataset, verified against the author's documentation to
  exclude the current fight from the average.
- **ELO ratings**, implemented from first principles (`R_new = R_old + K
  (S - E)`), with an adaptive K-factor (K=40 while a fighter has fewer
  than 5 tracked fights, K=32 once established) to avoid a static K
  either overreacting to early noise or underreacting to a fighter's
  true current form. Hand-validated against a toy dataset and against
  Anderson Silva's real career trajectory.
- **Physical and matchup differentials** — reach, height, age, stance
  matchup, and days since last fight, all computed as simple,
  no-lookahead differences between the two fighters' pre-fight states.

## Validation methodology

Standard k-fold cross-validation is invalid here: it would train on
future fights and test on past ones, leaking information no real,
point-in-time system could have had. Instead, **purged walk-forward
cross-validation** was used — an expanding training window, followed by
a non-overlapping test window, advanced forward in time, repeated across
16 folds (`initial_train_size=1200, test_size=200`). Fold boundaries were
snapped to full date changes so that a single UFC event card (multiple
fights sharing one date) is never split across a train/test boundary —
enforced by an automated leakage validator run on every fold.

## Model selection

A logistic regression baseline (mean AUC 0.625) was compared against
XGBoost, tuned once via Optuna on an internal chronological split of the
final fold's training data (never touching any test fold). The two
models were essentially tied — XGBoost modestly better on log-loss and
Brier score, logistic regression better on AUC and accuracy. XGBoost was
retained specifically because SHAP explainability (a required
deliverable) works substantially better on tree ensembles, not because
it was the stronger predictor. This honest tie, reported rather than
hidden, is itself a finding: added model complexity did not clearly
improve the result.

## Calibration

Platt scaling and isotonic regression were both tested against raw
XGBoost output. Neither meaningfully improved Brier score (isotonic beat
raw in only 4 of 16 folds; Platt in 7 of 16). This is explained by the
model's Optuna-tuned hyperparameters producing shallow, conservative
trees rather than the deep, overconfident ones calibration typically
corrects. Confirmed visually via a pooled reliability diagram across all
16 folds' test predictions — raw XGBoost tracked close to the diagonal in
well-populated probability bins.

## Backtest and the central finding

An event-driven backtest compared the model's probability to the
bookmaker's implied probability on each fight, betting the side with the
larger edge above a threshold. Two important methodological corrections
were made during this process, both worth stating explicitly:

**1. Reproducibility.** An early, unseeded version of the pipeline
produced a backtest return of +109.9%. After fixing a missing
`random_state` in XGBoost's construction (a real bug, not a parameter
tuning choice), the identical code and data reproducibly produced
+0.3% to +4.7% instead — an enormous swing that would have gone
undetected without deliberately re-running the pipeline twice and
comparing results. This is treated as one of the project's most
important findings: a backtest number is worthless without a
demonstrated ability to reproduce it.

**2. Decomposition of the remaining result.** The seeded, reproducible
+4.7% figure was decomposed year by year. It was found to be driven
almost entirely by a single year (2014), which alone contributed more
profit than the full 2014-2020 sample combined. Tracing this to the
walk-forward CV structure showed 2014 corresponds to the earliest folds
— the smallest training window (~1,200 fights), and therefore the
highest-variance, least-reliable probability estimates in the entire
pipeline. Excluding those folds and re-running the identical backtest
on the remaining, more-trustworthy 1,937 fights produced **-25.4%
return**, the opposite conclusion.

**Final conclusion: this model does not demonstrate a robust, exploitable
edge against bookmaker odds.** A stratified edge-decile analysis
(favourites vs. underdogs) on the stable data shows no reliable upward
trend in either group — the underdog deciles are uniformly negative,
including the highest-edge decile, which performed worst of all. The
Sharpe ratio on this result is -0.150, with a bootstrapped 95%
confidence interval of [-0.544, 0.228] that comfortably includes zero.
The model also underperformed both a random-betting benchmark ($768.39)
and a naive always-bet-favourite benchmark ($879.11) starting from the
same $1,000 bankroll on the same fights ($746.01).

This is reported as the project's central finding, not as a failure to
be minimised. A rigorous methodology that correctly identifies and
rejects a false positive is more valuable evidence of research skill
than an unexamined positive result — and the false positive here was
subtle enough (a fold-variance artifact concentrated in one calendar
year) that it would plausibly have survived into a published result
without the decomposition step.

## Position sizing

Fractional (half-)Kelly criterion sizing was implemented (`f* = (bp -
q)/b`) and systematically compared against flat staking via a cap sweep.
Every cap loose enough to let Kelly meaningfully differentiate stake by
edge magnitude also amplified variance faster than it captured any
genuine edge — a direct consequence of the underlying signal being thin
and noisy (see Backtest section above). The only cap tight enough to
avoid this degenerated to flat staking in practice, meaning Kelly's
actual sizing logic never contributed positively at any tested
parameterisation. Flat staking was used as the primary result for this
reason. This is a textbook illustration of why real Kelly
implementations use conservative fractional multipliers and hard caps:
position sizing proportional to an estimated edge is dangerous when the
edge estimate itself carries meaningful uncertainty.

## Limitations

- **In-sample threshold selection.** The `edge_threshold=0.20` cutoff
  used throughout the backtest was chosen by inspecting decile results
  on the same out-of-sample data the backtest is evaluated on — a mild,
  strategy-level form of lookahead, distinct from and less severe than
  the model-training lookahead rigorously avoided elsewhere. A stronger
  design would select the threshold on an earlier data slice and
  validate on a later held-out slice.
- **Data source noise.** ~3.3% of odds values were missing and 5.0% of
  fights had no odds match at all; both were excluded rather than
  imputed, which is conservative but reduces the effective sample.
- **Walk-forward fold variance.** The earliest folds carry materially
  higher-variance probability estimates than later folds due to smaller
  training windows — this drove the single largest false-positive result
  in the project (see Backtest section) and remains a structural property
  of the current fold configuration.

## Extensions

- Re-validate any future threshold or parameter choice on a genuinely
  held-out slice, never the same out-of-sample data used to report
  results.
- Test a larger `initial_train_size` to reduce early-fold variance, at
  the cost of fewer total folds.
- Bayesian ELO (PyMC) — deliberately excluded from this project's scope
  as too time-intensive for the timeline.
- Ensemble stacking and neural networks — deliberately excluded; a
  single, well-validated, explainable model was prioritised over added
  complexity without demonstrated lift, consistent with the finding that
  XGBoost and logistic regression were themselves nearly tied.
- Live feature construction for arbitrary (non-historical) fighter
  matchups, extending the current dashboard beyond browsing real
  historical fights.