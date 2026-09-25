"""
XGBoost model for the UFC fight predictor, evaluated inside the same
purged walk-forward CV loop as the logistic regression baseline
(baseline_model.py).

Design decisions (documented for defense in an interview):

- NO IMPUTATION. XGBoost handles missing values natively - at each
  split, it learns which direction (left/right branch) is best for
  rows with NaN in that feature, rather than needing a filled-in
  value beforehand. This is a genuine algorithmic difference from
  logistic regression, not an oversight - the missing-value pattern
  itself (e.g. "this fighter has no rolling-form history yet, i.e. a
  debut") can be informative, and imputing it away would throw that
  information out.

- NO SCALING. Tree splits are threshold comparisons (is feature X
  greater than value V), which are invariant to the scale/units of X.
  StandardScaler was necessary for logistic regression's gradient-based
  optimization; it does nothing useful for a tree model.

- HYPERPARAMETER TUNING HAPPENS ONCE, ON AN INTERNAL CHRONOLOGICAL
  SPLIT CARVED OUT OF TRAINING DATA ONLY - never against any fold's
  test set. Tuning against a test fold's actual score would leak that
  fold's answer key into the tuning process, silently inflating
  reported performance. Instead: take the largest available training
  window (the last fold's training data), split IT chronologically
  (e.g. earliest 85% / most recent 15%), tune against that internal
  validation slice with Optuna, then apply the SAME tuned
  hyperparameters across every fold's full training set in the walk-
  forward loop. This is a legitimate simplification versus tuning
  fresh per fold (which would be extremely slow - 16 folds x N optuna
  trials each) and keeps the tuning process itself leakage-free.

- FIXED RANDOM_STATE EVERYWHERE XGBOOST IS CONSTRUCTED. XGBoost's
  tree-building involves randomness (e.g. feature/row subsampling
  internals), so without a fixed seed, two runs of "the same"
  experiment - same data, same hyperparameters - can produce
  meaningfully different fitted models. This was caught empirically in
  Week 5: re-running the walk-forward + backtest pipeline after a
  kernel restart (same code, same data) produced swings from roughly
  +110% to -37% total backtest return, which is far too large to be
  explainable by legitimate randomness alone and made the entire
  underdog-edge finding untrustworthy until fixed. random_state=42 is
  set both inside the Optuna objective (so tuning itself is
  deterministic) and explicitly added to the returned best_params dict
  (study.best_params from Optuna only contains the tuned/suggested
  values - n_estimators, max_depth, learning_rate - NOT fixed
  parameters, so random_state must be added back in explicitly or it
  silently reverts to XGBoost's default nondeterministic behaviour
  everywhere best_params is later unpacked with **xgb_params).
"""

import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score

optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep notebook output readable

XGB_RANDOM_STATE = 42


def tune_xgb_hyperparams(
    df: pd.DataFrame,
    train_idx: list[int],
    feature_cols: list[str],
    n_trials: int = 30,
    inner_val_fraction: float = 0.15,
) -> dict:
    """
    Tunes n_estimators, max_depth, learning_rate via Optuna, using an
    internal chronological split WITHIN train_idx only. The most recent
    `inner_val_fraction` of train_idx becomes the internal validation
    set; everything before it is the internal training set. No data
    outside train_idx (i.e. no test fold) is touched at any point.

    Returns the best hyperparameter dict found (minimizing log-loss on
    the internal validation slice), with random_state=XGB_RANDOM_STATE
    explicitly included so every downstream model fit using this dict
    is reproducible (see module docstring - this is NOT automatic,
    Optuna's study.best_params only records the tuned parameters).
    """
    n = len(train_idx)
    split_point = int(n * (1 - inner_val_fraction))
    inner_train_idx = train_idx[:split_point]
    inner_val_idx = train_idx[split_point:]

    X_inner_train = df.iloc[inner_train_idx][feature_cols]
    y_inner_train = df.iloc[inner_train_idx]["target"]
    X_inner_val = df.iloc[inner_val_idx][feature_cols]
    y_inner_val = df.iloc[inner_val_idx]["target"]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "eval_metric": "logloss",
            "random_state": XGB_RANDOM_STATE,
        }
        model = XGBClassifier(**params)
        model.fit(X_inner_train, y_inner_train)
        preds = model.predict_proba(X_inner_val)[:, 1]
        return log_loss(y_inner_val, preds, labels=[0, 1])

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=XGB_RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_params["random_state"] = XGB_RANDOM_STATE

    print(f"Tuning complete: best internal log-loss = {study.best_value:.4f}")
    print(f"Best params: {best_params}")
    return best_params


def train_and_evaluate_fold_xgb(
    df: pd.DataFrame,
    train_idx: list[int],
    test_idx: list[int],
    feature_cols: list[str],
    xgb_params: dict,
) -> dict:
    """
    Trains an XGBoost classifier with the given (already-tuned)
    hyperparameters on one fold's training data, scores it on that
    fold's test data. No imputation, no scaling - see module docstring.

    xgb_params is expected to already include random_state (set by
    tune_xgb_hyperparams). As a safety net, random_state is defaulted
    here too via setdefault, in case xgb_params was constructed by hand
    elsewhere (e.g. conservative_params in earlier diagnostic notebook
    cells) without one.
    """
    xgb_params = dict(xgb_params)
    xgb_params.setdefault("random_state", XGB_RANDOM_STATE)

    X_train = df.iloc[train_idx][feature_cols]
    y_train = df.iloc[train_idx]["target"]
    X_test = df.iloc[test_idx][feature_cols]
    y_test = df.iloc[test_idx]["target"]

    model = XGBClassifier(**xgb_params, eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_label = (y_pred_proba >= 0.5).astype(int)

    return {
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "auc": roc_auc_score(y_test, y_pred_proba),
        "log_loss": log_loss(y_test, y_pred_proba, labels=[0, 1]),
        "brier": brier_score_loss(y_test, y_pred_proba),
        "accuracy": accuracy_score(y_test, y_pred_label),
    }


def run_xgb_walk_forward_evaluation(
    df: pd.DataFrame,
    folds: list,
    feature_cols: list[str],
    xgb_params: dict,
) -> pd.DataFrame:
    """
    Runs train_and_evaluate_fold_xgb across every fold using the same
    (pre-tuned) hyperparameters throughout. Returns a per-fold results
    DataFrame plus prints a summary - same shape as
    baseline_model.run_walk_forward_evaluation, so the two are directly
    comparable side by side.
    """
    rows = []
    for i, (train_idx, test_idx) in enumerate(folds, start=1):
        metrics = train_and_evaluate_fold_xgb(df, train_idx, test_idx, feature_cols, xgb_params)
        metrics["fold"] = i
        rows.append(metrics)

    results = pd.DataFrame(rows).set_index("fold")

    print(f"\n{'='*60}")
    print("XGBOOST - PER-FOLD RESULTS")
    print(f"{'='*60}")
    print(results.round(4).to_string())

    print(f"\n{'='*60}")
    print("XGBOOST - SUMMARY ACROSS ALL FOLDS")
    print(f"{'='*60}")
    for metric in ["auc", "log_loss", "brier", "accuracy"]:
        print(f"{metric:>10}: mean={results[metric].mean():.4f}  std={results[metric].std():.4f}  "
              f"min={results[metric].min():.4f}  max={results[metric].max():.4f}")

    return results


def compare_models(logreg_results: pd.DataFrame, xgb_results: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the logistic regression results (from baseline_model.py's
    run_walk_forward_evaluation) with XGBoost results (from this
    module) into one side-by-side comparison table, fold by fold, plus
    a summary of how often each model wins on each metric.
    """
    comparison = pd.DataFrame({
        "auc_logreg": logreg_results["auc_model"],
        "auc_xgb": xgb_results["auc"],
        "log_loss_logreg": logreg_results["log_loss_model"],
        "log_loss_xgb": xgb_results["log_loss"],
        "brier_logreg": logreg_results["brier_model"],
        "brier_xgb": xgb_results["brier"],
        "accuracy_logreg": logreg_results["accuracy_model"],
        "accuracy_xgb": xgb_results["accuracy"],
    })

    print(f"\n{'='*70}")
    print("LOGISTIC REGRESSION vs XGBOOST")
    print(f"{'='*70}")
    print(comparison.round(4).to_string())

    print(f"\n{'='*70}")
    print("HEAD-TO-HEAD SUMMARY")
    print(f"{'='*70}")
    for metric in ["auc", "log_loss", "brier", "accuracy"]:
        logreg_col, xgb_col = f"{metric}_logreg", f"{metric}_xgb"
        xgb_wins = (comparison[xgb_col] > comparison[logreg_col]) if metric in ("auc", "accuracy") \
            else (comparison[xgb_col] < comparison[logreg_col])
        print(f"{metric:>10}  logreg mean={comparison[logreg_col].mean():.4f}  "
              f"xgb mean={comparison[xgb_col].mean():.4f}  "
              f"xgb wins in {xgb_wins.sum()}/{len(comparison)} folds")

    return comparison