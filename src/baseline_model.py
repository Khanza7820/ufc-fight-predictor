"""
Baseline model for the UFC fight predictor: logistic regression, trained
and scored inside the purged walk-forward CV loop from walk_forward_cv.py.

Design decisions (documented for defense in an interview):

- OUTPUTS PROBABILITIES, NOT LABELS. predict_proba, not predict. A
  probability is the actual deliverable for this project (it's what
  feeds the Kelly criterion and calibration work in Weeks 4-5) - a
  bare win/loss label throws away the information the whole rest of
  the project depends on.

- TARGET: Winner == 'Red' -> 1, 'Blue' -> 0. Draws are EXCLUDED from
  the modelling target (documented explicitly, not silently dropped):
  they're ~1.8% of fights (109/6011), and there's no clean way to
  represent a draw in binary win-probability terms without a separate
  multi-class framework, which is out of scope for a baseline. This
  matches your Week 1 cleaning notes, which already flagged draws as
  "retained but excluded at modelling stage."

- IMPUTATION AND SCALING ARE FIT PER-FOLD, ON TRAINING DATA ONLY, then
  APPLIED to that fold's test data. Never fit on the full dataset -
  this is the same no-lookahead discipline as the feature engineering,
  just showing up at the modelling stage instead. Many features (rolling
  stats history, days-since-last-fight, ELO fight-count-dependent K)
  are NaN for a fighter's early/debut fights - median imputation fills
  these, and the median itself must come only from what a fold's
  training data actually contained.

- FEATURES ARE NUMERIC-ONLY for this baseline (auto-detected, with an
  explicit exclude list for IDs/leakage-risk columns below).
  Categorical columns (weight_class, stance_matchup, raw Stance) are
  left out of the baseline deliberately - encoding them properly
  (one-hot, or letting XGBoost handle them natively) is a Week 3
  Day 5-6 upgrade, not a baseline requirement. Note this explicitly so
  it doesn't look like an oversight.

- LOGISTIC REGRESSION is the chosen baseline algorithm because it's
  the simplest model that outputs calibrated-ish probabilities and
  gives a real floor to beat before reaching for XGBoost - matching
  your project's own Week 3 goal ("baseline to beat established").
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score

# Columns that must NEVER be used as model features - identifiers, the
# target itself, raw dates, and text/categorical columns excluded from
# this baseline (see module docstring).
EXCLUDE_COLS = {
    "R_fighter", "B_fighter", "date", "Winner", "target",
    "weight_class", "stance_matchup", "R_Stance", "B_Stance",
    "Referee", "location",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Auto-selects numeric feature columns, excluding anything in
    EXCLUDE_COLS and anything that isn't actually numeric (object/
    string dtype columns some other part of the pipeline might have
    left behind).
    """
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    return [c for c in numeric_cols if c not in EXCLUDE_COLS]


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a binary `target` column (1 = Red wins, 0 = Blue wins) and
    DROPS draw rows. Returns a new dataframe - does not mutate the
    original, and does not silently reindex (the original positional
    index is preserved via .copy(), so fold indices computed on the
    FULL df before calling this will NOT line up afterward - call this
    BEFORE generating folds, not after, to avoid an index mismatch bug).
    """
    df = df.copy()
    mask = df["Winner"].isin(["Red", "Blue"])
    dropped = (~mask).sum()
    print(f"Dropping {dropped} draw/invalid-winner rows ({dropped/len(df):.1%} of data) from modelling target")
    df = df[mask].reset_index(drop=True)
    df["target"] = (df["Winner"] == "Red").astype(int)
    return df


def naive_baseline_fold(
    df: pd.DataFrame,
    train_idx: list[int],
    test_idx: list[int],
) -> dict:
    """
    The floor the real model must beat: for this fold, computes the
    Red-win rate WITHIN THIS FOLD'S OWN TRAINING SET ONLY (not the
    overall dataset average, and never touching test data - same
    no-lookahead discipline as everything else), then predicts that
    single constant probability for every fight in the test set.

    This is deliberately the simplest possible "model": it knows
    nothing about either fighter, it just always guesses the
    historical favourite-corner rate. Any real feature-based model
    that can't beat this consistently isn't actually learning
    anything from the fighters' attributes.

    AUC is not meaningful for a constant prediction (there's no ranking
    information - every fight gets the identical score), so it's
    reported as NaN rather than a misleading 0.5.
    """
    y_train = df.iloc[train_idx]["target"]
    y_test = df.iloc[test_idx]["target"]

    naive_prob = y_train.mean()  # this fold's training Red-win rate, nothing else
    y_pred_proba = np.full(len(test_idx), naive_prob)
    y_pred_label = int(naive_prob >= 0.5)

    return {
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "auc": np.nan,  # undefined for a constant predictor - see docstring
        "log_loss": log_loss(y_test, y_pred_proba, labels=[0, 1]),
        "brier": brier_score_loss(y_test, y_pred_proba),
        "accuracy": accuracy_score(y_test, [y_pred_label] * len(test_idx)),
    }


def train_and_evaluate_fold(
    df: pd.DataFrame,
    train_idx: list[int],
    test_idx: list[int],
    feature_cols: list[str],
) -> dict:
    """
    Trains a logistic regression on one fold's training data, scores it
    on that fold's test data. Imputation and scaling are fit on
    train_idx ONLY, then applied (not refit) to test_idx - see module
    docstring for why this matters.

    Returns a dict of metrics for this single fold.
    """
    X_train = df.iloc[train_idx][feature_cols]
    y_train = df.iloc[train_idx]["target"]
    X_test = df.iloc[test_idx][feature_cols]
    y_test = df.iloc[test_idx]["target"]

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)  # transform only, never fit

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)  # transform only, never fit

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # P(Red wins)
    y_pred_label = (y_pred_proba >= 0.5).astype(int)

    return {
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "auc": roc_auc_score(y_test, y_pred_proba),
        "log_loss": log_loss(y_test, y_pred_proba, labels=[0, 1]),
        "brier": brier_score_loss(y_test, y_pred_proba),
        "accuracy": accuracy_score(y_test, y_pred_label),
    }


def run_walk_forward_evaluation(
    df: pd.DataFrame,
    folds: list,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Runs train_and_evaluate_fold AND naive_baseline_fold across every
    fold, returns a DataFrame comparing them side by side. The naive
    baseline is the real floor - a feature-based model that doesn't
    consistently beat it isn't demonstrating any learned signal from
    the fighters' attributes, whatever its raw AUC looks like in
    isolation.
    """
    rows = []
    for i, (train_idx, test_idx) in enumerate(folds, start=1):
        model_metrics = train_and_evaluate_fold(df, train_idx, test_idx, feature_cols)
        naive_metrics = naive_baseline_fold(df, train_idx, test_idx)

        row = {"fold": i, "n_train": model_metrics["n_train"], "n_test": model_metrics["n_test"]}
        for metric in ["auc", "log_loss", "brier", "accuracy"]:
            row[f"{metric}_model"] = model_metrics[metric]
            row[f"{metric}_naive"] = naive_metrics[metric]
        rows.append(row)

    results = pd.DataFrame(rows).set_index("fold")

    print(f"\n{'='*70}")
    print("PER-FOLD: MODEL vs NAIVE BASELINE")
    print(f"{'='*70}")
    display_cols = ["n_train", "n_test", "auc_model", "log_loss_model", "log_loss_naive",
                     "brier_model", "brier_naive", "accuracy_model", "accuracy_naive"]
    print(results[display_cols].round(4).to_string())

    print(f"\n{'='*70}")
    print("SUMMARY ACROSS ALL FOLDS")
    print(f"{'='*70}")
    for metric in ["log_loss", "brier", "accuracy"]:
        model_col, naive_col = f"{metric}_model", f"{metric}_naive"
        beats_naive = (results[model_col] < results[naive_col]) if metric != "accuracy" else (results[model_col] > results[naive_col])
        print(f"{metric:>10}  model mean={results[model_col].mean():.4f}  "
              f"naive mean={results[naive_col].mean():.4f}  "
              f"model beat naive in {beats_naive.sum()}/{len(results)} folds")
    print(f"{'auc':>10}  model mean={results['auc_model'].mean():.4f}  (naive: undefined for a constant predictor)")

    return results