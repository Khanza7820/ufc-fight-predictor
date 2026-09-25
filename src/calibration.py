"""
Probability calibration for the UFC fight predictor.

Why this matters: an XGBoost model's raw predict_proba output is NOT
guaranteed to mean what it claims. A model can say "70% chance Red
wins" and be right only 55% of the time when you check every fight it
said that about - the RANKING of predictions can be good (decent AUC)
while the actual probability VALUES are systematically off (typically
overconfident - tree ensembles tend to push probabilities toward 0 and
1 more than the true frequency warrants). This matters enormously for
this project specifically, because Week 5's Kelly criterion sizing
takes these probabilities as ground truth - miscalibrated probabilities
mean miscalibrated bets, regardless of how good the model's ranking is.

Two calibration methods, both standard:
- PLATT SCALING (sigmoid): fits a 1-D logistic regression mapping raw
  probability -> calibrated probability. Simple, assumes the
  miscalibration follows a roughly sigmoid-shaped correction. Works
  well when you don't have a huge amount of calibration data.
- ISOTONIC REGRESSION: fits a non-decreasing step function mapping raw
  probability -> calibrated probability. More flexible than Platt (no
  assumed shape), but needs more data to avoid overfitting the
  calibration curve itself.

LOOKAHEAD DISCIPLINE, one level deeper than anywhere else in this
project: a calibrator cannot be fit on the same data used to train the
underlying model (it would just learn "trust the model's own training
performance," not correct genuine miscalibration on new data) and
obviously cannot touch the test set either. So each fold's TRAINING
data is itself split chronologically: the earlier portion trains the
XGBoost model, the most recent slice of training data (never touched
by model training) calibrates the probability mapping, and only then
does the actual test set get scored - by the model AND by both
calibrators, so all three (raw, Platt, isotonic) can be compared on
the exact same unseen fights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss


def calibrate_and_evaluate_fold(
    df: pd.DataFrame,
    train_idx: list[int],
    test_idx: list[int],
    feature_cols: list[str],
    xgb_params: dict,
    calibration_fraction: float = 0.15,
) -> dict:
    """
    For one fold:
    1. Splits train_idx chronologically into model_train (earlier) and
       calib (most recent calibration_fraction of train_idx).
    2. Trains XGBoost on model_train only.
    3. Gets raw probabilities on calib - fits Platt (logistic
       regression on the raw probability as a single feature) and
       isotonic regression, mapping raw -> calibrated probability.
    4. Scores raw, Platt-calibrated, and isotonic-calibrated
       probabilities all on the SAME test set, so they're directly
       comparable.

    Returns a dict of log-loss and Brier score for all three variants.
    """
    n_train = len(train_idx)
    split_point = int(n_train * (1 - calibration_fraction))
    model_train_idx = train_idx[:split_point]
    calib_idx = train_idx[split_point:]

    X_model_train = df.iloc[model_train_idx][feature_cols]
    y_model_train = df.iloc[model_train_idx]["target"]
    X_calib = df.iloc[calib_idx][feature_cols]
    y_calib = df.iloc[calib_idx]["target"]
    X_test = df.iloc[test_idx][feature_cols]
    y_test = df.iloc[test_idx]["target"]

    model = XGBClassifier(**xgb_params, eval_metric="logloss")
    model.fit(X_model_train, y_model_train)

    raw_calib_probs = model.predict_proba(X_calib)[:, 1]
    raw_test_probs = model.predict_proba(X_test)[:, 1]

    # Platt scaling: logistic regression on the raw probability as the
    # sole input feature, mapping it to a corrected probability.
    platt = LogisticRegression()
    platt.fit(raw_calib_probs.reshape(-1, 1), y_calib)
    platt_test_probs = platt.predict_proba(raw_test_probs.reshape(-1, 1))[:, 1]

    # Isotonic regression: non-decreasing step-function mapping.
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(raw_calib_probs, y_calib)
    isotonic_test_probs = isotonic.predict(raw_test_probs)

    return {
        "n_model_train": len(model_train_idx),
        "n_calib": len(calib_idx),
        "n_test": len(test_idx),
        "log_loss_raw": log_loss(y_test, raw_test_probs, labels=[0, 1]),
        "log_loss_platt": log_loss(y_test, platt_test_probs, labels=[0, 1]),
        "log_loss_isotonic": log_loss(y_test, isotonic_test_probs, labels=[0, 1]),
        "brier_raw": brier_score_loss(y_test, raw_test_probs),
        "brier_platt": brier_score_loss(y_test, platt_test_probs),
        "brier_isotonic": brier_score_loss(y_test, isotonic_test_probs),
        # Also return the raw arrays for this fold's test predictions vs
        # actual outcomes, so a reliability diagram can be built later
        # from the POOLED predictions across every fold (Week 4 Day 3-4).
        "test_actual": y_test.values,
        "raw_probs": raw_test_probs,
        "platt_probs": platt_test_probs,
        "isotonic_probs": isotonic_test_probs,
    }


def run_calibration_evaluation(
    df: pd.DataFrame,
    folds: list,
    feature_cols: list[str],
    xgb_params: dict,
    calibration_fraction: float = 0.15,
) -> tuple[pd.DataFrame, dict]:
    """
    Runs calibrate_and_evaluate_fold across every fold. Returns:
    - a per-fold summary DataFrame (log-loss and Brier for raw/Platt/
      isotonic, so you can see fold-by-fold whether calibration helped)
    - a dict of POOLED arrays (all folds' test predictions and actuals
      concatenated together) for building a reliability diagram across
      the full test period, not just one fold at a time.
    """
    rows = []
    pooled = {"test_actual": [], "raw_probs": [], "platt_probs": [], "isotonic_probs": []}

    for i, (train_idx, test_idx) in enumerate(folds, start=1):
        fold_result = calibrate_and_evaluate_fold(
            df, train_idx, test_idx, feature_cols, xgb_params, calibration_fraction
        )
        for key in pooled:
            pooled[key].append(fold_result[key])

        row = {k: v for k, v in fold_result.items() if k not in pooled}
        row["fold"] = i
        rows.append(row)

    for key in pooled:
        pooled[key] = np.concatenate(pooled[key])

    results = pd.DataFrame(rows).set_index("fold")

    print(f"\n{'='*70}")
    print("CALIBRATION: RAW vs PLATT vs ISOTONIC (per fold)")
    print(f"{'='*70}")
    display_cols = ["n_model_train", "n_calib", "n_test",
                     "brier_raw", "brier_platt", "brier_isotonic",
                     "log_loss_raw", "log_loss_platt", "log_loss_isotonic"]
    print(results[display_cols].round(4).to_string())

    print(f"\n{'='*70}")
    print("SUMMARY ACROSS ALL FOLDS")
    print(f"{'='*70}")
    for metric in ["brier", "log_loss"]:
        raw_mean = results[f"{metric}_raw"].mean()
        platt_mean = results[f"{metric}_platt"].mean()
        iso_mean = results[f"{metric}_isotonic"].mean()
        platt_wins = (results[f"{metric}_platt"] < results[f"{metric}_raw"]).sum()
        iso_wins = (results[f"{metric}_isotonic"] < results[f"{metric}_raw"]).sum()
        print(f"{metric:>10}  raw={raw_mean:.4f}  platt={platt_mean:.4f} (beat raw in {platt_wins}/{len(results)})  "
              f"isotonic={iso_mean:.4f} (beat raw in {iso_wins}/{len(results)})")

    return results, pooled


def plot_reliability_diagram(
    pooled: dict,
    n_bins: int = 10,
    variants: tuple = ("raw_probs", "platt_probs", "isotonic_probs"),
    labels: tuple = ("Raw XGBoost", "Platt-calibrated", "Isotonic-calibrated"),
    save_path: str = None,
) -> pd.DataFrame:
    """
    Builds the reliability diagram (calibration plot): x-axis is
    predicted probability, binned; y-axis is the ACTUAL win rate within
    each bin, computed from the pooled test-set predictions across
    every fold (never touching training data - these are all genuine
    out-of-sample predictions). Perfect calibration is the diagonal
    line: a bin of predictions around 0.7 should see an actual win
    rate of about 70%.

    Also returns the underlying bin-by-bin table (predicted midpoint,
    actual rate, count per bin) so the numbers behind the plot can be
    inspected directly, not just eyeballed from the chart.

    Uses pooled predictions (all 16 folds' test sets concatenated)
    rather than one fold at a time, since a single fold often has too
    few fights per probability bin to give a stable estimate of the
    true win rate in that bin.

    save_path: if given, saves the figure to this path BEFORE
    plt.show() is called. Some notebook/inline matplotlib backends
    clear or close the current figure on show(), so saving after the
    fact (e.g. a separate plt.savefig() call in the notebook, after
    this function returns) can silently write a blank or near-empty
    image - this was caught in Week 6 when a saved calibration PNG
    came back at ~4.7KB (vs tens of KB expected) and rendered blank
    inside the compiled tearsheet PDF. Saving inside the function,
    against the actual figure object, avoids depending on matplotlib's
    global "current figure" state at all.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    all_bin_tables = []

    for variant_key, label in zip(variants, labels):
        probs = pooled[variant_key]
        actual = pooled["test_actual"]

        bin_ids = np.digitize(probs, bin_edges) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)

        bin_mean_pred = []
        bin_actual_rate = []
        bin_counts = []
        for b in range(n_bins):
            mask = bin_ids == b
            count = mask.sum()
            bin_counts.append(count)
            if count > 0:
                bin_mean_pred.append(probs[mask].mean())
                bin_actual_rate.append(actual[mask].mean())
            else:
                bin_mean_pred.append(np.nan)
                bin_actual_rate.append(np.nan)

        bin_table = pd.DataFrame({
            "bin_mean_predicted": bin_mean_pred,
            "actual_win_rate": bin_actual_rate,
            "n_fights": bin_counts,
        })
        bin_table["variant"] = label
        all_bin_tables.append(bin_table)

        ax.plot(bin_mean_pred, bin_actual_rate, marker="o", label=label)

    ax.set_xlabel("Predicted probability (Red wins)")
    ax.set_ylabel("Actual win rate")
    ax.set_title("Reliability Diagram: Predicted vs Actual Win Rate")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

    return pd.concat(all_bin_tables, ignore_index=True)