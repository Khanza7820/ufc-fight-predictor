"""
SHAP explainability for the UFC fight predictor's final XGBoost model.

IMPORTANT DISTINCTION FROM EVERYTHING ELSE IN THIS PROJECT: every other
module (walk_forward_cv, baseline_model, xgboost_model, calibration)
exists to answer "how accurate is this model on fights it hasn't seen."
This module answers a different question entirely: "why does the
DEPLOYED model make the predictions it makes." Because of that
difference, this is the one place in the project where training on
the FULL clean dataset (not a held-out fold) is the correct choice,
not a lookahead violation - SHAP values explain the model's learned
behaviour, they are not a claim about predictive accuracy on unseen
data (that claim was already established, honestly, by the walk-
forward CV results in 03_modelling.ipynb and calibration.py). The
model interpreted here is the one that would actually be shipped.

Uses shap.TreeExplainer, which computes exact (not approximated) SHAP
values for tree ensembles like XGBoost - each fight's prediction is
decomposed into: base rate (the average prediction across all fights)
+ each feature's individual contribution, summing exactly to that
fight's final predicted probability (in log-odds space).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier


def train_final_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    xgb_params: dict,
) -> XGBClassifier:
    """
    Trains ONE final XGBoost model on ALL of df (every clean, non-draw
    fight), using the hyperparameters already tuned via Optuna. This is
    the "production" model - the one whose reasoning SHAP will explain.
    Not used for further accuracy evaluation (that's already done).
    """
    X = df[feature_cols]
    y = df["target"]
    model = XGBClassifier(**xgb_params, eval_metric="logloss")
    model.fit(X, y)
    print(f"Final model trained on {len(df)} fights, {len(feature_cols)} features.")
    return model


def compute_shap_values(model: XGBClassifier, X: pd.DataFrame):
    """
    Computes SHAP values for every row in X using TreeExplainer (exact
    for tree models, not approximated). Returns the shap.Explanation
    object, which carries both the values and the underlying feature
    data together - needed for the summary and waterfall plots below.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return shap_values


def plot_shap_summary(shap_values, feature_cols: list[str], max_display: int = 15):
    """
    The SHAP summary (beeswarm) plot: one dot per fight per feature,
    x-axis is that feature's impact on the prediction for that fight
    (positive = pushed toward Red winning), colour is the feature's
    actual value (high/low) for that fight. Features are ranked
    top-to-bottom by average |impact| across all fights - this IS the
    feature importance ranking, but richer than a bar chart, since it
    also shows whether high or low values of a feature push the
    prediction in which direction.
    """
    shap.summary_plot(shap_values, feature_names=feature_cols, max_display=max_display, show=False)
    plt.tight_layout()
    plt.show()


def get_top_features(shap_values, feature_cols: list[str], top_n: int = 5) -> pd.DataFrame:
    """
    Returns a ranked table of the top_n features by mean |SHAP value|
    across all fights - the numeric backbone behind the summary plot,
    for writing an explicit interpretation rather than eyeballing the
    chart.
    """
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    ranking = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return ranking.head(top_n)


def plot_shap_waterfall(shap_values, row_index: int, df: pd.DataFrame = None):
    """
    Waterfall plot for ONE specific fight: starts at the model's base
    rate (average prediction across all training fights), then shows
    each feature adding or subtracting from that base value, arriving
    at this specific fight's final predicted probability. This is the
    "why did the model think THIS fighter would win THIS fight" chart -
    pick a row_index for a fight you find interesting or want to
    narrate in an interview.

    If df is provided, prints the fighters' names and actual date for
    context alongside the plot.
    """
    if df is not None:
        row = df.iloc[row_index]
        fighters = f"{row.get('R_fighter', '?')} vs {row.get('B_fighter', '?')}"
        date = row.get("date", "?")
        print(f"Fight: {fighters}  |  Date: {date}")

    shap.plots.waterfall(shap_values[row_index], show=False)
    plt.tight_layout()
    plt.show()


def plot_shap_dependence(shap_values, feature_cols: list[str], feature_name: str):
    """
    Dependence plot for ONE feature: x-axis is that feature's actual
    value across all fights, y-axis is its SHAP contribution for that
    fight. Reveals the SHAPE of the relationship that the beeswarm plot
    only hints at via colour - e.g. whether the effect is roughly
    linear across the whole range, or whether it's strong up to some
    threshold and then plateaus/reverses.
    """
    if feature_name not in feature_cols:
        raise ValueError(f"'{feature_name}' not found in feature_cols")

    shap.plots.scatter(shap_values[:, feature_name], show=False)
    plt.tight_layout()
    plt.show()