"""
Odds utilities for the UFC fight predictor backtest.

American odds format:
- POSITIVE odds (e.g. +235): the underdog. Represents how much profit a
  $100 bet would win. Implied probability = 100 / (odds + 100).
- NEGATIVE odds (e.g. -130): the favourite. Represents how much you'd
  need to bet to win $100 profit. Implied probability =
  -odds / (-odds + 100).

Note: summing R_implied_prob + B_implied_prob across a single fight
will NOT equal exactly 1.0 - it'll be slightly ABOVE 1.0. That gap is
the bookmaker's built-in margin ("vig" / "juice" / "overround") - their
profit mechanism, baked into both sides of the line. This is expected
and correct, not a bug - it's the real-world quantity your model's
probability needs to beat to have genuine betting edge, not a
theoretical fair-odds baseline.
"""

import pandas as pd
import numpy as np


def american_odds_to_prob(odds: float) -> float:
    """
    Converts a single American odds value to implied probability.
    Handles both positive (underdog) and negative (favourite) odds.
    """
    if pd.isna(odds):
        return np.nan
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)


def add_implied_probabilities(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds R_implied_prob, B_implied_prob, and overround (the sum of both
    - bookmaker's margin) columns to odds_df. Vectorized using
    np.where rather than the scalar function above, for speed on the
    full dataset.
    """
    df = odds_df.copy()

    for corner in ["R", "B"]:
        odds_col = f"{corner}_odds"
        prob_col = f"{corner}_implied_prob"
        df[prob_col] = np.where(
            df[odds_col] > 0,
            100 / (df[odds_col] + 100),
            -df[odds_col] / (-df[odds_col] + 100),
        )
        # Preserve NaN for missing odds rather than letting the
        # comparison above silently produce a wrong value
        df.loc[df[odds_col].isna(), prob_col] = np.nan

    df["overround"] = df["R_implied_prob"] + df["B_implied_prob"]
    return df


def join_odds_to_features(
    features_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    r_fighter_col: str = "R_fighter",
    b_fighter_col: str = "B_fighter",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Joins odds_df (with implied probabilities already added - call
    add_implied_probabilities first) onto features_df, matching on
    (R_fighter, B_fighter, date). Prints match statistics so silent
    join failures (name spelling mismatches, date misalignment) are
    visible rather than hidden.

    Returns the merged dataframe (inner join - only fights present in
    BOTH datasets are kept, since a backtest needs both features and
    real odds for every row).
    """
    odds_cols = [r_fighter_col, b_fighter_col, date_col,
                 "R_odds", "B_odds", "R_implied_prob", "B_implied_prob", "overround"]
    odds_subset = odds_df[odds_cols].copy()

    merged = features_df.merge(
        odds_subset,
        on=[r_fighter_col, b_fighter_col, date_col],
        how="inner",
        suffixes=("", "_odds_source"),
    )

    print(f"features_df: {len(features_df)} fights")
    print(f"odds_df: {len(odds_df)} fights")
    print(f"Matched (inner join): {len(merged)} fights ({len(merged)/len(features_df):.1%} of features_df)")

    unmatched = len(features_df) - len(merged)
    if unmatched > 0:
        print(f"WARNING: {unmatched} fights in features_df had no matching odds row. "
              f"Common causes: fighter name spelling differences between sources, "
              f"or fights outside the odds dataset's date coverage. "
              f"Inspect before assuming these are all legitimately missing odds.")

    # Drop rows where odds were present in the file but recorded as NaN
    # (the ~3% missing-odds fights identified earlier)
    before_dropna = len(merged)
    merged = merged.dropna(subset=["R_odds", "B_odds"]).reset_index(drop=True)
    dropped_for_nan = before_dropna - len(merged)
    if dropped_for_nan > 0:
        print(f"Dropped {dropped_for_nan} additional fights with missing odds values.")

    print(f"Final backtest-ready dataset: {len(merged)} fights")
    return merged


if __name__ == "__main__":
    # Manual validation: known American odds -> known implied probabilities
    test_cases = [
        (-150, 0.6),      # -150 favourite -> 60% implied
        (150, 0.4),       # +150 underdog -> 40% implied
        (-110, 0.5238),   # standard "pick 'em" vig line
        (100, 0.5),       # even odds
    ]
    for odds, expected in test_cases:
        result = american_odds_to_prob(odds)
        status = "OK" if abs(result - expected) < 0.001 else "MISMATCH"
        print(f"odds={odds:>6}  implied_prob={result:.4f}  expected~{expected:.4f}  [{status}]")

    # Confirm overround is realistic (should be slightly above 1.0 -
    # e.g. 1.02-1.10 is typical for UFC lines)
    sample = pd.DataFrame({"R_odds": [-150, 235], "B_odds": [130, -320]})
    sample_with_probs = add_implied_probabilities(sample)
    print(f"\nSample overround values: {sample_with_probs['overround'].tolist()}")
    print("(should be slightly above 1.0 - this is the bookmaker's margin)")