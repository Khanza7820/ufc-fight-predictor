"""
Rolling pre-fight feature computation for the UFC fight predictor.

Window logic: uses the fighter's last 5 fights strictly before the current
fight date. If fewer than 5 prior fights exist, uses however many are
available (down to a minimum of 1). Fighters with 0 prior fights get NaN
(no history to compute stats from) - handle this at the modelling stage
(e.g. median-impute or use a "debutant" flag feature).

Assumes `fights_df` is already sorted strictly by fight date ascending,
and is in "long" format: one row per (fighter, fight) so each bout
appears twice (once per fighter). Adjust column names to match your
actual schema.
"""

import pandas as pd
import numpy as np

MAX_WINDOW = 5

def compute_rolling_stats(fights_df: pd.DataFrame, fighter_col: str, date_col: str,
                           stat_cols: list[str]) -> pd.DataFrame:
    """
    For each row (a fighter's upcoming fight), compute the mean of each
    stat in `stat_cols` over that fighter's last up-to-5 fights strictly
    before this fight's date.

    Returns a copy of fights_df with new columns: f"{col}_roll5" for each
    stat in stat_cols.

    IMPORTANT: fights_df must already be sorted by date_col ascending
    before calling this. This function does not re-sort, so lookahead
    bias will creep in silently if the input isn't sorted correctly.
    """
    df = fights_df.copy()

    for col in stat_cols:
        new_col = f"{col}_roll5"
        df[new_col] = np.nan

    # Group by fighter, then walk through their fights in date order
    for fighter, group in df.groupby(fighter_col, sort=False):
        idx = group.index  # preserves original row order (already date-sorted)
        for col in stat_cols:
            new_col = f"{col}_roll5"
            # shift(1) ensures we never include the current (upcoming) fight -
            # only fights strictly before it
            rolling_mean = (
                group[col]
                .shift(1)
                .rolling(window=MAX_WINDOW, min_periods=1)
                .mean()
            )
            df.loc[idx, new_col] = rolling_mean.values

    return df


if __name__ == "__main__":
    # Minimal example showing the adaptive-window behaviour
    example = pd.DataFrame({
        "fighter": ["A", "A", "A", "A", "A", "A", "A"],
        "date":    pd.date_range("2020-01-01", periods=7, freq="90D"),
        "landed_per_min": [3.0, 4.0, 2.0, 5.0, 3.5, 4.5, 6.0],
    })

    result = compute_rolling_stats(
        example, fighter_col="fighter", date_col="date",
        stat_cols=["landed_per_min"]
    )
    print(result[["fighter", "date", "landed_per_min", "landed_per_min_roll5"]])
    # Fight 1: NaN (no prior fights)
    # Fight 2: mean of fight 1 only
    # Fight 3: mean of fights 1-2
    # ...
    # Fight 6: mean of fights 1-5 (first full window)
    # Fight 7: mean of fights 2-6 (window slides, still only 5)