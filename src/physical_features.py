"""
Physical matchup features for the UFC fight predictor.

Two categories here, treated differently:

1. STATIC per-fight attributes (reach, height, age, stance at the time of
   the fight) - these are already present in fights_clean.csv as
   R_/B_ prefixed columns, computed by the original dataset. Turning them
   into a matchup feature (e.g. reach_diff = R - B) is a simple row-wise
   operation with NO lookahead risk, since both values are already
   pre-fight facts about each fighter (their reach doesn't change
   fight-to-fight the way form stats do).

2. FIGHTER-HISTORY-DEPENDENT features (days since last fight, whether
   this fight represents a weight class change for this fighter) - these
   genuinely need to walk each fighter's chronological history the same
   way elo_ratings.py and rolling_features.py do: look at what happened
   in THIS fighter's PREVIOUS fight, not any future one.

Assumes fights_df is already sorted strictly by date ascending.
"""

import pandas as pd
import numpy as np


def add_static_differentials(
    df: pd.DataFrame,
    reach_cols=("R_Reach_cms", "B_Reach_cms"),
    height_cols=("R_Height_cms", "B_Height_cms"),
    age_cols=("R_age", "B_age"),
) -> pd.DataFrame:
    """
    Adds simple row-wise differential columns. No lookahead risk - these
    are static per-fighter attributes already present pre-fight.

    Positive values favour the red corner (e.g. reach_diff > 0 means
    the red fighter has a longer reach).

    Silently skips a differential if either underlying column isn't
    present in df - check the returned column list to confirm which
    ones were actually added, since your exact column names may differ
    from the defaults given here.
    """
    df = df.copy()
    pairs = {
        "reach_diff": reach_cols,
        "height_diff": height_cols,
        "age_diff": age_cols,
    }
    for new_col, (r_col, b_col) in pairs.items():
        if r_col in df.columns and b_col in df.columns:
            df[new_col] = df[r_col] - df[b_col]
        else:
            print(f"Skipped {new_col}: missing {r_col} or {b_col} in df.columns")
    return df


def add_stance_matchup(
    df: pd.DataFrame,
    r_stance_col: str = "R_Stance",
    b_stance_col: str = "B_Stance",
) -> pd.DataFrame:
    """
    Encodes the stance matchup as a categorical string, e.g.
    'Orthodox_vs_Southpaw'. Also adds a same_stance binary flag, since
    "orthodox vs southpaw" is a commonly cited tactical mismatch in MMA
    (open-stance fights favour different angles/lead-hand exposure than
    closed-stance ones) - the model can use either the categorical
    matchup or the simpler binary flag, whichever it finds useful.

    No lookahead risk - stance is a static fighter attribute.
    """
    df = df.copy()
    if r_stance_col not in df.columns or b_stance_col not in df.columns:
        print(f"Skipped stance matchup: missing {r_stance_col} or {b_stance_col}")
        return df

    df["stance_matchup"] = df[r_stance_col].fillna("Unknown") + "_vs_" + df[b_stance_col].fillna("Unknown")
    df["same_stance"] = (df[r_stance_col] == df[b_stance_col]).astype(int)
    return df


def add_days_since_last_fight(
    df: pd.DataFrame,
    r_fighter_col: str = "R_fighter",
    b_fighter_col: str = "B_fighter",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    For each fighter, computes the number of days since their PREVIOUS
    fight in the dataset. A fighter's first tracked fight gets NaN (no
    prior fight to measure from) - handle at modelling time (e.g. a
    separate is_debut flag, or median-impute).

    Must be called on a df already sorted strictly by date ascending.
    Walks fights in order, tracking each fighter's last fight date in a
    dict, and only reads it BEFORE updating with the current fight's
    date - same no-lookahead discipline as elo_ratings.py.
    """
    df = df.copy()
    last_fight_date: dict[str, pd.Timestamp] = {}

    r_days_since = []
    b_days_since = []

    for _, row in df.iterrows():
        r_fighter = row[r_fighter_col]
        b_fighter = row[b_fighter_col]
        current_date = row[date_col]

        r_last = last_fight_date.get(r_fighter)
        b_last = last_fight_date.get(b_fighter)

        r_days_since.append((current_date - r_last).days if r_last is not None else np.nan)
        b_days_since.append((current_date - b_last).days if b_last is not None else np.nan)

        # Update AFTER reading - this fight becomes each fighter's "last
        # fight" only for fights that come after it.
        last_fight_date[r_fighter] = current_date
        last_fight_date[b_fighter] = current_date

    df["R_days_since_last_fight"] = r_days_since
    df["B_days_since_last_fight"] = b_days_since
    return df


def add_weight_class_change_flag(
    df: pd.DataFrame,
    r_fighter_col: str = "R_fighter",
    b_fighter_col: str = "B_fighter",
    weight_class_col: str = "weight_class",
) -> pd.DataFrame:
    """
    Flags whether this fight is at a DIFFERENT weight class than the
    fighter's previous fight - a proxy for "moving up/down in weight",
    which often signals a significant weight cut or a permanent class
    change. This is a proxy, not a direct measurement of weight cut
    severity (which would need actual cut poundage data we don't have) -
    say so explicitly in your write-up rather than overstating it.

    A fighter's first tracked fight gets NaN (no prior weight class to
    compare against) - treat as "unknown", not "no change".

    Same no-lookahead walk as add_days_since_last_fight: only reads a
    fighter's last weight class before updating it with the current one.
    """
    df = df.copy()
    last_weight_class: dict[str, str] = {}

    r_changed = []
    b_changed = []

    for _, row in df.iterrows():
        r_fighter = row[r_fighter_col]
        b_fighter = row[b_fighter_col]
        current_wc = row[weight_class_col]

        r_last_wc = last_weight_class.get(r_fighter)
        b_last_wc = last_weight_class.get(b_fighter)

        r_changed.append(np.nan if r_last_wc is None else int(current_wc != r_last_wc))
        b_changed.append(np.nan if b_last_wc is None else int(current_wc != b_last_wc))

        last_weight_class[r_fighter] = current_wc
        last_weight_class[b_fighter] = current_wc

    df["R_weight_class_changed"] = r_changed
    df["B_weight_class_changed"] = b_changed
    return df


def add_all_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper - runs all four functions above in sequence."""
    df = add_static_differentials(df)
    df = add_stance_matchup(df)
    df = add_days_since_last_fight(df)
    df = add_weight_class_change_flag(df)
    return df


if __name__ == "__main__":
    example = pd.DataFrame({
        "R_fighter": ["Alice", "Bob", "Alice"],
        "B_fighter": ["Bob", "Carol", "Carol"],
        "date": pd.to_datetime(["2020-01-01", "2020-04-01", "2020-08-01"]),
        "weight_class": ["Lightweight", "Lightweight", "Welterweight"],
        "R_Reach_cms": [180, 175, 180],
        "B_Reach_cms": [175, 190, 185],
        "R_Height_cms": [175, 178, 175],
        "B_Height_cms": [178, 182, 170],
        "R_age": [28, 30, 28],
        "B_age": [30, 25, 27],
        "R_Stance": ["Orthodox", "Southpaw", "Orthodox"],
        "B_Stance": ["Southpaw", "Orthodox", "Orthodox"],
    })

    result = add_all_physical_features(example)
    print(result[[
        "R_fighter", "B_fighter", "date", "reach_diff", "height_diff", "age_diff",
        "stance_matchup", "same_stance",
        "R_days_since_last_fight", "B_days_since_last_fight",
        "R_weight_class_changed", "B_weight_class_changed",
    ]].to_string())

    # Fight 3 (Alice vs Carol, 2020-08-01): Alice's last fight was
    # 2020-01-01 -> R_days_since_last_fight should be 213.
    # Carol's last fight was 2020-04-01 -> B_days_since_last_fight
    # should be 122.
    # This fight is Welterweight; Alice's fight 1 was Lightweight ->
    # R_weight_class_changed should be 1. Carol's fight 2 (as B) was
    # also Lightweight -> B_weight_class_changed should be 1 too.