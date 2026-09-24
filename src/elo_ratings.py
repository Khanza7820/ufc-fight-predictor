"""
ELO rating system for the UFC fight predictor.

Design decisions (documented here so they can be defended in an interview):

- Starting rating: 1500. Standard convention — every fighter is "average"
  until they've proven otherwise through results.
- Adaptive K-factor, based on each fighter's number of prior tracked
  fights (not calendar time): K_PROVISIONAL = 40 for fighters with fewer
  than PROVISIONAL_THRESHOLD (5) fights recorded so far, K_ESTABLISHED =
  32 once they pass that threshold. This mirrors how FIDE chess handles
  new players (higher K while a rating is unproven, lower once there's
  enough history to trust it) - the MMA case is stronger, since every
  fighter starts at a default 1500 regardless of true skill, so their
  early fights are the least informative and should move the rating
  faster. K is evaluated PER FIGHTER, independently for each side of the
  fight - a veteran fighting a debutant applies K_ESTABLISHED to the
  veteran's own update and K_PROVISIONAL to the debutant's.
  Deliberately NOT raising K simply because fights are infrequent
  (roughly 1-2/year): infrequency alone doesn't argue for reacting
  harder to each result, since MMA outcomes already carry more
  per-fight variance than chess (a single strike or bad decision can
  decide a fight independent of the real skill gap) - raising K
  indiscriminately would amplify that noise, not fix staleness. The
  experience-based schedule targets the actual problem (unproven
  ratings) without that side effect. Both K values, and the threshold,
  are legitimate hyperparameters to tune later via walk-forward CV
  (Week 3) - these are defensible starting points, not arbitrary
  guesses.
- Draws: both fighters receive an actual score (S) of 0.5, reflecting
  that neither outcome was "won" but the pre-fight expectation still
  gets corrected against.
- No-lookahead guarantee: for every fight, we record each fighter's
  PRE-FIGHT rating (their rating as it stood walking into that exact
  fight) BEFORE applying that fight's update. The update happens only
  after both pre-fight ratings have been captured. This is the same
  discipline as the rolling-stats function: shift-then-compute.
"""

import pandas as pd

INITIAL_RATING = 1500.0
K_PROVISIONAL = 40.0   # applied while a fighter has fewer than the threshold below
K_ESTABLISHED = 32.0   # applied once a fighter has enough tracked fights
PROVISIONAL_THRESHOLD = 5  # fights (not counting the current one)


def k_for_fighter(num_prior_fights: int) -> float:
    """
    Returns the K-factor to use for a fighter's rating update, based on
    how many fights of theirs we've already tracked (NOT counting the
    current fight - that count is only known after this fight, so using
    it here would be lookahead).
    """
    return K_PROVISIONAL if num_prior_fights < PROVISIONAL_THRESHOLD else K_ESTABLISHED


def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Probability that fighter A beats fighter B, given their current
    ratings, under the standard logistic ELO model.
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def compute_elo_ratings(
    fights_df: pd.DataFrame,
    r_fighter_col: str = "R_fighter",
    b_fighter_col: str = "B_fighter",
    winner_col: str = "Winner",
    initial_rating: float = INITIAL_RATING,
) -> pd.DataFrame:
    """
    Walks through fights_df IN ORDER (must already be sorted strictly by
    date ascending — this function trusts the input and does not re-sort)
    and computes each fighter's ELO rating, using an adaptive K-factor
    (see k_for_fighter / module docstring).

    For every row, adds:
        - R_elo_pre: red corner fighter's ELO immediately before this fight
        - B_elo_pre: blue corner fighter's ELO immediately before this fight
        - elo_diff: R_elo_pre - B_elo_pre (a convenient single feature)
        - R_fight_num: how many tracked fights this is for the red fighter
          (1 = their first fight in the dataset), B_fight_num: same for blue.
          Useful for sanity-checking the K schedule and for flagging
          fighters with very few fights at modelling time.

    Winner_col is expected to contain 'Red', 'Blue', or 'Draw'.

    Returns a copy of fights_df with the new columns added. Ratings AND
    fight counts are tracked in plain dicts, updated only after both
    pre-fight values for the current row have been read, so no fight ever
    sees its own outcome, its own fight count, or any fight after it.
    """
    df = fights_df.copy()
    ratings: dict[str, float] = {}
    fight_counts: dict[str, int] = {}

    r_elo_pre = []
    b_elo_pre = []
    r_fight_num = []
    b_fight_num = []

    for _, row in df.iterrows():
        r_fighter = row[r_fighter_col]
        b_fighter = row[b_fighter_col]

        # Look up current ratings and prior fight counts, defaulting new
        # fighters to initial_rating / 0 prior fights. This lookup happens
        # BEFORE any update for this fight - that's what guarantees no
        # lookahead, for both the rating AND the K-factor it determines.
        r_rating = ratings.get(r_fighter, initial_rating)
        b_rating = ratings.get(b_fighter, initial_rating)
        r_prior_fights = fight_counts.get(r_fighter, 0)
        b_prior_fights = fight_counts.get(b_fighter, 0)

        r_elo_pre.append(r_rating)
        b_elo_pre.append(b_rating)
        r_fight_num.append(r_prior_fights + 1)
        b_fight_num.append(b_prior_fights + 1)

        r_k = k_for_fighter(r_prior_fights)
        b_k = k_for_fighter(b_prior_fights)

        # Determine actual scores for this fight
        winner = row[winner_col]
        if winner == "Red":
            r_score, b_score = 1.0, 0.0
        elif winner == "Blue":
            r_score, b_score = 0.0, 1.0
        elif winner == "Draw":
            r_score, b_score = 0.5, 0.5
        else:
            # Unexpected value (e.g. NaN, "NC") - skip the rating update
            # for this fight but still record pre-fight ratings above,
            # and still count it as a fight for both fighters (it happened,
            # even if we can't score it). Flag these in your lookahead
            # audit rather than silently dropping them.
            ratings[r_fighter] = r_rating
            ratings[b_fighter] = b_rating
            fight_counts[r_fighter] = r_prior_fights + 1
            fight_counts[b_fighter] = b_prior_fights + 1
            continue

        # Expected scores given pre-fight ratings
        r_expected = expected_score(r_rating, b_rating)
        b_expected = expected_score(b_rating, r_rating)

        # Update AFTER recording pre-fight values - this is the update
        # this fight's outcome earns, applied only for future fights.
        # Each side uses ITS OWN adaptive K, independent of the other's.
        ratings[r_fighter] = r_rating + r_k * (r_score - r_expected)
        ratings[b_fighter] = b_rating + b_k * (b_score - b_expected)
        fight_counts[r_fighter] = r_prior_fights + 1
        fight_counts[b_fighter] = b_prior_fights + 1

    df["R_elo_pre"] = r_elo_pre
    df["B_elo_pre"] = b_elo_pre
    df["elo_diff"] = df["R_elo_pre"] - df["B_elo_pre"]
    df["R_fight_num"] = r_fight_num
    df["B_fight_num"] = b_fight_num

    return df


if __name__ == "__main__":
    # Manual validation example - walk through fights by hand and confirm
    # the code matches your own arithmetic. Do this with your real data
    # too before trusting the feature (pick a real fighter, trace their
    # ELO trajectory across their actual fight history).
    example = pd.DataFrame({
        "R_fighter": ["Alice", "Bob", "Alice"],
        "B_fighter": ["Bob", "Carol", "Carol"],
        "Winner": ["Red", "Blue", "Draw"],
        "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
    })

    result = compute_elo_ratings(example)
    print(result[[
        "R_fighter", "B_fighter", "Winner",
        "R_elo_pre", "B_elo_pre", "elo_diff",
        "R_fight_num", "B_fight_num",
    ]])

    # All fighters here have 0 prior fights at first appearance, so
    # K_PROVISIONAL (40) applies throughout this toy example - it's too
    # short to reach K_ESTABLISHED (5+ fights). Check against a longer
    # real fighter history to see the K=40 -> K=32 transition in action.
    #
    # Fight 1: Alice(1500, fight #1) vs Bob(1500, fight #1), Red wins.
    #   expected = 0.5 each, K=40 both sides
    #   -> Alice: 1500 + 40*(1-0.5) = 1520, Bob: 1500 + 40*(0-0.5) = 1480
    # Fight 2: Bob(1480, fight #2) vs Carol(1500, fight #1, new), Blue wins.
    #   Bob's pre-fight rating here should show 1480 (his post-fight-1 rating)
    # Fight 3: Alice(1520, fight #2) vs Carol(post-fight-2 rating, fight #2), Draw.
    #   Alice's pre-fight rating here should show 1520 (unchanged since fight 1)