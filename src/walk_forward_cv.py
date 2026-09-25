"""
Purged walk-forward cross-validation for the UFC fight predictor.

Design decisions (documented for defense in an interview):

- Fight-count-based windows, not calendar-date windows. UFC event
  frequency changed drastically across the dataset's history (a handful
  of events/year in the mid-90s vs 40+ now) - a fixed 6-month window
  would contain ~5 fights in 1996 and ~150 in 2019, making folds
  incomparable. A fixed fight-count keeps every fold's test set roughly
  the same statistical weight.
- Expanding training window, not rolling. Every fold's training set
  starts from fight #0 and grows - we never discard old fights. This
  matches how the ELO system already treats history (a fighter's rating
  reflects their ENTIRE tracked past, not just a recent window), so it
  would be inconsistent to then throw away old fights at the modelling
  stage.
- Non-overlapping test folds by default (step_size == test_size): each
  fight is used as a test example exactly once across the whole walk,
  which keeps the reported metrics honest - no fight's prediction
  quality gets counted twice.
- "Purging" here reduces to a single invariant, explained in the
  validate_no_leakage function below: every fight in a training set
  must have a date strictly before every fight in its corresponding
  test set. Because none of this project's features (ELO, rolling
  stats, days-since-last-fight) use an OVERLAPPING time window that
  could straddle a fold boundary - each is a function of a single
  fighter's own strictly-prior fights - there is no additional
  embargo/buffer needed beyond that strict date ordering. This is
  verified programmatically below, not just assumed.

Assumes df is already sorted strictly by date ascending (Week 1/2
guarantee) - this function trusts that ordering and does not re-sort.
"""

import pandas as pd


def _snap_to_date_boundary(df: pd.DataFrame, idx: int, date_col: str) -> int:
    """
    UFC events are CARDS - multiple fights share the exact same date
    (an entire night's fights are all dated identically). A fold
    boundary that lands in the middle of a card would put some of that
    night's fights in training and others in test, sharing a date
    across the split - a real leak, since those fights are, for all
    practical purposes, simultaneous.

    Advances idx forward (never backward, to avoid shrinking training
    windows unexpectedly) until it lands on a genuine date change - i.e.
    df.iloc[idx]'s date differs from df.iloc[idx - 1]'s date. This
    guarantees every fold boundary falls strictly BETWEEN two different
    event dates, never inside one.
    """
    n = len(df)
    while idx < n and df.iloc[idx][date_col] == df.iloc[idx - 1][date_col]:
        idx += 1
    return idx


def generate_walk_forward_folds(
    df: pd.DataFrame,
    initial_train_size: int,
    test_size: int,
    step_size: int | None = None,
    date_col: str = "date",
):
    """
    Yields (train_idx, test_idx) pairs of positional indices into df,
    one pair per fold, walking forward through time.

    - initial_train_size: approximate number of (chronologically
      earliest) fights in the FIRST training window. Must be large
      enough that the model isn't training on a tiny, unrepresentative
      slice of early UFC history. ACTUAL size may be slightly larger
      than requested - see _snap_to_date_boundary above: boundaries are
      snapped forward to the next full date change so a single night's
      fight card is never split across train and test.
    - test_size: approximate number of fights in each test fold. Same
      snapping applies - actual fold sizes vary slightly card-to-card.
    - step_size: how many fights to advance before the next fold.
      Defaults to test_size (non-overlapping folds - see module
      docstring).

    Stops automatically once there isn't enough remaining data for a
    full test fold.
    """
    if step_size is None:
        step_size = test_size

    n = len(df)
    train_end = _snap_to_date_boundary(df, initial_train_size, date_col)

    while True:
        test_end = _snap_to_date_boundary(df, train_end + test_size, date_col)
        if test_end > n:
            break  # not enough fights left for a full test fold

        train_idx = list(range(0, train_end))
        test_idx = list(range(train_end, test_end))

        yield train_idx, test_idx

        train_end = _snap_to_date_boundary(df, train_end + step_size, date_col)


def validate_no_leakage(df: pd.DataFrame, train_idx, test_idx, date_col: str = "date") -> bool:
    """
    The purging check: confirms every date in the training set is
    strictly before every date in the test set for this fold. Returns
    True if clean, raises an AssertionError with the offending dates
    if not. Run this on every fold - don't just trust the index
    arithmetic, verify it against the actual dates.
    """
    train_max_date = df.iloc[train_idx][date_col].max()
    test_min_date = df.iloc[test_idx][date_col].min()

    assert train_max_date < test_min_date, (
        f"LEAKAGE: training set's latest fight ({train_max_date}) is not "
        f"strictly before test set's earliest fight ({test_min_date})"
    )
    return True


if __name__ == "__main__":
    # Manual validation example #1: no same-date fights - the simple case.
    example = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=20, freq="30D"),
        "fight_id": range(20),
    })

    folds = list(generate_walk_forward_folds(
        example, initial_train_size=8, test_size=4, date_col="date"
    ))

    print(f"[Example 1] Generated {len(folds)} folds from 20 fights, no shared dates\n")

    for i, (train_idx, test_idx) in enumerate(folds, start=1):
        train_dates = example.iloc[train_idx]["date"]
        test_dates = example.iloc[test_idx]["date"]
        validate_no_leakage(example, train_idx, test_idx, date_col="date")

        print(f"Fold {i}: train {len(train_idx)} fights "
              f"({train_dates.min().date()} to {train_dates.max().date()}) "
              f"-> test {len(test_idx)} fights "
              f"({test_dates.min().date()} to {test_dates.max().date()})")

    # Manual validation example #2: DELIBERATELY puts several fights on the
    # SAME date right where a naive boundary would fall (mimicking a real
    # UFC event card), to prove the date-snapping fix actually works.
    # Fights 6, 7, 8 all share one date, positioned so a naive
    # initial_train_size=8 boundary would have split fights 7/8 from fight 6.
    dates = list(pd.date_range("2020-01-01", periods=6, freq="30D"))
    same_day = [pd.Timestamp("2020-07-01")] * 3          # fights 6, 7, 8 - one card
    dates += same_day
    dates += list(pd.date_range("2020-08-01", periods=11, freq="15D"))  # fights 9-19

    example2 = pd.DataFrame({"date": dates, "fight_id": range(len(dates))})

    print(f"\n[Example 2] Same-date card test: fights 6-8 all dated 2020-07-01\n")
    folds2 = list(generate_walk_forward_folds(
        example2, initial_train_size=8, test_size=4, date_col="date"
    ))

    for i, (train_idx, test_idx) in enumerate(folds2, start=1):
        validate_no_leakage(example2, train_idx, test_idx, date_col="date")  # would raise if broken
        train_dates = example2.iloc[train_idx]["date"]
        test_dates = example2.iloc[test_idx]["date"]
        print(f"Fold {i}: train {len(train_idx)} fights (ends {train_dates.max().date()}) "
              f"-> test {len(test_idx)} fights (starts {test_dates.min().date()})")
        # Confirm the same-day card (fights 6,7,8) landed ENTIRELY on one
        # side of every boundary - never split.
        card_positions = [j for j, d in enumerate(example2["date"]) if d == pd.Timestamp("2020-07-01")]
        in_train = [p in train_idx for p in card_positions]
        in_test = [p in test_idx for p in card_positions]
        assert not (any(in_train) and any(in_test)), "Same-day card was split across train/test!"

    print("\nAll folds passed - the 2020-07-01 card was never split across a boundary.")