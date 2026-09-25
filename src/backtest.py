"""
Backtest engine for the UFC fight predictor.

CRITICAL DESIGN DECISION: predictions used here come from the walk-
forward CV folds (xgboost_model.py), NOT the "final" model trained on
all data (used for SHAP in shap_analysis.py). Each fold's test-set
predictions come from a model that never saw those specific fights
during training - genuine out-of-sample predictions, exactly matching
what a real deployed system would have produced at the time. Using the
final model here would be lookahead at the backtest stage: that model
was trained on every fight including the ones being "bet on."

Betting logic (Day 3-4 baseline - flat staking; Kelly sizing replaces
this in kelly.py):
1. For each fight in the test period, in chronological order:
2. Compute edge on both sides: model probability minus bookmaker
   implied probability, for Red and for Blue.
3. If either side's edge exceeds edge_threshold, bet on the side with
   the larger edge (only one side per fight - betting both sides of
   the same fight is not a real edge, it's just paying the vig twice).
4. If no side clears the threshold, no bet is placed - track bankroll
   unchanged for that fight.
5. Stake is currently a FLAT fraction of current bankroll (placeholder
   for Kelly sizing) - this lets the engine's core betting/outcome
   logic be validated independently of the position-sizing formula.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def american_to_decimal_odds(odds: float) -> float:
    """
    Converts American odds to decimal odds (the format Kelly's formula
    uses: b = decimal_odds - 1, i.e. the net profit multiple on a
    winning $1 stake).
    """
    if odds > 0:
        return odds / 100 + 1
    else:
        return 100 / (-odds) + 1


def generate_oos_predictions(
    df: pd.DataFrame,
    folds: list,
    feature_cols: list[str],
    xgb_params: dict,
    id_cols: tuple = ("R_fighter", "B_fighter", "date"),
) -> pd.DataFrame:
    """
    Trains a fresh XGBoost model on each fold's training data, predicts
    on that fold's test data, and collects the results into one
    DataFrame covering every fold's test period. Because folds are
    non-overlapping and walk forward in time, this naturally produces
    one genuine out-of-sample prediction per fight across the whole
    walk-forward period - no fight is predicted by a model that had
    already seen it.

    Returns a DataFrame with id_cols + model_prob (P(Red wins)) +
    target (actual outcome), one row per fight in the test folds.
    Fights in the INITIAL training window (before the first fold's
    test period begins) are NOT included - there's no out-of-sample
    prediction for them, since they were only ever used for training.
    """
    rows = []
    for train_idx, test_idx in folds:
        X_train = df.iloc[train_idx][feature_cols]
        y_train = df.iloc[train_idx]["target"]
        X_test = df.iloc[test_idx][feature_cols]

        model = XGBClassifier(**xgb_params, eval_metric="logloss")
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        fold_result = df.iloc[test_idx][list(id_cols) + ["target"]].copy()
        fold_result["model_prob"] = probs
        rows.append(fold_result)

    oos_df = pd.concat(rows, ignore_index=True)
    print(f"Generated {len(oos_df)} out-of-sample predictions across {len(folds)} folds.")
    return oos_df


def run_backtest(
    oos_with_odds: pd.DataFrame,
    edge_threshold: float = 0.05,
    stake_fraction: float = 0.01,
    initial_bankroll: float = 1000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the event-driven backtest, one fight at a time, in
    chronological order (oos_with_odds MUST already be sorted by date
    - this function trusts that ordering and does not re-sort).

    oos_with_odds must contain: model_prob, target, R_implied_prob,
    B_implied_prob, R_odds, B_odds, date (and id columns for logging).

    Returns:
    - bankroll_history: one row per fight, with running bankroll
      (including no-bet fights, so the full timeline is visible)
    - bet_log: one row per fight WHERE A BET WAS ACTUALLY PLACED, with
      full detail (edge, side, stake, odds, outcome, profit) - this is
      what you manually spot-check against a few fights by hand.
    """
    bankroll = initial_bankroll
    history_rows = []
    bet_rows = []

    for _, fight in oos_with_odds.iterrows():
        edge_red = fight["model_prob"] - fight["R_implied_prob"]
        edge_blue = (1 - fight["model_prob"]) - fight["B_implied_prob"]

        side = None
        edge = 0.0
        if edge_red > edge_threshold and edge_red >= edge_blue:
            side = "Red"
            edge = edge_red
        elif edge_blue > edge_threshold and edge_blue > edge_red:
            side = "Blue"
            edge = edge_blue

        if side is not None:
            odds = fight["R_odds"] if side == "Red" else fight["B_odds"]
            decimal_odds = american_to_decimal_odds(odds)
            stake = bankroll * stake_fraction

            won = (side == "Red" and fight["target"] == 1) or (side == "Blue" and fight["target"] == 0)
            profit = stake * (decimal_odds - 1) if won else -stake
            bankroll += profit

            bet_rows.append({
                "date": fight["date"],
                "R_fighter": fight.get("R_fighter"),
                "B_fighter": fight.get("B_fighter"),
                "side_bet": side,
                "model_prob": fight["model_prob"],
                "implied_prob": fight["R_implied_prob"] if side == "Red" else fight["B_implied_prob"],
                "edge": edge,
                "odds_american": odds,
                "decimal_odds": decimal_odds,
                "stake": stake,
                "won": won,
                "profit": profit,
                "bankroll_after": bankroll,
            })

        history_rows.append({"date": fight["date"], "bankroll": bankroll, "bet_placed": side is not None})

    bankroll_history = pd.DataFrame(history_rows)
    bet_log = pd.DataFrame(bet_rows)

    print(f"Total fights evaluated: {len(oos_with_odds)}")
    print(f"Bets placed: {len(bet_log)} ({len(bet_log)/len(oos_with_odds):.1%} of fights)")
    print(f"Starting bankroll: {initial_bankroll:.2f}")
    print(f"Final bankroll: {bankroll:.2f}")
    print(f"Total return: {(bankroll/initial_bankroll - 1):+.1%}")
    if len(bet_log) > 0:
        print(f"Win rate on placed bets: {bet_log['won'].mean():.1%}")

    return bankroll_history, bet_log


def compute_edge_decile_analysis(oos_with_odds: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    """
    DIAGNOSTIC, run BEFORE trusting any backtest bankroll result: for
    every fight (regardless of any threshold), identifies whichever
    side (Red/Blue) the model favours MORE than the market, records
    that side's edge magnitude, whether it actually won, and the flat-
    $1-stake ROI. Fights are then bucketed into deciles by edge
    magnitude.

    THE KEY QUESTION THIS ANSWERS: does a LARGER self-reported edge
    actually correspond to BETTER real-world performance (higher win
    rate / ROI in higher deciles)? If yes, the model's edge estimate is
    trustworthy and worth acting on. If flat or inverted, large
    "edges" are dominated by per-fight prediction noise/overconfidence
    rather than genuine insight - in which case a naive edge threshold
    is picking the model's biggest mistakes, not its best calls.
    """
    df = oos_with_odds.copy()

    side_bet = np.where(
        (df["model_prob"] - df["R_implied_prob"]) >= ((1 - df["model_prob"]) - df["B_implied_prob"]),
        "Red", "Blue"
    )
    edge = np.where(
        side_bet == "Red",
        df["model_prob"] - df["R_implied_prob"],
        (1 - df["model_prob"]) - df["B_implied_prob"],
    )
    odds = np.where(side_bet == "Red", df["R_odds"], df["B_odds"])
    won = np.where(side_bet == "Red", df["target"] == 1, df["target"] == 0)
    decimal_odds = np.array([american_to_decimal_odds(o) for o in odds])
    roi_per_dollar = np.where(won, decimal_odds - 1, -1.0)

    analysis = pd.DataFrame({
        "edge": edge, "side_bet": side_bet, "won": won, "roi_per_dollar": roi_per_dollar,
    })
    analysis["edge_decile"] = pd.qcut(analysis["edge"], n_deciles, labels=False, duplicates="drop")

    summary = analysis.groupby("edge_decile").agg(
        n_fights=("edge", "count"),
        min_edge=("edge", "min"),
        max_edge=("edge", "max"),
        win_rate=("won", "mean"),
        avg_roi_per_dollar=("roi_per_dollar", "mean"),
    ).round(4)

    print(summary.to_string())
    print("\nIf the model's edge estimate is trustworthy, win_rate and avg_roi_per_dollar should")
    print("generally INCREASE from decile 0 to decile 9. Flat or inverted = edge is not reliable")
    print("as a bet-selection signal at face value.")
    return summary


def compute_stratified_edge_decile_analysis(oos_with_odds: pd.DataFrame, n_deciles: int = 5) -> pd.DataFrame:
    """
    Refines compute_edge_decile_analysis by first splitting fights into
    FAVOURITES (chosen side's market-implied probability > 0.5) and
    UNDERDOGS, then computing edge deciles SEPARATELY within each
    group, using ROI (not win rate) as the primary quality metric.

    WHY THIS MATTERS: underdog bets mechanically produce larger raw
    edge NUMBERS for the same amount of genuine model confidence
    (longer odds amplify probability disagreement), and underdogs
    genuinely win less often regardless of model quality (that's what
    "underdog" means). Sorting all fights together by raw edge
    magnitude conflates "how much of an underdog is this" with "how
    confident/correct is the model" - stratifying removes that
    confound, so a genuinely reliable edge signal should show
    increasing ROI by decile WITHIN each group, not just overall.
    """
    df = oos_with_odds.copy()

    side_bet = np.where(
        (df["model_prob"] - df["R_implied_prob"]) >= ((1 - df["model_prob"]) - df["B_implied_prob"]),
        "Red", "Blue"
    )
    edge = np.where(
        side_bet == "Red",
        df["model_prob"] - df["R_implied_prob"],
        (1 - df["model_prob"]) - df["B_implied_prob"],
    )
    chosen_implied_prob = np.where(side_bet == "Red", df["R_implied_prob"], df["B_implied_prob"])
    odds = np.where(side_bet == "Red", df["R_odds"], df["B_odds"])
    won = np.where(side_bet == "Red", df["target"] == 1, df["target"] == 0)
    decimal_odds = np.array([american_to_decimal_odds(o) for o in odds])
    roi_per_dollar = np.where(won, decimal_odds - 1, -1.0)

    analysis = pd.DataFrame({
        "edge": edge, "won": won, "roi_per_dollar": roi_per_dollar,
        "favourite": chosen_implied_prob > 0.5,
    })

    results = {}
    for group_name, group_df in analysis.groupby("favourite"):
        label = "Favourites" if group_name else "Underdogs"
        group_df = group_df.copy()
        group_df["edge_decile"] = pd.qcut(group_df["edge"], n_deciles, labels=False, duplicates="drop")
        summary = group_df.groupby("edge_decile").agg(
            n_fights=("edge", "count"),
            min_edge=("edge", "min"),
            max_edge=("edge", "max"),
            win_rate=("won", "mean"),
            avg_roi_per_dollar=("roi_per_dollar", "mean"),
        ).round(4)
        print(f"\n{'='*60}\n{label} (n={len(group_df)})\n{'='*60}")
        print(summary.to_string())
        results[label] = summary

    return results


if __name__ == "__main__":
    # Manual validation: hand-construct a few fights with known odds and
    # outcomes, confirm the engine's bankroll math matches by-hand
    # calculation exactly.
    test_fights = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15", "2020-01-22"]),
        "R_fighter": ["A", "C", "E", "G"],
        "B_fighter": ["B", "D", "F", "H"],
        "target": [1, 0, 1, 0],  # Red, Blue, Red, Blue win
        "model_prob": [0.70, 0.30, 0.54, 0.20],  # P(Red wins)
        "R_implied_prob": [0.55, 0.55, 0.50, 0.60],
        "B_implied_prob": [0.50, 0.50, 0.54, 0.44],
        "R_odds": [-120, -120, -100, 150],
        "B_odds": [100, 100, 117, -140],
    })

    print("=== Fight-by-fight manual check ===")
    print(test_fights[["date", "model_prob", "R_implied_prob", "B_implied_prob"]].to_string())
    # Fight 1: edge_red = 0.70-0.55=0.15 (bet Red, threshold 0.05, odds -120 -> decimal 1.833)
    # Fight 2: edge_red = 0.30-0.55=-0.25, edge_blue = 0.70-0.50=0.20 (bet Blue, odds 100 -> decimal 2.0)
    # Fight 3: edge_red = 0.54-0.50=0.04, edge_blue = 0.46-0.54=-0.08 -> NO BET (clearly below threshold 0.05)
    # Fight 4: edge_red = 0.20-0.60=-0.40, edge_blue=0.80-0.44=0.36 (bet Blue, odds -140 -> decimal 1.714)

    bankroll_history, bet_log = run_backtest(
        test_fights, edge_threshold=0.05, stake_fraction=0.10, initial_bankroll=1000.0
    )
    print("\n=== Bet log ===")
    print(bet_log[["date", "side_bet", "edge", "decimal_odds", "stake", "won", "profit", "bankroll_after"]].to_string())

    # Hand-computed expected trajectory (stake_fraction=0.10):
    # Fight 1: bet Red, stake=100, WON, decimal_odds=1.8333, profit=100*0.8333=83.33 -> bankroll=1083.33
    # Fight 2: bet Blue, stake=108.33, WON, decimal_odds=2.0, profit=108.33*1.0=108.33 -> bankroll=1191.67
    # Fight 3: NO BET (edge exactly at threshold) -> bankroll unchanged=1191.67
    # Fight 4: bet Blue, stake=119.17, WON, decimal_odds=1.7143, profit=119.17*0.7143=85.12 -> bankroll=1276.78
    assert len(bet_log) == 3, f"Expected 3 bets (fight 3 should be skipped), got {len(bet_log)}"
    assert abs(bankroll_history['bankroll'].iloc[-1] - 1276.78) < 1.0, \
        f"Final bankroll {bankroll_history['bankroll'].iloc[-1]:.2f} doesn't match hand-calculated ~1276.78"
    print("\nAll manual validation checks PASSED.")