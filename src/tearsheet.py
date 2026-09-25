"""
Tearsheet metrics for the UFC fight predictor backtest.

IMPORTANT: these metrics are computed on bankroll_history_stable /
bet_log_stable - the Week 5 result AFTER excluding the earliest,
smallest-training-window walk-forward folds (2013-12-28 to 2014-10-25),
which were found to be driving a spurious positive result. This is the
honest, decomposition-validated result, not the original +4.7% headline
number.

Event-driven annualisation: standard Sharpe ratio annualisation
(sqrt(252) for daily returns) does NOT apply here - fights are
irregularly spaced, multiple fights share event-card dates, and event
frequency varies across UFC history. Instead, we annualise using the
empirical average number of BETTING EVENTS per year in the sample:
sharpe_annualised = sharpe_per_bet * sqrt(bets_per_year). This is the
correct generalisation of the daily-returns formula to irregular event
timing - it answers "how many independent bet opportunities occur in
a year" rather than assuming a fixed calendar cadence.
"""

import numpy as np
import pandas as pd


def compute_per_fight_returns(bankroll_history: pd.DataFrame) -> pd.Series:
    """
    Per-fight fractional returns on bankroll, including no-bet fights
    (return = 0). This is the correct basis for Sharpe/drawdown - using
    only bet-fights would understate variance by ignoring the "flat"
    periods between bets, and would make the equity curve non-continuous.
    """
    bankroll = bankroll_history["bankroll"].values
    prev_bankroll = np.concatenate([[bankroll_history["bankroll"].iloc[0] /
                                      (1 + 0)], bankroll[:-1]])
    # first row: return relative to initial_bankroll is captured naturally
    # since bankroll_history's first row already reflects fight 1's outcome;
    # we instead prepend the true starting bankroll for a clean first return.
    returns = pd.Series(bankroll, index=bankroll_history.index).pct_change()
    returns.iloc[0] = 0.0  # no return before the first fight
    return returns


def compute_sharpe_ratio(
    bankroll_history: pd.DataFrame,
    bet_log: pd.DataFrame,
) -> dict:
    """
    Annualised Sharpe ratio using event-driven annualisation (see module
    docstring). Also returns the raw per-bet Sharpe and the annualisation
    factor used, so the calculation is fully auditable/explainable.
    """
    returns = compute_per_fight_returns(bankroll_history)

    mean_return = returns.mean()
    std_return = returns.std()
    sharpe_per_fight = mean_return / std_return if std_return > 0 else np.nan

    date_range_years = (
        bankroll_history["date"].max() - bankroll_history["date"].min()
    ).days / 365.25
    n_bets = len(bet_log)
    bets_per_year = n_bets / date_range_years if date_range_years > 0 else np.nan

    sharpe_annualised = sharpe_per_fight * np.sqrt(bets_per_year)

    return {
        "sharpe_per_fight": sharpe_per_fight,
        "bets_per_year": bets_per_year,
        "sharpe_annualised": sharpe_annualised,
        "mean_return_per_fight": mean_return,
        "std_return_per_fight": std_return,
    }


def compute_max_drawdown(bankroll_history: pd.DataFrame) -> dict:
    """
    Maximum peak-to-trough decline in the bankroll curve, as a fraction.
    Also returns the dates of the peak and trough for interpretation.
    """
    bankroll = bankroll_history["bankroll"]
    running_max = bankroll.cummax()
    drawdown = (bankroll - running_max) / running_max

    trough_idx = drawdown.idxmin()
    max_dd = drawdown.loc[trough_idx]
    trough_date = bankroll_history.loc[trough_idx, "date"]

    # peak is the running max at the trough, i.e. the most recent high
    # before the trough occurred
    peak_value = running_max.loc[trough_idx]
    peak_idx = bankroll[bankroll == peak_value].index
    peak_date = bankroll_history.loc[peak_idx[0], "date"] if len(peak_idx) else None

    return {
        "max_drawdown": max_dd,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "drawdown_series": drawdown,
    }


def compute_calmar_ratio(
    bankroll_history: pd.DataFrame,
    initial_bankroll: float,
) -> dict:
    """
    Calmar ratio = annualised return / abs(max drawdown). Measures
    return earned per unit of worst-case pain endured - complements
    Sharpe (which penalises all volatility equally) by focusing
    specifically on the deepest loss an investor would have lived
    through.
    """
    final_bankroll = bankroll_history["bankroll"].iloc[-1]
    total_return = final_bankroll / initial_bankroll - 1
    date_range_years = (
        bankroll_history["date"].max() - bankroll_history["date"].min()
    ).days / 365.25
    annualised_return = (1 + total_return) ** (1 / date_range_years) - 1 if date_range_years > 0 else np.nan

    dd_result = compute_max_drawdown(bankroll_history)
    max_dd = dd_result["max_drawdown"]

    calmar = annualised_return / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "annualised_return": annualised_return,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
    }


def bootstrap_sharpe_ci(
    bankroll_history: pd.DataFrame,
    bet_log: pd.DataFrame,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict:
    """
    Bootstrap confidence interval on the annualised Sharpe ratio.
    Resamples per-fight returns WITH REPLACEMENT (not blocks - this
    ignores any autocorrelation in returns, a simplifying assumption
    worth stating explicitly: it treats each fight's outcome as
    independent, which is reasonable here since consecutive fights in
    the walk-forward test period are unrelated bouts, not a continuous
    trading signal with momentum).
    """
    returns = compute_per_fight_returns(bankroll_history).values
    rng = np.random.default_rng(random_state)

    date_range_years = (
        bankroll_history["date"].max() - bankroll_history["date"].min()
    ).days / 365.25
    n_bets = len(bet_log)
    bets_per_year = n_bets / date_range_years if date_range_years > 0 else np.nan
    annualisation_factor = np.sqrt(bets_per_year)

    boot_sharpes = []
    n = len(returns)
    for _ in range(n_bootstrap):
        sample = rng.choice(returns, size=n, replace=True)
        std = sample.std()
        sharpe = (sample.mean() / std * annualisation_factor) if std > 0 else np.nan
        boot_sharpes.append(sharpe)

    boot_sharpes = np.array(boot_sharpes)
    lower_pct = (1 - ci) / 2 * 100
    upper_pct = (1 + ci) / 2 * 100

    return {
        "point_estimate": compute_sharpe_ratio(bankroll_history, bet_log)["sharpe_annualised"],
        "ci_lower": np.nanpercentile(boot_sharpes, lower_pct),
        "ci_upper": np.nanpercentile(boot_sharpes, upper_pct),
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
        "boot_sharpes": boot_sharpes,
    }


def simulate_random_betting(
    oos_with_odds: pd.DataFrame,
    n_bets: int,
    stake_fraction: float = 0.01,
    initial_bankroll: float = 1000.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Benchmark 1: randomly select n_bets fights from oos_with_odds (no
    edge logic at all), pick a random side, flat-stake stake_fraction.
    Simulates "what if you just bet at random with no model" - the
    floor any real strategy needs to beat.
    """
    rng = np.random.default_rng(random_state)
    from src.backtest import american_to_decimal_odds

    sample = oos_with_odds.sample(n=n_bets, random_state=random_state).sort_values("date").reset_index(drop=True)
    sides = rng.choice(["Red", "Blue"], size=len(sample))

    bankroll = initial_bankroll
    history = []
    for i, fight in sample.iterrows():
        side = sides[i]
        odds = fight["R_odds"] if side == "Red" else fight["B_odds"]
        decimal_odds = american_to_decimal_odds(odds)
        won = (side == "Red" and fight["target"] == 1) or (side == "Blue" and fight["target"] == 0)
        stake = bankroll * stake_fraction
        profit = stake * (decimal_odds - 1) if won else -stake
        bankroll += profit
        history.append({"date": fight["date"], "bankroll": bankroll})

    return pd.DataFrame(history)


def simulate_always_favourite(
    oos_with_odds: pd.DataFrame,
    n_bets: int,
    stake_fraction: float = 0.01,
    initial_bankroll: float = 1000.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Benchmark 2: always bet the market favourite (lower American odds
    magnitude on the negative side / higher implied probability),
    flat-staked, on the same NUMBER of fights as the real strategy
    (sampled from the full oos_with_odds set, not underdog-filtered,
    since "always bet favourite" is a genuinely different strategy that
    should be judged on its own natural fight set).
    """
    from src.backtest import american_to_decimal_odds

    sample = oos_with_odds.sample(n=n_bets, random_state=random_state).sort_values("date").reset_index(drop=True)

    bankroll = initial_bankroll
    history = []
    for _, fight in sample.iterrows():
        side = "Red" if fight["R_implied_prob"] >= fight["B_implied_prob"] else "Blue"
        odds = fight["R_odds"] if side == "Red" else fight["B_odds"]
        decimal_odds = american_to_decimal_odds(odds)
        won = (side == "Red" and fight["target"] == 1) or (side == "Blue" and fight["target"] == 0)
        stake = bankroll * stake_fraction
        profit = stake * (decimal_odds - 1) if won else -stake
        bankroll += profit
        history.append({"date": fight["date"], "bankroll": bankroll})

    return pd.DataFrame(history)