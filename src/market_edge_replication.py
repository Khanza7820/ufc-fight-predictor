"""
Independent-price replication for the frozen market-edge strategy.

The holdout picks were made and priced with ufc-master.csv odds. Here the SAME
picks (no re-selection) are re-priced at a second, independent odds source:
betmma.tips decimal odds from jansen88/ufc-data (complete_ufc_data.csv).
If the result depended on quirks of one odds feed, it should not survive.

Usage: python market_edge_replication.py <path to complete_ufc_data.csv>
"""
import sys
import unicodedata
from market_edge import *
from market_edge_report import holdout_predictions, bet_table, boot_ci


def norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return "".join(ch for ch in s if ch.isalpha())


def main(path):
    ext = pd.read_csv(path)
    ext["date"] = pd.to_datetime(ext.event_date)
    ext = ext.dropna(subset=["favourite_odds", "underdog_odds"])
    # the feed contains a few corrupt prices (e.g. inf); keep only plausible decimal odds
    ext = ext[ext.favourite_odds.between(1.01, 50) & ext.underdog_odds.between(1.01, 50)]
    price = {}
    for r in ext.itertuples(index=False):
        key = (r.date, frozenset([norm(r.fighter1), norm(r.fighter2)]))
        price[key] = {norm(r.favourite): float(r.favourite_odds), norm(r.underdog): float(r.underdog_odds)}

    o = holdout_predictions(load())
    b = bet_table(o)
    # rebuild fighter/side info for each bet, using the same selection rule as bet_table
    ev_r, ev_b = o.p * o.R_dec - 1, (1 - o.p) * o.B_dec - 1
    red = ev_r >= ev_b
    sel = np.where(red, ev_r, ev_b) > 0.03
    picks = o[sel].assign(side_red=red[sel])
    picks["pick"] = np.where(picks.side_red, picks.R_fighter, picks.B_fighter)
    picks["won"] = np.where(picks.side_red, picks.y == 1, picks.y == 0)
    picks["odds_a"] = np.where(picks.side_red, picks.R_dec, picks.B_dec)
    assert len(picks) == len(b)

    odds_b = []
    for r in picks.itertuples(index=False):
        pair = frozenset([norm(r.R_fighter), norm(r.B_fighter)])
        hit = None
        for dd in (0, -1, 1):  # tolerate time-zone date offsets
            hit = price.get((r.date + pd.Timedelta(days=dd), pair))
            if hit:
                break
        odds_b.append(hit.get(norm(r.pick), np.nan) if hit else np.nan)
    picks["odds_b"] = odds_b
    m = picks.dropna(subset=["odds_b"])
    covered = picks[picks.date <= ext.date.max()]
    print(f"Holdout bets inside the second source's date range: {len(covered)}; matched: {len(m)} "
          f"({len(m) / len(covered):.0%})")

    for label, col in [("original feed (ufc-master)", "odds_a"), ("independent feed (betmma.tips)", "odds_b")]:
        pnl = np.where(m.won, m[col] - 1, -1.0)
        lo, hi, p = boot_ci(pnl)
        print(f"  {label:32s} ROI {pnl.mean():+.2%}  CI [{lo:+.2%}, {hi:+.2%}]  p(ROI<=0)={p:.3f}  units {pnl.sum():+.1f}")
    best = np.maximum(m.odds_a, m.odds_b)
    pnl = np.where(m.won, best - 1, -1.0)
    print(f"  {'best of the two (line shopping)':32s} ROI {pnl.mean():+.2%}  units {pnl.sum():+.1f}")
    diff = np.log(m.odds_b / m.odds_a)
    print(f"\nPrice agreement: median |difference| {np.median(np.abs(m.odds_b / m.odds_a - 1)):.1%}; "
          f"independent feed better on {(diff > 0).mean():.0%} of picks, worse on {(diff < 0).mean():.0%}")
    print(f"Sanity: side-by-side prices of 5 matched picks:\n"
          f"{m[['date', 'pick', 'odds_a', 'odds_b', 'won']].head().to_string(index=False)}")


if __name__ == "__main__":
    main(sys.argv[1])
