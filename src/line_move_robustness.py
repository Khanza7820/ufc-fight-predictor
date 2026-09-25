"""
Robustness checks for the frozen Week 10 strategy (Pinnacle value, thr 0.02).
Descriptive only - the strategy and its bets are unchanged.

1. De-vig bias: proportional de-vig overstates underdog probability, and the
   bets are mostly underdogs. Re-score CLV against a POWER-de-vigged consensus
   close and against Pinnacle's own power-de-vigged close.
2. Concentration: CLV without the top 10% of bets; price vs. third-best price.
3. Realised results: dev bets matched to ufc-master.csv outcomes (to 2026-03).
Usage: python line_move_robustness.py <UFC_betting_odds.csv>
"""
import sys
import numpy as np, pandas as pd
from scipy.optimize import brentq
from line_move import pinnacle_any_snapshot, summarise, DEV_END
from market_edge_clv import load_snapshots
from market_edge_replication import norm
from market_edge import load


def power_p1(o1, o2):
    a, b = 1 / o1, 1 / o2
    f = lambda k: a ** k + b ** k - 1
    return a ** brentq(f, 0.05, 20) if f(0.05) * f(20) < 0 else a / (a + b)


def close_lines(path):
    k = load_snapshots(path)
    k = k[k.ts < k.event.dt.tz_localize("UTC")]
    out = {}
    for (ev, n1, n2), g in k.groupby(["event", "n1", "n2"]):
        c = g[g.ts == g.ts.max()]
        pw = np.array([power_p1(x, y) for x, y in zip(c.odds_1, c.odds_2)])
        pin = c.source.values == "Pinnacle"
        out[f"{ev.date()}|{n1}|{n2}"] = (np.median(pw), np.median(pw[pin]) if pin.any() else np.nan)
    return out


def main(path):
    b = pinnacle_any_snapshot(path, 0.02)
    cl = close_lines(path)
    pw = np.array([cl[f][0] for f in b.fight]); pin = np.array([cl[f][1] for f in b.fight])
    b["close_pow"] = np.where(b.side == 1, pw, 1 - pw)
    b["close_pin"] = np.where(b.side == 1, pin, 1 - pin)
    for name, part in [("DEV", b[b.event < DEV_END]), ("HOLDOUT", b[b.event >= DEV_END])]:
        print(f"\n== {name}")
        summarise(part, "CLV, proportional consensus")
        summarise(part.assign(clv=part.price * part.close_pow - 1), "CLV, power consensus")
        pp = part.dropna(subset=["close_pin"])
        summarise(pp.assign(clv=pp.price * pp.close_pin - 1), "CLV, Pinnacle power close")
        cut = part.clv.quantile(0.9)
        summarise(part[part.clv <= cut], "CLV excl. top 10% of bets")
        print(f"  largest single-bet CLVs: {np.round(np.sort(part.clv.values)[-5:], 3)}")

    # realised outcomes for dev bets, matched to ufc-master results
    m = load()
    res = {}
    for r in m[m.date >= pd.Timestamp("2025-07-01")].itertuples(index=False):
        res[(r.date, frozenset([norm(r.R_fighter), norm(r.B_fighter)]))] = norm(r.R_fighter if r.y == 1 else r.B_fighter)
    won = []
    for r in b[b.event < DEV_END].itertuples(index=False):
        _, n1, n2 = r.fight.split("|")
        w = None
        for dd in (0, -1, 1):
            w = res.get((r.event + pd.Timedelta(days=dd), frozenset([n1, n2])))
            if w:
                break
        won.append(np.nan if w is None else float(w == (n1 if r.side == 1 else n2)))
    d = b[b.event < DEV_END].assign(won=won).dropna(subset=["won"])
    pnl = np.where(d.won == 1, d.price - 1, -1.0)
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(pnl, len(pnl)).mean() for _ in range(5000)])
    print(f"\n== DEV realised results (matched {len(d)} of {(b.event < DEV_END).sum()} bets)")
    print(f"  ROI {pnl.mean():+.2%} CI [{np.percentile(bs, 2.5):+.2%}, {np.percentile(bs, 97.5):+.2%}] "
          f"units {pnl.sum():+.1f}; win rate {d.won.mean():.1%} vs close-implied {d.close_p.mean():.1%}")


if __name__ == "__main__":
    main(sys.argv[1])
