"""
Robustness report for the pre-registered market-edge strategy on the locked
holdout. Nothing here selects or tunes anything - the strategy (lr, C=0.05,
EV > 0.03, flat 1-unit stakes) was frozen before the holdout was scored. These
are descriptive stress tests of that single, already-reported result.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from market_edge import *

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def holdout_predictions(d):
    out = []
    for yr in range(2021, 2027):
        tr = d[d.date < max(pd.Timestamp(f"{yr}-01-01"), HOLDOUT_START)]
        te = d[(d.date.dt.year == yr) & (d.date >= HOLDOUT_START)].copy()
        te["p"] = fit_predict("lr", tr, te)
        out.append(te)
    return pd.concat(out)


def bet_table(o, ev_min=0.03):
    ev_r, ev_b = o.p * o.R_dec - 1, (1 - o.p) * o.B_dec - 1
    red = ev_r >= ev_b
    t = pd.DataFrame({
        "date": o.date.values,
        "ev": np.where(red, ev_r, ev_b),
        "dec_odds": np.where(red, o.R_dec, o.B_dec),
        "won": np.where(red, o.y == 1, o.y == 0),
        "mkt_p": np.where(red, o.p_mkt, 1 - o.p_mkt),
    })
    return t[t.ev > ev_min].reset_index(drop=True)


def boot_ci(pnl, seed=0, n=5000):
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(pnl, len(pnl)).mean() for _ in range(n)])
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5), (bs <= 0).mean()


def main():
    o = holdout_predictions(load())
    b = bet_table(o)
    b["pnl"] = np.where(b.won, b.dec_odds - 1, -1.0)
    cum = b.pnl.cumsum()
    dd = (cum - cum.cummax()).min()
    streak = max_run = 0
    for w in b.won:
        streak = 0 if w else streak + 1
        max_run = max(max_run, streak)

    lo, hi, p = boot_ci(b.pnl.values)
    print(f"Holdout bets: {len(b)} of {len(o)} fights | ROI {b.pnl.mean():+.2%} "
          f"CI [{lo:+.2%}, {hi:+.2%}] p(ROI<=0)={p:.3f} | units {b.pnl.sum():+.1f}")
    print(f"Max drawdown: {dd:.1f} units | longest losing run: {max_run} bets | "
          f"win rate {b.won.mean():.1%} | mean decimal odds {b.dec_odds.mean():.2f}")

    print("\nFavourite / underdog split:")
    for name, g in b.groupby(b.mkt_p >= 0.5):
        print(f"  {'favourites' if name else 'underdogs ':10s} n={len(g):4d} ROI {g.pnl.mean():+.2%}")

    print("\nPrice haircut (bets and picks unchanged; payout at worse odds):")
    haircuts = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    rois = []
    for h in haircuts:
        worse = 1 + (b.dec_odds - 1) * (1 - h)
        pnl = np.where(b.won, worse - 1, -1.0)
        rois.append(pnl.mean())
        print(f"  {h:.0%} worse payout: ROI {pnl.mean():+.2%}")
    for h1, h2, r1, r2 in zip(haircuts, haircuts[1:], rois, rois[1:]):
        if r1 > 0 >= r2:
            print(f"  -> break-even at ~{h1 + (h2 - h1) * r1 / (r1 - r2):.1%} worse payouts")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(pd.to_datetime(b.date), cum.values, color="#1f5fa8", lw=1.6)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_title("Market-anchored model - locked holdout (Apr 2021 - Mar 2026), flat 1-unit stakes")
    ax.set_ylabel("Cumulative profit (units)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORTS / "week9_holdout_equity.png", dpi=150)
    print(f"\nSaved {REPORTS / 'week9_holdout_equity.png'}")


if __name__ == "__main__":
    main()
