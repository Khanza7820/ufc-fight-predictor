"""
PRE-REGISTERED HOLDOUT TEST â€” written before any holdout row was scored.

Holdout: 2021-04-01 .. 2026-03-28 (fights with odds, Red/Blue winner only).
Protocol: walk-forward, retrain each calendar year on ALL fights before that year
(2021 test window starts 2021-04-01). Flat 1-unit stakes. One bet max per fight.

Candidates (frozen from dev):
  A. "lr"        C=0.05, bet side with EV > 0.03   (best dev ROI cell: +3.3%, CI [-1.8%, +8.4%])
  B. "mkt_recal" bet side with EV > 0.03           (favourite-longshot recalibration; theory-backed)

Pass criterion: holdout ROI > 0 with bootstrap 95% CI lower bound > 0.
(With 2 candidates, a pass on either one alone at 95% would be ~2.5% per test after Holm â€” report both.)
"""
from market_edge import *

d = load()
res = {}
for kind in ["lr", "mkt_recal"]:
    out = []
    for yr in range(2021, 2027):
        tr = d[d.date < max(pd.Timestamp(f"{yr}-01-01"), HOLDOUT_START)]
        te = d[(d.date.dt.year == yr) & (d.date >= HOLDOUT_START)].copy()
        te["p"] = fit_predict(kind, tr, te)
        out.append(te)
    o = pd.concat(out)
    pnl, dates = bets(o, ev_min=0.03)
    res[kind] = (o, pnl, dates)
    print(f"\n== HOLDOUT {kind}: logloss {log_loss(o.y, o.p):.4f} vs market {log_loss(o.y, o.p_mkt):.4f}")
    print("  ", summary(pnl), f"units={pnl.sum():+.1f}")
    yrs = pd.DatetimeIndex(dates).year
    print("   by year:", {int(y): round(float(pnl[yrs == y].mean()), 3) for y in sorted(set(yrs))})
    rng = np.random.default_rng(1)
    bs = np.array([rng.choice(pnl, len(pnl)).mean() for _ in range(5000)])
    print(f"   one-sided bootstrap p(ROI<=0) = {(bs <= 0).mean():.3f}")

