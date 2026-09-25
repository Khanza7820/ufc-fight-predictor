"""
PRE-REGISTERED HOLDOUT TEST v2 (Week 10) - written before this version was run.

Why v2 exists: v1 (line_move_holdout.py) PASSED, but its robustness check
(line_move_robustness.py) showed the pass was produced by proportional
de-vigging, which overstates underdog probability; the strategy mostly backs
underdogs. v2 applies POWER de-vig to both Pinnacle's fair line (selection)
and the consensus close (scoring). Threshold is unchanged at 0.02.

Dev (events < 2026-04-01): n=120, CLV +2.26%, 95% CI [+0.51%, +3.82%].
Disclosure: this is the SECOND look at the Apr-Sep 2026 holdout for this
strategy family. The change was dictated by a known metric bias, not tuned
on holdout performance (the v1 bets re-scored with power de-vig looked worse).

Pass: mean CLV > 0 AND card-bootstrap 95% CI lower bound > 0.
Secondary: CLV against Pinnacle's own power-de-vigged close.
Usage: python line_move_holdout_v2.py <UFC_betting_odds.csv>
"""
import sys
import numpy as np
import line_move as lm
from line_move_robustness import power_p1
from market_edge_clv import load_snapshots, card_boot

THR = 0.02


def power_snapshots(path):
    k = load_snapshots(path)
    k["p1"] = [power_p1(a, b) for a, b in zip(k.odds_1, k.odds_2)]
    return k


if __name__ == "__main__":
    k = power_snapshots(sys.argv[1])
    lm.load_snapshots = lambda _: k
    b = lm.pinnacle_any_snapshot(sys.argv[1], THR)
    kc = k[k.ts < k.event.dt.tz_localize("UTC")]
    pin_close = {}
    for (ev, n1, n2), g in kc[kc.source == "Pinnacle"].groupby(["event", "n1", "n2"]):
        pin_close[f"{ev.date()}|{n1}|{n2}"] = g[g.ts == g.ts.max()].p1.median()
    for name, part in [("DEV", b[b.event < lm.DEV_END]), ("HOLDOUT", b[b.event >= lm.DEV_END])]:
        print(f"== {name}: {part.event.nunique()} cards")
        lm.summarise(part, "CLV vs power consensus close")
        pc = np.array([pin_close.get(f, np.nan) for f in part.fight])
        pc = np.where(part.side == 1, pc, 1 - pc)
        pp = part.assign(clv=part.price * pc - 1).dropna(subset=["clv"])
        lm.summarise(pp, "CLV vs Pinnacle power close")
    h = b[b.event >= lm.DEV_END]
    lo, _, _ = card_boot(h.clv.values, h.event.values)
    print(f"\nPRE-REGISTERED CRITERION: {'PASS' if h.clv.mean() > 0 and lo > 0 else 'FAIL'}")
    print(f"By month: {h.groupby(h.event.dt.to_period('M')).clv.agg(['count', 'mean']).round(4).to_dict('index')}")
    print(f"Median decimal price {h.price.median():.2f}; median hours before event {h.hours_before.median():.0f}")
    cut = h.clv.quantile(0.9)
    lm.summarise(h[h.clv <= cut], "holdout excl. top 10% of bets")
