"""
PRE-REGISTERED HOLDOUT TEST (Week 10) - written before any holdout row was scored.

Strategy (frozen from dev, events 2025-07-27 .. 2026-03-31):
  Pinnacle value betting. Monitor every pre-close snapshot; bet a side at the
  first snapshot where the second-best non-Pinnacle price beats Pinnacle's
  de-vigged fair line by > 2% (thr = 0.02). One bet per fight, flat stakes.
  Dev result: n=139, CLV +2.20%, 95% CI [+1.09%, +3.40%], 65% beat the close.

Holdout: events 2026-04-01 onward whose close is observable (event day at
least one day before the last scrape). Never used for any decision.

Primary metric: mean CLV = price * consensus_close_fair_p - 1.
Pass: mean CLV > 0 AND card-bootstrap 95% CI lower bound > 0.
Usage: python line_move_holdout.py <UFC_betting_odds.csv>
"""
import sys
from line_move import pinnacle_any_snapshot, summarise, DEV_END
from market_edge_clv import card_boot

THR = 0.02

if __name__ == "__main__":
    b = pinnacle_any_snapshot(sys.argv[1], THR)
    h = b[b.event >= DEV_END]
    print(f"HOLDOUT events {h.event.min().date()} .. {h.event.max().date()}, {h.event.nunique()} cards")
    summarise(h, "pinnacle value thr>0.02")
    lo, hi, _ = card_boot(h.clv.values, h.event.values)
    print(f"\nPRE-REGISTERED CRITERION: {'PASS' if h.clv.mean() > 0 and lo > 0 else 'FAIL'}")
    print(f"By month: {h.groupby(h.event.dt.to_period('M')).clv.agg(['count', 'mean']).round(4).to_dict('index')}")
    print(f"Median hours before event at bet time: {h.hours_before.median():.0f}; "
          f"median decimal price {h.price.median():.2f}")
