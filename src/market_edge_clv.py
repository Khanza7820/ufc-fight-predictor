"""
Closing-line-value (CLV) test of the frozen market-edge strategy.

Written BEFORE any CLV number was computed. The picks are the pre-registered
holdout bets (lr, C=0.05, EV > 0.03); nothing is re-selected or tuned.

Data: Kaggle "UFC Betting Odds (Daily Updated Dataset)" (jerzyszocik, CC0),
time-stamped prices from ~35 bookmakers. Snapshots start 2025-07-26, so the
test covers holdout bets on events from 2025-07-27 to 2026-03-28 only.

Definitions
- A price row is usable if both odds are in [1.01, 50] and the book's
  overround is in [1.00, 1.15] (drops exchange artefacts such as 1.03/1.03).
- Per-book fair probability = proportional de-vig of that book's two prices.
- Consensus fair probability at a snapshot = median across books.
- OPEN  = earliest snapshot for the fight.
- CLOSE = latest snapshot taken before 00:00 UTC on the event date
  (conservative: guarantees no snapshot can post-date any fight on the card).

Metrics (picked side)
1. PRIMARY  CLV at our price: bet_decimal * close_fair_p - 1, where bet_decimal
   is the ufc-master price the strategy actually used. > 0 = beat the close.
2. Line movement: close_fair_p - open_fair_p. > 0 = market moved toward pick.
3. Pinnacle-only versions of 1 and 2 (sharpest single book).
4. Executable CLV: best price across books at OPEN * close_fair_p - 1.
Diagnostic: is the ufc-master price nearer the OPEN or the CLOSE consensus?

Pass: metric 1 mean > 0 with bootstrap 95% CI lower bound > 0 (resampled by card).

Usage: python market_edge_clv.py <path to UFC_betting_odds.csv>
"""
import sys
from market_edge import *
from market_edge_report import holdout_predictions
from market_edge_replication import norm

CLV_START = pd.Timestamp("2025-07-27")


def load_snapshots(path):
    k = pd.read_csv(path, low_memory=False)
    k["event"] = pd.to_datetime(k.event_date)
    k["ts"] = pd.to_datetime(k.adding_date, utc=True, format="mixed")
    k = k[(k.event >= CLV_START) & k.odds_1.between(1.01, 50) & k.odds_2.between(1.01, 50)].copy()
    over = 1 / k.odds_1 + 1 / k.odds_2
    k = k[over.between(1.0, 1.15)].copy()
    k["p1"] = (1 / k.odds_1) / (1 / k.odds_1 + 1 / k.odds_2)
    k["n1"], k["n2"] = k.fighter_1.map(norm), k.fighter_2.map(norm)
    return k


def fight_lines(k):
    """Per fight: open/close consensus fair prob for fighter_1, Pinnacle open/close,
    best open decimal price for each fighter."""
    out = {}
    for (ev, n1, n2), g in k.groupby(["event", "n1", "n2"]):
        cutoff = pd.Timestamp(ev, tz="UTC")
        pre = g[g.ts < cutoff]
        if pre.empty:
            continue
        t_open, t_close = pre.ts.min(), pre.ts.max()
        o, c = pre[pre.ts == t_open], pre[pre.ts == t_close]
        pin = pre[pre.source == "Pinnacle"]
        rec = {
            "n_snaps": pre.ts.nunique(),
            "open_p1": o.p1.median(), "close_p1": c.p1.median(),
            "best_open_1": o.odds_1.max(), "best_open_2": o.odds_2.max(),
            "pin_open_p1": pin[pin.ts == pin.ts.min()].p1.median() if len(pin) else np.nan,
            "pin_close_p1": pin[pin.ts == pin.ts.max()].p1.median() if len(pin) else np.nan,
        }
        out[(ev, frozenset([n1, n2]))] = (n1, rec)
    return out


def card_boot(x, cards, n=10000, seed=0):
    df = pd.DataFrame({"x": x, "c": cards}).groupby("c").x.agg(["sum", "count"])
    s, c = df["sum"].values, df["count"].values
    idx = np.random.default_rng(seed).integers(0, len(s), size=(n, len(s)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5), (bs <= 0).mean()


def report(name, x, cards):
    x = np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    lo, hi, p = card_boot(x[ok], np.asarray(cards)[ok])
    print(f"  {name:44s} n={ok.sum():3d} mean {x[ok].mean():+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
          f"p(<=0)={p:.3f}  positive {np.mean(x[ok] > 0):.0%}")


def main(path):
    lines = fight_lines(load_snapshots(path))
    o = holdout_predictions(load())
    ev_r, ev_b = o.p * o.R_dec - 1, (1 - o.p) * o.B_dec - 1
    red = ev_r >= ev_b
    sel = np.where(red, ev_r, ev_b) > 0.03
    picks = o[sel].assign(side_red=red[sel])
    picks = picks[picks.date >= CLV_START]
    picks["pick"] = np.where(picks.side_red, picks.R_fighter, picks.B_fighter)
    picks["bet_dec"] = np.where(picks.side_red, picks.R_dec, picks.B_dec)
    picks["pick_mkt_p"] = np.where(picks.side_red, picks.p_mkt, 1 - picks.p_mkt)
    picks["won"] = np.where(picks.side_red, picks.y == 1, picks.y == 0)

    rows = []
    for r in picks.itertuples(index=False):
        pair = frozenset([norm(r.R_fighter), norm(r.B_fighter)])
        hit = None
        for dd in (0, -1, 1):
            hit = lines.get((r.date + pd.Timedelta(days=dd), pair))
            if hit:
                break
        if not hit:
            continue
        n1, L = hit
        is1 = norm(r.pick) == n1
        f = (lambda p: p) if is1 else (lambda p: 1 - p)
        rows.append({
            "date": r.date, "won": r.won, "bet_dec": r.bet_dec, "pick_mkt_p": r.pick_mkt_p,
            "n_snaps": L["n_snaps"],
            "open_p": f(L["open_p1"]), "close_p": f(L["close_p1"]),
            "pin_open_p": f(L["pin_open_p1"]), "pin_close_p": f(L["pin_close_p1"]),
            "best_open": L["best_open_1"] if is1 else L["best_open_2"],
        })
    t = pd.DataFrame(rows)
    print(f"Holdout bets on events from {CLV_START.date()}: {len(picks)}; matched to snapshots: {len(t)} "
          f"on {t.date.nunique()} cards; median snapshots per fight {t.n_snaps.median():.0f}")

    print("\nDiagnostic - which consensus line is the ufc-master price closer to?")
    print(f"  mean |ufc-master fair p - OPEN|  = {np.mean(np.abs(t.pick_mkt_p - t.open_p)):.4f}")
    print(f"  mean |ufc-master fair p - CLOSE| = {np.mean(np.abs(t.pick_mkt_p - t.close_p)):.4f}")

    print("\nResults (picked side):")
    report("1. PRIMARY CLV at our price vs consensus close", t.bet_dec * t.close_p - 1, t.date)
    report("2. line movement, consensus (close - open)", t.close_p - t.open_p, t.date)
    report("3a. CLV at our price vs Pinnacle close", t.bet_dec * t.pin_close_p - 1, t.date)
    report("3b. line movement, Pinnacle", t.pin_close_p - t.pin_open_p, t.date)
    report("4. executable CLV: best open price vs close", t.best_open * t.close_p - 1, t.date)
    print(f"\nFor context - realised flat-stake ROI on these {len(t)} bets: "
          f"{np.where(t.won, t.bet_dec - 1, -1.0).mean():+.2%}")


if __name__ == "__main__":
    main(sys.argv[1])
