"""
Early-line value model (Week 10).

Bets at the FIRST observed snapshot for each fight (typically ~4-5 days out),
at the second-best price across bookmakers (conservative: a single stale or
mistyped price can't create a fake edge), and is scored by closing-line value:
    CLV = bet_price * close_fair_p - 1
where close_fair_p is the median de-vigged probability across books at the
last snapshot before 00:00 UTC on event day.

Candidates (compared on dev only):
  consensus : bet when price beats the t0 consensus fair line by > thr
  pinnacle  : bet when price beats Pinnacle's t0 fair line by > thr
  model     : ridge regression predicts close_fair_p from t0 features;
              bet when price * predicted close p - 1 > thr

Data: Kaggle "UFC Betting Odds (Daily Updated Dataset)" (jerzyszocik, CC0).
DEV: events 2025-07-27 .. 2026-03-31.  HOLDOUT: 2026-04-01 .. end of data.
Run this file for dev research: python line_move.py <UFC_betting_odds.csv>
"""
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from market_edge_clv import load_snapshots, card_boot

DEV_END = pd.Timestamp("2026-04-01")


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def build_fights(path):
    k = load_snapshots(path)
    k = k[k.ts < k.event.dt.tz_localize("UTC")]
    data_end = k.ts.max()
    rows = []
    for (ev, n1, n2), g in k.groupby(["event", "n1", "n2"]):
        # the close must be observable: event day must be before the last scrape
        if pd.Timestamp(ev, tz="UTC") > data_end - pd.Timedelta(days=1):
            continue
        times = np.sort(g.ts.unique())
        if len(times) < 2:
            continue
        t0, tc = times[0], times[-1]
        o, c = g[g.ts == t0], g[g.ts == tc]
        if len(o) < 3 or len(c) < 3:
            continue
        pin = o[o.source == "Pinnacle"]
        p_c = c.p1.median()
        for side, pcol, oc in ((1, "p1", "odds_1"), (2, "p1", "odds_2")):
            f = (lambda p: p) if side == 1 else (lambda p: 1 - p)
            prices = np.sort(o[oc].values)[::-1]
            rows.append({
                "event": ev, "fight": f"{ev.date()}|{n1}|{n2}", "side": side,
                "open_p": f(o.p1.median()),
                "pin_p": f(pin.p1.median()) if len(pin) else np.nan,
                "disp": o.p1.std(),
                "n_books": len(o),
                "lead_days": (pd.Timestamp(ev, tz="UTC") - t0).total_seconds() / 86400,
                "best": prices[0], "price": prices[1],
                "close_p": f(p_c),
            })
    F = pd.DataFrame(rows)
    F["open_logit"] = logit(F.open_p)
    F["pin_gap"] = (logit(F.pin_p) - F.open_logit).fillna(0)
    F["has_pin"] = F.pin_p.notna().astype(int)
    F["price_gap"] = logit(1 / F.price) - F.open_logit  # <0: price longer than consensus
    F["clv"] = F.price * F.close_p - 1
    return F


FEATS = ["open_logit", "pin_gap", "has_pin", "price_gap", "disp", "n_books", "lead_days"]


def predict_close(tr, te, alpha=10.0):
    y = logit(tr.close_p) - tr.open_logit
    m = Ridge(alpha=alpha).fit(tr[FEATS], y)
    return 1 / (1 + np.exp(-(te.open_logit + m.predict(te[FEATS]))))


def signal(F, kind, train=None):
    if kind == "consensus":
        return F.price * F.open_p - 1
    if kind == "pinnacle":
        return F.price * F.pin_p - 1  # NaN (no bet) where Pinnacle absent
    if kind == "model":
        return F.price * predict_close(train, F) - 1
    raise ValueError(kind)


def walk_forward_model(F, months):
    out = []
    for m0 in months:
        lo, hi = m0.to_timestamp(), (m0 + 1).to_timestamp()
        tr = F[F.event < lo]
        te = F[(F.event >= lo) & (F.event < hi)].copy()
        if len(tr) < 200 or len(te) == 0:
            continue
        te["sig"] = signal(te, "model", tr)
        out.append(te)
    return pd.concat(out)


def pick(F, thr):
    b = F[F.sig > thr]
    return b.sort_values("sig", ascending=False).drop_duplicates("fight")


def summarise(b, label):
    if len(b) < 5:
        print(f"  {label:28s} n={len(b)}")
        return
    lo, hi, p = card_boot(b.clv.values, b.event.values)
    print(f"  {label:28s} n={len(b):4d} CLV {b.clv.mean():+.4f} CI [{lo:+.4f}, {hi:+.4f}] "
          f"p(<=0)={p:.3f} beat close {np.mean(b.clv > 0):.0%}")


def pinnacle_any_snapshot(path, thr, min_other_books=2):
    """Monitor every pre-close snapshot; bet a side at the FIRST snapshot where the
    second-best non-Pinnacle price beats Pinnacle's de-vigged fair line by > thr.
    One bet per fight. Scored by CLV against the consensus close."""
    k = load_snapshots(path)
    k = k[k.ts < k.event.dt.tz_localize("UTC")]
    data_end = k.ts.max()
    rows = []
    for (ev, n1, n2), g in k.groupby(["event", "n1", "n2"]):
        if pd.Timestamp(ev, tz="UTC") > data_end - pd.Timedelta(days=1):
            continue
        times = np.sort(g.ts.unique())
        if len(times) < 2:
            continue
        c = g[g.ts == times[-1]]
        if len(c) < 3:
            continue
        p_close1 = c.p1.median()
        for t in times[:-1]:  # the closing snapshot itself is never a bet time
            s = g[g.ts == t]
            pin, soft = s[s.source == "Pinnacle"], s[s.source != "Pinnacle"]
            if pin.empty or len(soft) < min_other_books:
                continue
            pin_p1 = pin.p1.median()
            best = None
            for side, oc, pp, pc in ((1, "odds_1", pin_p1, p_close1), (2, "odds_2", 1 - pin_p1, 1 - p_close1)):
                price = np.sort(soft[oc].values)[::-1][1]
                edge = price * pp - 1
                if edge > thr and (best is None or edge > best["sig"]):
                    best = {"event": ev, "fight": f"{ev.date()}|{n1}|{n2}", "side": side, "sig": edge,
                            "price": price, "close_p": pc, "clv": price * pc - 1,
                            "hours_before": (pd.Timestamp(ev, tz="UTC") - t).total_seconds() / 3600}
            if best:
                rows.append(best)
                break
    return pd.DataFrame(rows)


if __name__ == "__main__":
    F = build_fights(sys.argv[1])
    dev = F[F.event < DEV_END].copy()
    print(f"dev fights {dev.fight.nunique()} on {dev.event.nunique()} cards; "
          f"all-sides baseline CLV at 2nd-best price {dev.clv.mean():+.4f}")
    for kind in ["consensus", "pinnacle"]:
        dev["sig"] = signal(dev, kind)
        print(f"\n== {kind}")
        for thr in [0.0, 0.01, 0.02, 0.03, 0.05]:
            summarise(pick(dev, thr), f"thr>{thr:.2f}")
    # model: walk-forward by month within dev (first months are training only)
    months = pd.period_range("2025-10", "2026-03", freq="M")
    wf = walk_forward_model(dev, months)
    print(f"\n== model (walk-forward, test months {months[0]}..{months[-1]})")
    for thr in [0.0, 0.01, 0.02, 0.03, 0.05]:
        summarise(pick(wf, thr), f"thr>{thr:.2f}")
    for kind in ["consensus", "pinnacle"]:
        sub = wf.copy(); sub["sig"] = signal(sub, kind)
        summarise(pick(sub, 0.0), f"{kind} thr>0 same months")
    print("\n== pinnacle value, monitoring every pre-close snapshot")
    for thr in [0.0, 0.01, 0.02, 0.03]:
        b = pinnacle_any_snapshot(sys.argv[1], thr)
        summarise(b[b.event < DEV_END], f"thr>{thr:.2f}")
