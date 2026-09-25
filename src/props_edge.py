"""
Method-of-victory prop model (Week 9b). Six outcomes per fight:
R_KO, R_SUB, R_DEC, B_KO, B_SUB, B_DEC.

Each outcome is modelled as a binary event anchored on the market:
    p = sigmoid(a * logit(q_mkt) + b + w . features)
where q_mkt is the proportionally de-vigged prop probability. The a/b terms
learn the market's systematic miscalibration (e.g. longshot bias); features
add fighter finishing/durability history computed only from prior fights.

Run this file for dev-only research (never touches HOLDOUT_START onward).
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from market_edge import load, ip, dec, HOLDOUT_START

OUTCOMES = ["R_KO", "R_SUB", "R_DEC", "B_KO", "B_SUB", "B_DEC"]
ODDS_COL = {"R_KO": "r_ko_odds", "R_SUB": "r_sub_odds", "R_DEC": "r_dec_odds",
            "B_KO": "b_ko_odds", "B_SUB": "b_sub_odds", "B_DEC": "b_dec_odds"}
METHOD = {"KO/TKO": "KO", "SUB": "SUB", "U-DEC": "DEC", "S-DEC": "DEC", "M-DEC": "DEC"}


def add_method_history(m: pd.DataFrame) -> pd.DataFrame:
    """Per-fighter prior counts of wins/losses by method, updated after each card date
    (so fights on the same card never see each other)."""
    hist = {}
    cols = {f"{c}_{k}": [] for c in "RB" for k in
            ["n", "ko_w", "sub_w", "dec_w", "ko_l", "sub_l", "dec_l"]}
    pending = []
    last_date = None
    for r in m.itertuples(index=False):
        if r.date != last_date:
            for f, key in pending:
                h = hist.setdefault(f, dict.fromkeys(["n", "ko_w", "sub_w", "dec_w", "ko_l", "sub_l", "dec_l"], 0))
                h["n"] += 1
                if key:
                    h[key] += 1
            pending = []
            last_date = r.date
        for c, f in (("R", r.R_fighter), ("B", r.B_fighter)):
            h = hist.get(f, dict.fromkeys(["n", "ko_w", "sub_w", "dec_w", "ko_l", "sub_l", "dec_l"], 0))
            for k, v in h.items():
                cols[f"{c}_{k}"].append(v)
        meth = METHOD.get(r.finish)
        win_c = "R" if r.y == 1 else "B"
        for c, f in (("R", r.R_fighter), ("B", r.B_fighter)):
            key = None
            if meth:
                key = f"{meth.lower()}_{'w' if c == win_c else 'l'}"
            pending.append((f, key))
    for k, v in cols.items():
        m[k] = v
    return m


def build_long(m: pd.DataFrame) -> pd.DataFrame:
    """One row per (fight, outcome) with market prob, decimal odds, label, features."""
    m = m.dropna(subset=list(ODDS_COL.values())).copy()
    m = m[m.finish.isin(METHOD)].copy()
    imp = np.column_stack([ip(m[ODDS_COL[o]]) for o in OUTCOMES])
    m["prop_over"] = imp.sum(1)
    fair = imp / imp.sum(1, keepdims=True)
    win_c = np.where(m.y == 1, "R", "B")
    realized = np.array([f"{w}_{METHOD[f]}" for w, f in zip(win_c, m.finish)])
    rows = []
    for j, o in enumerate(OUTCOMES):
        c, meth = o.split("_")
        oc = "B" if c == "R" else "R"
        mk = meth.lower()
        n_self = m[f"{c}_n"] + 1
        n_opp = m[f"{oc}_n"] + 1
        wins_self = m[f"{c}_wins"] + 1
        career_col = {"KO": "win_by_KO/TKO", "SUB": "win_by_Submission"}.get(meth)
        if career_col:
            career_rate = m[f"{c}_{career_col}"] / wins_self
        else:
            career_rate = (m[f"{c}_win_by_Decision_Unanimous"] + m[f"{c}_win_by_Decision_Split"]
                           + m[f"{c}_win_by_Decision_Majority"]) / wins_self
        df = pd.DataFrame({
            "date": m.date.values,
            "fight_id": m.index.values,
            "outcome": o,
            "q": fair[:, j],
            "dec_odds": dec(m[ODDS_COL[o]].values),
            "label": (realized == o).astype(int),
            "p_win_mkt": np.where(c == "R", m.p_mkt, 1 - m.p_mkt),
            "self_meth_w_rate": m[f"{c}_{mk}_w"] / n_self,
            "opp_meth_l_rate": m[f"{oc}_{mk}_l"] / n_opp,
            "self_career_rate": career_rate.values,
            "self_n": np.log1p(m[f"{c}_n"]),
            "opp_n": np.log1p(m[f"{oc}_n"]),
            "five_rd": m.five_rd.values,
            "female": m.female.values,
            "heavy": m.heavy.values,
            "prop_over": m.prop_over.values,
            "is_ko": int(meth == "KO"), "is_sub": int(meth == "SUB"), "is_dec": int(meth == "DEC"),
        })
        rows.append(df)
    L = pd.concat(rows, ignore_index=True)
    L["q_logit"] = np.log(L.q / (1 - L.q))
    # market-miscalibration can differ by method: interact the anchor with method type
    for t in ["is_ko", "is_sub", "is_dec"]:
        L[f"q_logit_{t}"] = L.q_logit * L[t]
    L["win_logit"] = np.log(L.p_win_mkt / (1 - L.p_win_mkt))
    return L.sort_values(["date", "fight_id", "outcome"]).reset_index(drop=True)


ANCHOR = ["q_logit_is_ko", "q_logit_is_sub", "q_logit_is_dec", "is_ko", "is_sub"]
EXTRA = ["win_logit", "self_meth_w_rate", "opp_meth_l_rate", "self_career_rate",
         "self_n", "opp_n", "five_rd", "female", "heavy"]


def fit_predict(kind, tr, te, C=0.05):
    if kind == "market":
        return te.q.values
    feats = ANCHOR if kind == "recal" else ANCHOR + EXTRA
    if kind == "full":
        # method-specific copies of the extras so e.g. KO history only informs KO props
        for t in ["is_ko", "is_sub", "is_dec"]:
            for f in EXTRA:
                tr[f"{f}_{t}"] = tr[f] * tr[t]; te[f"{f}_{t}"] = te[f] * te[t]
        feats = ANCHOR + [f"{f}_{t}" for f in EXTRA for t in ["is_ko", "is_sub", "is_dec"]]
    X, Xt = tr[feats].fillna(0).astype(float), te[feats].fillna(0).astype(float)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C if kind != "recal" else 1e6, max_iter=5000).fit(sc.transform(X), tr.label)
    return lr.predict_proba(sc.transform(Xt))[:, 1]


def walk_forward(L, kind, years, holdout=False, **kw):
    out = []
    for yr in years:
        cut = max(pd.Timestamp(f"{yr}-01-01"), HOLDOUT_START) if holdout else pd.Timestamp(f"{yr}-01-01")
        tr = L[L.date < cut].copy()
        te = L[(L.date.dt.year == yr) & ((L.date >= HOLDOUT_START) if holdout else (L.date < HOLDOUT_START))].copy()
        if len(te) == 0:
            continue
        te["p"] = fit_predict(kind, tr, te, **kw)
        out.append(te)
    return pd.concat(out)


def select_bets(o, ev_min, max_odds=None, one_per_fight=True):
    o = o.copy()
    o["ev"] = o.p * o.dec_odds - 1
    b = o[o.ev > ev_min]
    if max_odds:
        b = b[b.dec_odds <= max_odds]
    if one_per_fight:
        b = b.sort_values("ev", ascending=False).drop_duplicates("fight_id")
    b = b.sort_values("date")
    b["pnl"] = np.where(b.label == 1, b.dec_odds - 1, -1.0)
    return b


def summary(pnl, seed=0):
    pnl = np.asarray(pnl)
    if len(pnl) == 0:
        return "n=0"
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(pnl, len(pnl)).mean() for _ in range(2000)])
    return (f"n={len(pnl):4d} roi={pnl.mean():+.3f} ci=[{np.percentile(bs,2.5):+.3f},"
            f"{np.percentile(bs,97.5):+.3f}] p(roi<=0)={(bs<=0).mean():.3f}")


def logloss_bin(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


if __name__ == "__main__":
    m = add_method_history(load())
    L = build_long(m)
    dev = L[L.date < HOLDOUT_START]
    print("dev fights", dev.fight_id.nunique(), "rows", len(dev))
    # market calibration by odds bucket (dev only)
    dev_b = dev.assign(bucket=pd.cut(dev.dec_odds, [1, 2, 3, 5, 8, 15, 1000]))
    dev_b["pnl"] = np.where(dev_b.label == 1, dev_b.dec_odds - 1, -1.0)
    print(dev_b.groupby("bucket").agg(n=("label", "size"), mkt=("q", "mean"), actual=("label", "mean"),
                                      roi_all=("pnl", "mean")).round(3))
    for kind in ["market", "recal", "full"]:
        for C in ([0.05] if kind != "full" else [0.01, 0.05, 0.2]):
            o = walk_forward(L, kind, range(2014, 2022), C=C)
            print(f"\n== {kind} C={C}: logloss {logloss_bin(o.label, o.p):.4f} (market {logloss_bin(o.label, o.q):.4f})")
            if kind == "market":
                continue
            for ev in [0.0, 0.05, 0.10, 0.20]:
                for mo in [None, 6.0]:
                    b = select_bets(o, ev, max_odds=mo)
                    print(f"  ev>{ev:.2f} max_odds={mo} {summary(b.pnl)}")
