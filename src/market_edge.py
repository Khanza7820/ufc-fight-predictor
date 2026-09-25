"""Market-anchored edge model (Week 9). Dev research: run this file. Locked holdout: src/market_edge_holdout.py."""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

HOLDOUT_START = pd.Timestamp("2021-04-01")

def ip(o):
    return np.where(o > 0, 100 / (o + 100), -o / (-o + 100))

def dec(o):
    return np.where(o > 0, o / 100 + 1, 100 / (-o) + 1)

def load():
    m = pd.read_csv(str(Path(__file__).resolve().parents[1] / "data" / "raw" / "ufc-master.csv")).copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m.iloc[::-1].reset_index(drop=True)  # file is date-descending
    m = m[m.Winner.isin(["Red", "Blue"])].dropna(subset=["R_odds", "B_odds"]).reset_index(drop=True)
    m["y"] = (m.Winner == "Red").astype(int)
    pr, pb = ip(m.R_odds), ip(m.B_odds)
    m["p_mkt"] = pr / (pr + pb)
    m["R_dec"], m["B_dec"] = dec(m.R_odds), dec(m.B_odds)
    m["mkt_logit"] = np.log(m.p_mkt / (1 - m.p_mkt))
    m["overround"] = pr + pb
    m["fav_is_red"] = (m.p_mkt >= 0.5).astype(int)
    m["abs_logit"] = m.mkt_logit.abs()
    # method-prop derived finish expectation (pre-fight prices)
    for c in ["r_dec_odds", "b_dec_odds", "r_sub_odds", "b_sub_odds", "r_ko_odds", "b_ko_odds"]:
        m[c + "_p"] = np.where(m[c].isna(), np.nan, ip(m[c].fillna(100)))
    m["dec_p"] = m.r_dec_odds_p + m.b_dec_odds_p
    # simple diffs (all pre-fight, verified)
    m["rank_R"] = m.R_match_weightclass_rank.fillna(20)
    m["rank_B"] = m.B_match_weightclass_rank.fillna(20)
    m["rank_diff"] = m.rank_R - m.rank_B
    m["exp_diff"] = (m.R_wins + m.R_losses) - (m.B_wins + m.B_losses)
    m["winpct_diff"] = (m.R_wins + 1) / (m.R_wins + m.R_losses + 2) - (m.B_wins + 1) / (m.B_wins + m.B_losses + 2)
    m["debut_R"] = ((m.R_wins + m.R_losses) == 0).astype(int)
    m["debut_B"] = ((m.B_wins + m.B_losses) == 0).astype(int)
    m["five_rd"] = (m.no_of_rounds == 5).astype(int)
    m["female"] = (m.gender == "FEMALE").astype(int)
    m["heavy"] = m.weight_class.isin(["Heavyweight", "Light Heavyweight"]).astype(int)
    add_elo(m)
    return m

def add_elo(m, k=32):
    elo = {}
    re, be = [], []
    for r, b, y in zip(m.R_fighter, m.B_fighter, m.y):
        er, eb = elo.get(r, 1500.0), elo.get(b, 1500.0)
        re.append(er); be.append(eb)
        e = 1 / (1 + 10 ** ((eb - er) / 400))
        elo[r] = er + k * (y - e)
        elo[b] = eb - k * (y - e)
    # rows on the same date use pre-card ratings for fighters' first bout that night;
    # a fighter never fights twice on one card, so sequential update is leak-free
    m["elo_diff"] = np.array(re) - np.array(be)
    m["elo_p"] = 1 / (1 + 10 ** (-m.elo_diff / 400))
    m["elo_vs_mkt"] = np.log(m.elo_p / (1 - m.elo_p)) - m.mkt_logit

FEATS = ["mkt_logit", "elo_vs_mkt", "rank_diff", "exp_diff", "winpct_diff", "age_dif", "reach_dif",
         "height_dif", "sig_str_dif", "avg_td_dif", "avg_sub_att_dif", "win_streak_dif",
         "lose_streak_dif", "debut_R", "debut_B", "five_rd", "female", "heavy", "title_bout"]

def fit_predict(kind, tr, te, C=0.05):
    if kind == "market":
        return te.p_mkt.values
    if kind == "mkt_recal":
        lr = LogisticRegression(C=1e6).fit(tr[["mkt_logit"]], tr.y)
        return lr.predict_proba(te[["mkt_logit"]])[:, 1]
    if kind == "lr":
        X = tr[FEATS].fillna(0).astype(float); Xt = te[FEATS].fillna(0).astype(float)
        sc = StandardScaler().fit(X)
        lr = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X), tr.y)
        return lr.predict_proba(sc.transform(Xt))[:, 1]
    if kind == "xgb_resid":
        f = [c for c in FEATS if c != "mkt_logit"]
        mdl = XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.03, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=20, reg_lambda=5, random_state=42)
        mdl.fit(tr[f].astype(float), tr.y, base_margin=tr.mkt_logit)
        return mdl.predict_proba(te[f].astype(float), base_margin=te.mkt_logit)[:, 1]
    raise ValueError(kind)

def walk_forward(d, kind, start_year=2013, **kw):
    out = []
    for yr in range(start_year, 2022):
        tr = d[d.date.dt.year < yr]
        te = d[(d.date.dt.year == yr)]
        if len(te) == 0:
            continue
        te = te.copy(); te["p"] = fit_predict(kind, tr, te, **kw)
        out.append(te)
    return pd.concat(out)

def bets(o, ev_min=0.0, p_min=0.0, side="both"):
    ev_r = o.p * o.R_dec - 1
    ev_b = (1 - o.p) * o.B_dec - 1
    pick_r = ev_r >= ev_b
    ev = np.where(pick_r, ev_r, ev_b)
    pw = np.where(pick_r, o.p, 1 - o.p)
    dd = np.where(pick_r, o.R_dec, o.B_dec)
    won = np.where(pick_r, o.y == 1, o.y == 0)
    mkt_p = np.where(pick_r, o.p_mkt, 1 - o.p_mkt)
    mask = (ev > ev_min) & (pw >= p_min)
    if side == "fav":
        mask &= mkt_p >= 0.5
    if side == "dog":
        mask &= mkt_p < 0.5
    pnl = np.where(won, dd - 1, -1.0)[mask]
    return pnl, o.date.values[mask]

def summary(pnl):
    if len(pnl) == 0:
        return "n=0"
    rng = np.random.default_rng(0)
    bs = [rng.choice(pnl, len(pnl)).mean() for _ in range(2000)]
    return f"n={len(pnl):4d} roi={pnl.mean():+.3f} ci=[{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]"

if __name__ == "__main__":
    d = load()
    d = d[d.date < HOLDOUT_START].reset_index(drop=True)  # HOLDOUT LOCKED
    print("dev rows", len(d), d.date.min().date(), d.date.max().date())
    for kind in ["market", "mkt_recal", "lr", "xgb_resid"]:
        o = walk_forward(d, kind)
        print(f"\n== {kind}: logloss {log_loss(o.y, o.p):.4f} (market {log_loss(o.y, o.p_mkt):.4f})")
        for ev_min in [0.0, 0.03, 0.06]:
            for side in ["both", "fav", "dog"]:
                pnl, _ = bets(o, ev_min=ev_min, side=side)
                print(f"  ev>{ev_min:.2f} {side:4s} {summary(pnl)}")
    # structural: bet every market favourite above a fair-prob floor
    o = walk_forward(d, "market")
    print("\n== blanket favourite betting (no model) ==")
    for lo in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]:
        fav_r = o.p_mkt >= 0.5
        fp = np.where(fav_r, o.p_mkt, 1 - o.p_mkt)
        won = np.where(fav_r, o.y == 1, o.y == 0)
        dd = np.where(fav_r, o.R_dec, o.B_dec)
        mk = fp >= lo
        print(f"  fav_p>={lo:.2f} {summary(np.where(won, dd-1, -1.0)[mk])}")

