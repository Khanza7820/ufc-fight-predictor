"""
Accuracy evidence for the frozen market-edge model on the locked holdout.

ROI only uses fights the strategy bet on and is dominated by payout variance.
This tests the underlying claim - "the model's probabilities are more accurate
than the market's" - on EVERY holdout fight, via the paired per-fight log-loss
difference (a Diebold-Mariano-style test). Resampling is by event date (card),
since fights on the same card share conditions. Nothing is selected or tuned.
"""
from market_edge import *
from market_edge_report import holdout_predictions


def ll(y, p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def card_bootstrap(diff, dates, n=10000, seed=0):
    cards = pd.Series(diff).groupby(pd.Series(dates).values).agg(["sum", "count"])
    s, c = cards["sum"].values, cards["count"].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n, len(s)))
    return s[idx].sum(1) / c[idx].sum(1)


def main():
    o = holdout_predictions(load())
    y = o.y.values
    diff = ll(y, o.p_mkt.values) - ll(y, o.p.values)  # > 0 means model better
    bs = card_bootstrap(diff, o.date.values)
    print(f"Holdout fights: {len(o)} on {o.date.nunique()} cards")
    print(f"Log loss: market {ll(y, o.p_mkt.values).mean():.4f}, model {ll(y, o.p.values).mean():.4f}")
    print(f"Mean improvement per fight: {diff.mean():+.5f} "
          f"95% CI [{np.percentile(bs, 2.5):+.5f}, {np.percentile(bs, 97.5):+.5f}]")
    print(f"One-sided p(model no better than market) = {(bs <= 0).mean():.4f}")

    print("\nBy year (improvement > 0 = model more accurate):")
    for yr, g in o.assign(diff=diff).groupby(o.date.dt.year):
        print(f"  {yr}: {g['diff'].mean():+.5f}  ({len(g)} fights)")

    # Same test on the development walk-forward, for comparison
    d = load()
    dev = walk_forward(d[d.date < HOLDOUT_START].reset_index(drop=True), "lr")
    ddiff = ll(dev.y.values, dev.p_mkt.values) - ll(dev.y.values, dev.p.values)
    dbs = card_bootstrap(ddiff, dev.date.values)
    print(f"\nDevelopment (2013-Mar 2021): improvement {ddiff.mean():+.5f} "
          f"95% CI [{np.percentile(dbs, 2.5):+.5f}, {np.percentile(dbs, 97.5):+.5f}] "
          f"p={(dbs <= 0).mean():.4f}")


if __name__ == "__main__":
    main()
