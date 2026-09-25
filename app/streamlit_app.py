"""
Streamlit dashboard for the UFC fight predictor.

Scope: browses REAL historical fights already in the dataset (not
hypothetical future matchups - see reports/methodology.md for why this
was the deliberate Week 7 scope choice, with live arbitrary-matchup
inference noted as a Week 8 extension).

Framing: given the Week 5-6 finding (no statistically significant
betting edge vs bookmaker odds), this app is presented as a model-
transparency tool - what the model predicts, why (SHAP), and how that
compares to the market - not a betting tool.

Run from the repo root: streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from src.baseline_model import get_feature_columns, prepare_target
from src.shap_analysis import train_final_model
from src.odds_utils import add_implied_probabilities

st.set_page_config(page_title="UFC Fight Predictor", layout="centered")


# ---------- cached data / model loading ----------

@st.cache_data
def load_fights_with_odds():
    """
    Loads the same feature matrix + odds join used throughout the
    project, restricted to fights with a genuine outcome and matched
    odds, so every fight in the dropdown has a real prediction, real
    SHAP explanation, and a real market line to compare against.
    """
    df = pd.read_csv('data/processed/features.csv')
    df['date'] = pd.to_datetime(df['date'])
    df_prepped = prepare_target(df)
    df_clean = df_prepped[df_prepped['date'] >= '2010-03-21'].reset_index(drop=True)

    odds_df = pd.read_csv('data/raw/ufc-master.csv')
    odds_df['date'] = pd.to_datetime(odds_df['date'])
    odds_df_with_probs = add_implied_probabilities(odds_df)

    merged = df_clean.merge(
        odds_df_with_probs[['R_fighter', 'B_fighter', 'date', 'R_odds', 'B_odds',
                             'R_implied_prob', 'B_implied_prob']],
        on=['R_fighter', 'B_fighter', 'date'], how='inner'
    ).dropna(subset=['R_odds', 'B_odds']).sort_values('date', ascending=False).reset_index(drop=True)

    return df_clean, merged


@st.cache_resource
def load_model_and_explainer(_df_clean, feature_cols, best_params):
    """
    Trains the SHAP "final" model (fit on all data - correct here,
    since this app explains the deployed model's reasoning, not its
    out-of-sample accuracy; see shap_analysis.py docstring).
    Leading underscore on _df_clean tells st.cache_resource not to
    hash the (large, unhashable) DataFrame as a cache key.
    """
    model = train_final_model(_df_clean, feature_cols, best_params)
    explainer = shap.TreeExplainer(model)
    return model, explainer


# NOTE: matches the tuned params confirmed in notebooks/07_dashboard_prep.ipynb
BEST_PARAMS = {
    'n_estimators': 275,
    'max_depth': 5,
    'learning_rate': 0.010449979777117415,
    'random_state': 42,
}


# ---------- header ----------

st.title("UFC Fight Outcome Predictor")
st.caption(
    "Calibrated win-probability model with SHAP explainability, browsing real "
    "historical UFC fights. **Backtesting found no statistically significant "
    "edge against bookmaker odds** (Sharpe 95% CI includes zero) — see the "
    "full tearsheet for methodology. This tool demonstrates model reasoning "
    "and transparency; it is not betting guidance."
)

df_clean, fights = load_fights_with_odds()
feature_cols = get_feature_columns(df_clean)
model, explainer = load_model_and_explainer(df_clean, feature_cols, BEST_PARAMS)


# ---------- fight selector ----------

fights['label'] = (
    fights['date'].dt.strftime('%Y-%m-%d') + " — " +
    fights['R_fighter'] + " vs " + fights['B_fighter']
)
selected_label = st.selectbox(
    "Select a historical fight",
    fights['label'],
    index=0,
)
fight_row = fights[fights['label'] == selected_label].iloc[0]


# ---------- prediction ----------

X_fight = fight_row[feature_cols].to_frame().T.astype(float)
model_prob_red = model.predict_proba(X_fight)[:, 1][0]
model_prob_blue = 1 - model_prob_red

red_won = fight_row['target'] == 1

st.subheader(f"{fight_row['R_fighter']} vs {fight_row['B_fighter']}")
st.caption(f"{fight_row['date'].strftime('%B %d, %Y')} · Actual winner: "
           f"**{fight_row['R_fighter'] if red_won else fight_row['B_fighter']}**")

col1, col2 = st.columns(2)
with col1:
    st.metric(
        f"Model: P({fight_row['R_fighter']} wins)",
        f"{model_prob_red:.1%}",
        delta="Won" if red_won else "Lost",
        delta_color="normal" if red_won else "inverse",
    )
with col2:
    st.metric(
        f"Model: P({fight_row['B_fighter']} wins)",
        f"{model_prob_blue:.1%}",
        delta="Won" if not red_won else "Lost",
        delta_color="normal" if not red_won else "inverse",
    )

# ---------- market comparison ----------

st.markdown("#### vs. Bookmaker Line")
implied_red = fight_row['R_implied_prob']
implied_blue = fight_row['B_implied_prob']
edge_red = model_prob_red - implied_red
edge_blue = model_prob_blue - implied_blue

market_col1, market_col2 = st.columns(2)
with market_col1:
    st.metric(f"Market implied P({fight_row['R_fighter']})", f"{implied_red:.1%}",
               delta=f"{edge_red:+.1%} vs model")
with market_col2:
    st.metric(f"Market implied P({fight_row['B_fighter']})", f"{implied_blue:.1%}",
               delta=f"{edge_blue:+.1%} vs model")

st.caption(
    "Positive delta = model more confident than the market on that side. "
    "Backtesting showed these deltas do not translate into a reliable edge — "
    "see the tearsheet's edge decile analysis for why."
)

# ---------- SHAP explanation ----------

st.markdown("#### Top drivers of this prediction")
sv = explainer.shap_values(X_fight)[0]

top_idx = np.argsort(np.abs(sv))[::-1][:3]
for i in top_idx:
    feat_name = feature_cols[i]
    feat_val = X_fight.iloc[0, i]
    contribution = sv[i]
    direction = f"toward {fight_row['R_fighter']}" if contribution > 0 else f"toward {fight_row['B_fighter']}"
    st.write(f"**{feat_name}** = {feat_val:.2f} → pushed prediction {direction} "
             f"(SHAP: {contribution:+.3f})")

fig, ax = plt.subplots(figsize=(8, 5))
shap.plots.waterfall(shap.Explanation(
    values=sv,
    base_values=explainer.expected_value,
    data=X_fight.iloc[0].values,
    feature_names=feature_cols,
), max_display=10, show=False)
st.pyplot(fig)

st.divider()
st.caption(
    "Full methodology, backtest results, and the honest negative finding on "
    "market edge: see reports/tearsheet.pdf and reports/methodology.md."
)