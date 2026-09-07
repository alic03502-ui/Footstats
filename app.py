import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import streamlit as st
import pandas as pd
from pathlib import Path
from footstats.core.poisson_bayesian import predict_match_bayesian
from footstats.core.markets import build_market_catalog
# ---------------------------------------------------------
# PAGE
# ---------------------------------------------------------
st.set_page_config(
    page_title="FootStats Predictor",
    page_icon="⚽",
    layout="wide",
)
st.title("⚽ FootStats — Real Prediction Engine")
st.caption("Bayesian Poisson + Dixon-Coles market engine")
# ---------------------------------------------------------
# LOAD HISTORICAL DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    path = Path("data/hist_cache/full_dataset.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"Historical dataset not found: {path}"
        )
    df = pd.read_parquet(path)
    # Adapter: historical loader → Bayesian model schema
    df = df.rename(
        columns={
            "home": "gospodarz",
            "away": "goscie",
            "hg": "gole_g",
            "ag": "gole_a",
        }
    )
    # Make sure goals are numeric
    df["gole_g"] = pd.to_numeric(
        df["gole_g"],
        errors="coerce",
    )
    df["gole_a"] = pd.to_numeric(
        df["gole_a"],
        errors="coerce",
    )
    df = df.dropna(
        subset=[
            "gospodarz",
            "goscie",
            "gole_g",
            "gole_a",
        ]
    )
    return df
try:
    df = load_data()
except Exception as e:
    st.error("Could not load the FootStats historical dataset.")
    st.exception(e)
    st.stop()
# ---------------------------------------------------------
# DATA STATUS
# ---------------------------------------------------------
st.success(
    f"Historical database loaded: {len(df):,} matches"
)
# ---------------------------------------------------------
# LEAGUE SELECTION
# ---------------------------------------------------------
if "league" in df.columns:
    leagues = sorted(
        df["league"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
else:
    leagues = ["All leagues"]
league = st.selectbox(
    "League",
    leagues,
)
# Filter league
if league != "All leagues":
    league_df = df[
        df["league"].astype(str) == league
    ].copy()
else:
    league_df = df.copy()
# ---------------------------------------------------------
# TEAM SELECTION
# ---------------------------------------------------------
teams = sorted(
    set(
        league_df["gospodarz"].dropna().astype(str)
    ).union(
        set(
            league_df["goscie"].dropna().astype(str)
        )
    )
)
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox(
        "Home Team",
        teams,
        index=0,
    )
with col2:
    away_options = [
        team for team in teams
        if team != home_team
    ]
    away_team = st.selectbox(
        "Away Team",
        away_options,
        index=0,
    )
    prediction_date = st.date_input(
    "Match Date",
    value=pd.Timestamp.today().date()
)
# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------
predict_button = st.button(
    "🔮 Predict Match",
    type="primary",
    use_container_width=True,
)
if predict_button:
    with st.spinner("Running FootStats model..."):
        # Only use matches before the prediction.
        # The current dataset contains historical completed matches.
        model_df = league_df.copy()
        prediction_date = pd.Timestamp(prediction_date)

model_df["date"] = pd.to_datetime(model_df["date"], errors="coerce")

model_df = model_df[
    model_df["date"] < prediction_date
].copy()
        prediction = predict_match_bayesian(
    home_team,
    away_team,
    model_df
)
# ─────────────────────────────────────────────
# TEAM HISTORY DEBUG
# ─────────────────────────────────────────────

home_history = model_df[
    (model_df["home"] == home_team) |
    (model_df["away"] == home_team)
].copy()

away_history = model_df[
    (model_df["home"] == away_team) |
    (model_df["away"] == away_team)
].copy()

st.subheader("Historical Data Used")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Home Team Historical Matches",
        len(home_history)
    )

with col2:
    st.metric(
        "Away Team Historical Matches",
        len(away_history)
    )

if not home_history.empty:
    st.caption(
        f"{home_team}: "
        f"{home_history['date'].min().date()} → "
        f"{home_history['date'].max().date()}"
    )

if not away_history.empty:
    st.caption(
        f"{away_team}: "
        f"{away_history['date'].min().date()} → "
        f"{away_history['date'].max().date()}"
    )
    # -----------------------------------------------------
    # CHECK MODEL RESULT
    # -----------------------------------------------------
    if prediction is None:
        st.error(
            "The model could not generate a prediction. "
            "The selected teams may not have enough historical data."
        )
        st.stop()
    # -----------------------------------------------------
    # MAIN MODEL OUTPUT
    # -----------------------------------------------------
    st.divider()
    st.header(
        f"{home_team} vs {away_team}"
    )
    # Expected goals
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Home Expected Goals",
            f"{prediction['lambda_g']:.2f}",
        )
    with c2:
        st.metric(
            "Away Expected Goals",
            f"{prediction['lambda_a']:.2f}",
        )
    with c3:
        st.metric(
            "Expected Total Goals",
            f"{prediction['lambda_g'] + prediction['lambda_a']:.2f}",
        )
    # -----------------------------------------------------
    # 1X2
    # -----------------------------------------------------
    st.subheader("🎯 1X2")
    c1, c2, c3 = st.columns(3)
    with c1:
        p = prediction["pw"]
        st.metric(
            "1 — Home",
            f"{p * 100:.1f}%",
            f"Fair {1 / p:.2f}" if p > 0 else None,
        )
    with c2:
        p = prediction["pr"]
        st.metric(
            "X — Draw",
            f"{p * 100:.1f}%",
            f"Fair {1 / p:.2f}" if p > 0 else None,
        )
    with c3:
        p = prediction["pa"]
        st.metric(
            "2 — Away",
            f"{p * 100:.1f}%",
            f"Fair {1 / p:.2f}" if p > 0 else None,
        )
    # -----------------------------------------------------
    # FULL MARKET CATALOG
    # -----------------------------------------------------
    st.divider()
    st.header("📊 Betting Markets")
    markets = build_market_catalog(
    prediction["lambda_g"],
    prediction["lambda_a"],
    rho=0.0,
)
    for group in markets:
        st.subheader(
            group["grupa"]
        )
        rows = []
        for market in group["rynki"]:
            rows.append(
                {
                    "Market": market["rynek"],
                    "Tip": market["tip"],
                    "Probability": f"{market['szansa']:.1f}%",
                    "Fair Odds": f"{market['kurs']:.2f}",
                    "Source": market["zrodlo"],
                }
            )
        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )
    # -----------------------------------------------------
    # MODEL INFORMATION
    # -----------------------------------------------------
    st.divider()
    st.subheader("🧠 Model Information")
    info1, info2, info3 = st.columns(3)
    with info1:
        st.metric(
            "Home Historical Matches",
            prediction["n_home"],
        )
    with info2:
        st.metric(
            "Away Historical Matches",
            prediction["n_away"],
        )
    with info3:
        st.metric(
            "Model",
            "Bayesian Poisson",
        )
    st.caption(
        "Probabilities are generated by the FootStats "
        "Bayesian Poisson engine. Markets are derived "
        "from the resulting goal probability matrix."
    )