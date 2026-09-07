import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import pandas as pd

from footstats.core.streamlit_engine import analyze_match
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

    # Historical dataset → FootStats model schema
    df = df.rename(
        columns={
            "home": "gospodarz",
            "away": "goscie",
            "hg": "gole_g",
            "ag": "gole_a",
        }
    )

    # Goals
    df["gole_g"] = pd.to_numeric(
        df["gole_g"],
        errors="coerce",
    )

    df["gole_a"] = pd.to_numeric(
        df["gole_a"],
        errors="coerce",
    )

    # Dates
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    # Remove invalid rows
    df = df.dropna(
        subset=[
            "gospodarz",
            "goscie",
            "gole_g",
            "gole_a",
            "date",
        ]
    )

    # Important: keep historical data chronological
    df = df.sort_values("date").reset_index(drop=True)

    return df


# ---------------------------------------------------------
# LOAD DATA SAFELY
# ---------------------------------------------------------

try:

    df = load_data()

except Exception as e:

    st.error(
        "Could not load the FootStats historical dataset."
    )

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


# ---------------------------------------------------------
# FILTER LEAGUE
# ---------------------------------------------------------

if league != "All leagues":

    league_df = df[
        df["league"].astype(str) == league
    ].copy()

else:

    league_df = df.copy()


# Keep filtered data chronological
league_df = (
    league_df
    .sort_values("date")
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# TEAM SELECTION
# ---------------------------------------------------------

teams = sorted(
    set(
        league_df["gospodarz"]
        .dropna()
        .astype(str)
    ).union(
        set(
            league_df["goscie"]
            .dropna()
            .astype(str)
        )
    )
)


if not teams:

    st.error("No teams are available for the selected league.")
    st.stop()


col1, col2 = st.columns(2)


with col1:

    home_team = st.selectbox(
        "Home Team",
        teams,
        index=0,
    )


with col2:

    away_options = [
        team
        for team in teams
        if team != home_team
    ]

    if not away_options:

        st.error(
            "There are not enough teams available "
            "to select a different away team."
        )

        st.stop()

    away_team = st.selectbox(
        "Away Team",
        away_options,
        index=0,
    )


# ---------------------------------------------------------
# MATCH DATE
# ---------------------------------------------------------

prediction_date = st.date_input(
    "Match Date",
    value=pd.Timestamp.today().date(),
)


# ---------------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------------

predict_button = st.button(
    "🔮 Predict Match",
    type="primary",
    use_container_width=True,
)


# =========================================================
# RUN PREDICTION
# =========================================================

if predict_button:

    with st.spinner("Running FootStats model..."):

        # -------------------------------------------------
        # DATE
        # -------------------------------------------------

        prediction_date = pd.Timestamp(
            prediction_date
        )

        # -------------------------------------------------
        # HISTORICAL DATA CUTOFF
        # -------------------------------------------------

        model_df = league_df[
            league_df["date"] < prediction_date
        ].copy()

        model_df = (
            model_df
            .sort_values("date")
            .reset_index(drop=True)
        )

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        if model_df.empty:

            st.error(
                "No historical matches are available "
                "before the selected prediction date."
            )

            st.stop()

        # -------------------------------------------------
        # TEAM HISTORY
        # -------------------------------------------------

        home_history = model_df[
            (
                model_df["gospodarz"] == home_team
            )
            |
            (
                model_df["goscie"] == home_team
            )
        ].copy()

        away_history = model_df[
            (
                model_df["gospodarz"] == away_team
            )
            |
            (
                model_df["goscie"] == away_team
            )
        ].copy()

        # =================================================
        # UNIFIED FOOTSTATS ENGINE
        # =================================================

        engine_result = analyze_match(
            home_team=home_team,
            away_team=away_team,
            league_df=league_df,
            prediction_date=prediction_date,
        )

        # Extract prediction
        prediction = engine_result["prediction"]

        # -------------------------------------------------
        # PREDICTION VALIDATION
        # -------------------------------------------------

        if prediction is None:

            st.error(
                "The FootStats engine could not "
                "generate a prediction."
            )

            st.stop()

        # =================================================
        # HISTORICAL DATA INFORMATION
        # =================================================

        st.subheader("📚 Historical Data Used")

        h1, h2 = st.columns(2)

        with h1:

            st.metric(
                "Home Team Historical Matches",
                len(home_history),
            )

        with h2:

            st.metric(
                "Away Team Historical Matches",
                len(away_history),
            )

        d1, d2 = st.columns(2)

        with d1:

            if not home_history.empty:

                st.caption(
                    f"{home_team}: "
                    f"{home_history['date'].min().date()} "
                    f"→ "
                    f"{home_history['date'].max().date()}"
                )

        with d2:

            if not away_history.empty:

                st.caption(
                    f"{away_team}: "
                    f"{away_history['date'].min().date()} "
                    f"→ "
                    f"{away_history['date'].max().date()}"
                )

        st.info(
            f"Prediction date: {prediction_date.date()}  |  "
            f"League historical matches used: "
            f"{len(model_df):,}  |  "
            f"Latest eligible match: "
            f"{model_df['date'].max().date()}"
        )

        # =================================================
        # MAIN MODEL OUTPUT
        # =================================================

        st.divider()

        st.header(
            f"{home_team} vs {away_team}"
        )

        # =================================================
        # EXPECTED GOALS
        # =================================================

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

            total_xg = (
                prediction["lambda_g"]
                + prediction["lambda_a"]
            )

            st.metric(
                "Expected Total Goals",
                f"{total_xg:.2f}",
            )

        # =================================================
        # 1X2
        # =================================================

        st.subheader("🎯 1X2")

        c1, c2, c3 = st.columns(3)

        with c1:

            p = prediction["pw"]

            st.metric(
                "1 — Home",
                f"{p * 100:.1f}%",
                (
                    f"Fair {1 / p:.2f}"
                    if p > 0
                    else None
                ),
            )

        with c2:

            p = prediction["pr"]

            st.metric(
                "X — Draw",
                f"{p * 100:.1f}%",
                (
                    f"Fair {1 / p:.2f}"
                    if p > 0
                    else None
                ),
            )

        with c3:

            p = prediction["pa"]

            st.metric(
                "2 — Away",
                f"{p * 100:.1f}%",
                (
                    f"Fair {1 / p:.2f}"
                    if p > 0
                    else None
                ),
            )

        # =================================================
        # FULL MARKET CATALOG
        # =================================================

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
                        "Probability": (
                            f"{market['szansa']:.1f}%"
                        ),
                        "Fair Odds": (
                            f"{market['kurs']:.2f}"
                        ),
                        "Source": market["zrodlo"],
                    }
                )

            if rows:

                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )