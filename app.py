import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# PROJECT PATH
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_SRC = PROJECT_ROOT / "src"

if LOCAL_SRC.exists():
    sys.path.insert(0, str(LOCAL_SRC))

# -----------------------------------------------------------------------------
# FOOTSTATS CORE
# -----------------------------------------------------------------------------

from footstats.config import DC_RHO_CLASSIC
from footstats.core.poisson import predict_match
from footstats.core.poisson_bayesian import predict_match_bayesian
from footstats.core.markets import build_market_catalog
from footstats.core.bet_builder import get_betbuilder_suggestions
from footstats.core.value_bet import calculate_ev, kelly_fraction
from footstats.core.confidence import komentarz_analityka
from footstats.core.h2h import AnalizaH2H
from footstats.core.fatigue import HeurystaZmeczeniaRotacji
from footstats.core.fortress import HomeFortress
from footstats.core.classifier import KlasyfikatorMeczu
from footstats.core.standings import table_asof, season_start_year
from footstats.core.importance import ImportanceIndex


# -----------------------------------------------------------------------------
# PAGE
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="FootStats Integrated Predictor",
    page_icon="â½",
    layout="wide",
)

st.title("â½ FootStats â Integrated Prediction Engine")
st.caption(
    "Full FootStats Poisson + Dixon-Coles + xG + form + H2H + fatigue + fortress "
    "+ markets + value analysis"
)


# -----------------------------------------------------------------------------
# DATA LOADER
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    candidates = [
        Path("data/hist_cache/full_dataset.parquet"),
        PROJECT_ROOT / "data/hist_cache/full_dataset.parquet",
    ]

    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "Historical dataset not found: data/hist_cache/full_dataset.parquet"
        )

    df = pd.read_parquet(path).copy()

    df = df.rename(
        columns={
            "home": "gospodarz",
            "away": "goscie",
            "hg": "gole_g",
            "ag": "gole_a",
        }
    )

    required = ["gospodarz", "goscie", "gole_g", "gole_a", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Historical dataset is missing required columns: "
            + ", ".join(missing)
        )

    df["gospodarz"] = df["gospodarz"].astype(str).str.strip()
    df["goscie"] = df["goscie"].astype(str).str.strip()
    df["gole_g"] = pd.to_numeric(df["gole_g"], errors="coerce")
    df["gole_a"] = pd.to_numeric(df["gole_a"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(
        subset=["gospodarz", "goscie", "gole_g", "gole_a", "date"]
    )
    df = df.sort_values("date").reset_index(drop=True)

    # Some FootStats modules use the older Polish date column.
    if "data" not in df.columns:
        df["data"] = df["date"]

    # Safe defaults for modules whose live datasets may contain these columns.
    if "stage" not in df.columns:
        df["stage"] = "REGULAR_SEASON"

    return df


try:
    df = load_data()
except Exception as exc:
    st.error("Could not load the FootStats historical dataset.")
    st.exception(exc)
    st.stop()


# -----------------------------------------------------------------------------
# SIDEBAR / MATCH SETUP
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("âï¸ Match Setup")

    if "league" in df.columns:
        leagues = sorted(
            df["league"].dropna().astype(str).unique().tolist()
        )
    else:
        leagues = ["All leagues"]

    league = st.selectbox("League", leagues)

    if league == "All leagues":
        league_df = df.copy()
    else:
        league_df = df[df["league"].astype(str) == league].copy()

    league_df = league_df.sort_values("date").reset_index(drop=True)

    teams = sorted(
        set(league_df["gospodarz"].astype(str))
        | set(league_df["goscie"].astype(str))
    )

    if not teams:
        st.error("No teams are available for the selected league.")
        st.stop()

    home_team = st.selectbox("Home Team", teams)

    away_options = [team for team in teams if team != home_team]
    if not away_options:
        st.error("There is no different away team available.")
        st.stop()

    away_team = st.selectbox("Away Team", away_options)

    prediction_date = pd.Timestamp(
        st.date_input(
            "Match Date",
            value=pd.Timestamp.today().date(),
        )
    )

    stage_options = [
        "REGULAR_SEASON",
        "ROUND_OF_16",
        "QUARTER_FINALS",
        "SEMI_FINALS",
        "FINAL",
        "PLAYOFFS",
    ]
    stage = st.selectbox("Match Stage", stage_options)

    use_xg = st.checkbox("Use Understat xG cache", value=True)
    use_calibration = st.checkbox("Use FootStats Î» calibration", value=True)

    st.divider()
    predict_button = st.button(
        "ð® RUN FULL PREDICTION",
        type="primary",
        use_container_width=True,
    )


# -----------------------------------------------------------------------------
# DATA STATUS
# -----------------------------------------------------------------------------

st.success(f"Historical database loaded: {len(df):,} matches")

if league != "All leagues":
    st.info(f"Selected league: {league} Â· {len(league_df):,} matches")


# -----------------------------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------------------------

if not predict_button:
    st.markdown(
        "### Ready\n"
        "Choose the league, teams, date and match stage in the sidebar, "
        "then press **RUN FULL PREDICTION**."
    )
    st.stop()


with st.spinner("Running the full FootStats engine..."):
    # Strict pre-match cutoff. No match on or after the prediction date is used.
    model_df = league_df[league_df["date"] < prediction_date].copy()
    model_df = model_df.sort_values("date").reset_index(drop=True)

    if model_df.empty:
        st.error("No historical matches are available before the prediction date.")
        st.stop()

    # -------------------------------------------------------------------------
    # CLASSIFIER
    # -------------------------------------------------------------------------

    classifier_df = model_df.copy()
    if "stage" not in classifier_df.columns:
        classifier_df["stage"] = "REGULAR_SEASON"

    classifier = KlasyfikatorMeczu(
        classifier_df,
        kod_ligi=str(league) if league != "All leagues" else "",
    )

    classification = classifier.klasyfikuj(
        home_team,
        away_team,
        stage,
        str(prediction_date.date()),
    )

    # -------------------------------------------------------------------------
    # H2H
    # -------------------------------------------------------------------------

    h2h_engine = AnalizaH2H(model_df)
    h2h_home = h2h_engine.analiza(
        home_team,
        away_team,
        str(prediction_date.date()),
    )
    h2h_away = h2h_engine.analiza(
        away_team,
        home_team,
        str(prediction_date.date()),
    )

    # -------------------------------------------------------------------------
    # FATIGUE / ROTATION
    # -------------------------------------------------------------------------

    fatigue_engine = HeurystaZmeczeniaRotacji(model_df)
    fatigue_home = fatigue_engine.analiza(
        home_team,
        str(prediction_date.date()),
    )
    fatigue_away = fatigue_engine.analiza(
        away_team,
        str(prediction_date.date()),
    )

    # -------------------------------------------------------------------------
    # HOME FORTRESS
    # -------------------------------------------------------------------------

    fortress_engine = HomeFortress(model_df)
    fortress_home = fortress_engine.analiza(home_team)

    # -------------------------------------------------------------------------
    # IMPORTANCE INDEX
    # -------------------------------------------------------------------------

    importance_home = None
    importance_away = None

    if "season" in model_df.columns and "league" in model_df.columns and league != "All leagues":
        try:
            season_values = model_df["season"].dropna().tolist()
            if season_values:
                latest_season = season_values[-1]
                standings = table_asof(
                    df,
                    league,
                    latest_season,
                    prediction_date,
                )
                if standings is not None and not standings.empty:
                    importance_engine = ImportanceIndex(
                        standings,
                        n_druzyn=len(standings),
                    )
                    importance_home = importance_engine.analiza(home_team)
                    importance_away = importance_engine.analiza(away_team)
        except Exception:
            # Importance is an optional enrichment. The main model remains intact.
            importance_home = None
            importance_away = None

    if importance_home is None:
        importance_home = {
            "bonus_atak": 1.0,
            "komentarz": "",
            "status": "NORMAL",
        }

    if importance_away is None:
        importance_away = {
            "bonus_atak": 1.0,
            "komentarz": "",
            "status": "NORMAL",
        }

    # -------------------------------------------------------------------------
    # FULL FOOTSTATS PREDICTION ENGINE
    # -------------------------------------------------------------------------

    prediction = predict_match(
        home_team,
        away_team,
        model_df,
        importance_g=importance_home,
        importance_a=importance_away,
        heurystyka_g=fatigue_home,
        heurystyka_a=fatigue_away,
        h2h_g=h2h_home,
        h2h_a=h2h_away,
        fortress_g=fortress_home,
        stage=stage,
        klasyfikacja=classification,
        use_xg=use_xg,
        use_calibration=use_calibration,
    )

    if prediction is None:
        st.error(
            "FootStats could not generate a prediction from the available history."
        )
        st.stop()

    # -------------------------------------------------------------------------
    # SECONDARY BAYESIAN MODEL FOR DIAGNOSTICS
    # -------------------------------------------------------------------------

    bayesian_prediction = None
    try:
        bayesian_prediction = predict_match_bayesian(
            home_team,
            away_team,
            model_df,
        )
    except Exception:
        bayesian_prediction = None

    # -------------------------------------------------------------------------
    # MARKETS â SAME DIXON-COLES RHO AS THE MAIN ENGINE
    # -------------------------------------------------------------------------

    markets = build_market_catalog(
        prediction["lambda_g"],
        prediction["lambda_a"],
        rho=DC_RHO_CLASSIC,
    )

    # -------------------------------------------------------------------------
    # BET BUILDER SUGGESTIONS
    # -------------------------------------------------------------------------

    try:
        betbuilder_suggestions = get_betbuilder_suggestions(
            prediction["lambda_g"],
            prediction["lambda_a"],
        )
    except Exception:
        betbuilder_suggestions = []


# -----------------------------------------------------------------------------
# TOP SUMMARY
# -----------------------------------------------------------------------------

st.header(f"{home_team} vs {away_team}")

if classification.get("etykieta_plain"):
    st.caption(classification["etykieta_plain"])

summary = st.columns(5)

with summary[0]:
    st.metric("Home Î»", f"{prediction['lambda_g']:.2f}")

with summary[1]:
    st.metric("Away Î»", f"{prediction['lambda_a']:.2f}")

with summary[2]:
    st.metric(
        "Most Likely Score",
        f"{prediction['wynik_g']}â{prediction['wynik_a']}",
    )

with summary[3]:
    st.metric("BTTS", f"{prediction['btts']:.1f}%")

with summary[4]:
    st.metric("Over 2.5", f"{prediction['over25']:.1f}%")


# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------

tab_prediction, tab_markets, tab_value, tab_analysis, tab_diagnostics = st.tabs(
    [
        "ð¯ Prediction",
        "ð Markets",
        "ð° Value / Kelly",
        "ð§  Match Analysis",
        "ð¬ Diagnostics",
    ]
)


# -----------------------------------------------------------------------------
# PREDICTION TAB
# -----------------------------------------------------------------------------

with tab_prediction:
    st.subheader("1X2")

    c1, c2, c3 = st.columns(3)

    with c1:
        p = prediction["p_wygrana"]
        st.metric(
            "1 â Home",
            f"{p:.1f}%",
            f"Fair {100 / p:.2f}" if p > 0 else None,
        )

    with c2:
        p = prediction["p_remis"]
        st.metric(
            "X â Draw",
            f"{p:.1f}%",
            f"Fair {100 / p:.2f}" if p > 0 else None,
        )

    with c3:
        p = prediction["p_przegrana"]
        st.metric(
            "2 â Away",
            f"{p:.1f}%",
            f"Fair {100 / p:.2f}" if p > 0 else None,
        )

    st.subheader("Top 5 exact scores")

    score_rows = []
    for score, probability in prediction.get("top5", []):
        score_rows.append(
            {
                "Score": score.replace(":", "â"),
                "Probability": f"{probability:.1f}%",
            }
        )

    if score_rows:
        st.dataframe(
            pd.DataFrame(score_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Model confidence")
    st.progress(
        max(0.0, min(float(prediction.get("pewnosc", 0)) / 100.0, 1.0))
    )
    st.write(f"Confidence: **{prediction.get('pewnosc', 0)}%**")

    st.info(komentarz_analityka(prediction))


# -----------------------------------------------------------------------------
# MARKETS TAB
# -----------------------------------------------------------------------------

with tab_markets:
    st.subheader("Full FootStats Market Catalog")
    st.caption(
        f"Market probabilities are generated from Î» using DC rho = {DC_RHO_CLASSIC:.4f}."
    )

    for group in markets:
        st.markdown(f"### {group['grupa']}")

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

    st.divider()
    st.subheader("BetBuilder suggestions")

    if betbuilder_suggestions:
        for suggestion in betbuilder_suggestions:
            st.write(f"â¢ {suggestion}")
    else:
        st.info("The existing BetBuilder module returned no qualifying suggestions.")


# -----------------------------------------------------------------------------
# VALUE / KELLY TAB
# -----------------------------------------------------------------------------

with tab_value:
    st.subheader("Bookmaker Odds â EV â Kelly")
    st.caption(
        "Enter real bookmaker odds. EV and Kelly are calculated by the existing FootStats value module."
    )

    odds_c1, odds_c2, odds_c3 = st.columns(3)

    with odds_c1:
        home_odds = st.number_input("Home (1) odds", min_value=1.01, value=2.00, step=0.01)

    with odds_c2:
        draw_odds = st.number_input("Draw (X) odds", min_value=1.01, value=3.50, step=0.01)

    with odds_c3:
        away_odds = st.number_input("Away (2) odds", min_value=1.01, value=3.50, step=0.01)

    odds_rows = [
        ("1 â Home", prediction["p_wygrana"] / 100.0, home_odds),
        ("X â Draw", prediction["p_remis"] / 100.0, draw_odds),
        ("2 â Away", prediction["p_przegrana"] / 100.0, away_odds),
    ]

    value_rows = []
    for name, probability, odds in odds_rows:
        ev = calculate_ev(probability, float(odds))
        kelly = kelly_fraction(probability, float(odds)) * 100.0
        value_rows.append(
            {
                "Market": name,
                "Model Probability": f"{probability * 100:.1f}%",
                "Odds": f"{odds:.2f}",
                "Fair Odds": f"{1 / probability:.2f}" if probability > 0 else "â",
                "EV": f"{ev:+.2f}%",
                "Kelly": f"{kelly:.2f}%",
                "Value": "YES" if ev >= 3.0 and kelly >= 1.0 else "NO",
            }
        )

    st.dataframe(
        pd.DataFrame(value_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "Kelly is a mathematical sizing output, not a guarantee of profit. "
        "Use conservative fractional Kelly if you actually stake money."
    )


# -----------------------------------------------------------------------------
# MATCH ANALYSIS TAB
# -----------------------------------------------------------------------------

with tab_analysis:
    st.subheader("Historical / Context Analysis")

    a1, a2, a3 = st.columns(3)

    with a1:
        st.metric("Home historical matches", int(
            ((model_df["gospodarz"] == home_team) | (model_df["goscie"] == home_team)).sum()
        ))

    with a2:
        st.metric("Away historical matches", int(
            ((model_df["gospodarz"] == away_team) | (model_df["goscie"] == away_team)).sum()
        ))

    with a3:
        st.metric("H2H matches, 24 months", int(h2h_home.get("n_h2h", 0)))

    st.subheader("Form / team strength")

    strength_rows = [
        {
            "Team": home_team,
            "Attack": prediction["sila_at_g"],
            "Defense": prediction["sila_ob_g"],
            "Form points/match": prediction["forma_g"],
        },
        {
            "Team": away_team,
            "Attack": prediction["sila_at_a"],
            "Defense": prediction["sila_ob_a"],
            "Form points/match": prediction["forma_a"],
        },
    ]

    st.dataframe(
        pd.DataFrame(strength_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("H2H")
    h2h_c1, h2h_c2 = st.columns(2)

    with h2h_c1:
        st.write(f"**{home_team}**")
        st.write(f"Patent: {bool(h2h_home.get('patent'))}")
        st.write(f"Zemsta: {bool(h2h_home.get('zemsta'))}")
        st.write(f"H2H confidence: {h2h_home.get('pewnosc', 20)}%")
        if h2h_home.get("opis"):
            st.caption(h2h_home["opis"])

    with h2h_c2:
        st.write(f"**{away_team}**")
        st.write(f"Patent: {bool(h2h_away.get('patent'))}")
        st.write(f"Zemsta: {bool(h2h_away.get('zemsta'))}")
        st.write(f"H2H confidence: {h2h_away.get('pewnosc', 20)}%")
        if h2h_away.get("opis"):
            st.caption(h2h_away["opis"])

    st.subheader("Fatigue / rotation")
    fatigue_rows = [
        {
            "Team": home_team,
            "Rotation": bool(fatigue_home.get("rotacja")),
            "Fatigue": bool(fatigue_home.get("zmeczenie")),
            "Attack multiplier": fatigue_home.get("mnoznik_atak", 1.0),
            "Defense multiplier": fatigue_home.get("mnoznik_obr", 1.0),
        },
        {
            "Team": away_team,
            "Rotation": bool(fatigue_away.get("rotacja")),
            "Fatigue": bool(fatigue_away.get("zmeczenie")),
            "Attack multiplier": fatigue_away.get("mnoznik_atak", 1.0),
            "Defense multiplier": fatigue_away.get("mnoznik_obr", 1.0),
        },
    ]
    st.dataframe(
        pd.DataFrame(fatigue_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Home fortress")
    st.write(
        f"{home_team}: "
        f"{'ACTIVE' if fortress_home.get('fortress') else 'inactive'} Â· "
        f"unbeaten home streak = {fortress_home.get('seria', 0)}"
    )
    if fortress_home.get("opis"):
        st.info(fortress_home["opis"])

    if classification.get("opis"):
        st.subheader("Match classification")
        st.info(classification["opis"])

    h2h_df = h2h_home.get("h2h_df")
    if isinstance(h2h_df, pd.DataFrame) and not h2h_df.empty:
        st.subheader("Recent H2H results")
        display_cols = [
            c for c in ["date", "data", "gospodarz", "goscie", "gole_g", "gole_a"]
            if c in h2h_df.columns
        ]
        if display_cols:
            st.dataframe(
                h2h_df[display_cols].sort_values(
                    display_cols[0], ascending=False
                ),
                use_container_width=True,
                hide_index=True,
            )


# -----------------------------------------------------------------------------
# DIAGNOSTICS TAB
# -----------------------------------------------------------------------------

with tab_diagnostics:
    st.subheader("Model diagnostics")

    diag_rows = [
        {"Component": "Main engine", "Status": "ACTIVE", "Details": "core.poisson.predict_match"},
        {"Component": "Dixon-Coles", "Status": "ACTIVE", "Details": f"rho={DC_RHO_CLASSIC:.4f}"},
        {"Component": "Î» calibration", "Status": "ON" if use_calibration else "OFF", "Details": "lambda_optimizer"},
        {"Component": "Understat xG cache", "Status": "ON" if use_xg else "OFF", "Details": "cache-only; no live request"},
        {"Component": "H2H", "Status": "ACTIVE", "Details": f"{h2h_home.get('n_h2h', 0)} recent matches"},
        {"Component": "Fatigue/rotation", "Status": "ACTIVE", "Details": "pre-match history only"},
        {"Component": "Home fortress", "Status": "ACTIVE", "Details": f"streak={fortress_home.get('seria', 0)}"},
        {"Component": "Importance", "Status": "ACTIVE" if prediction.get("imp_g", {}).get("status") != "NORMAL" or prediction.get("imp_a", {}).get("status") != "NORMAL" else "NORMAL", "Details": "standings-as-of-date when season data exists"},
        {"Component": "Market catalog", "Status": "ACTIVE", "Details": f"{sum(len(g['rynki']) for g in markets)} markets"},
        {"Component": "Value / Kelly", "Status": "ACTIVE", "Details": "core.value_bet"},
    ]

    st.dataframe(
        pd.DataFrame(diag_rows),
        use_container_width=True,
        hide_index=True,
    )

    if bayesian_prediction:
        st.subheader("Secondary Bayesian model")
        comparison_rows = [
            {
                "Metric": "Home Î»",
                "Full FootStats": prediction["lambda_g"],
                "Bayesian": bayesian_prediction["lambda_g"],
            },
            {
                "Metric": "Away Î»",
                "Full FootStats": prediction["lambda_a"],
                "Bayesian": bayesian_prediction["lambda_a"],
            },
            {
                "Metric": "Home win",
                "Full FootStats": prediction["p_wygrana"],
                "Bayesian": bayesian_prediction["pw"] * 100,
            },
            {
                "Metric": "Draw",
                "Full FootStats": prediction["p_remis"],
                "Bayesian": bayesian_prediction["pr"] * 100,
            },
            {
                "Metric": "Away win",
                "Full FootStats": prediction["p_przegrana"],
                "Bayesian": bayesian_prediction["pa"] * 100,
            },
        ]
        st.dataframe(
            pd.DataFrame(comparison_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("No-lookahead information")
    st.write(
        f"Prediction date: **{prediction_date.date()}**  |  "
        f"Latest eligible historical match: **{model_df['date'].max().date()}**  |  "
        f"Historical matches supplied to engine: **{len(model_df):,}**"
    )

    st.warning(
        "Important: this dashboard exposes the existing FootStats modules; it does not "
        "invent bookmaker odds, injuries, lineups, referee data or Bzzoiro probabilities "
        "when those inputs are not present in the selected historical dataset."
    )
