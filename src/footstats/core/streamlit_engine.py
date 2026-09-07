"""
Streamlit integration engine.

Connects the existing Footstats prediction engine
to the Streamlit application.
"""

from __future__ import annotations

from typing import Any

from footstats.core.poisson_bayesian import predict_match_bayesian


def analyze_match(
    home_team: str,
    away_team: str,
    league_df,
    prediction_date,
) -> dict[str, Any]:
    """
    Main entry point for Streamlit.

    Uses only historical matches before the prediction date.
    """

    if league_df is None:
        raise ValueError("Historical league data is required.")

    # -------------------------------------------------
    # PREVENT FUTURE-DATA LEAKAGE
    # -------------------------------------------------

    model_df = league_df[
        league_df["date"] < prediction_date
    ].copy()

    if model_df.empty:
        raise ValueError(
            "No historical matches available before the prediction date."
        )

    # -------------------------------------------------
    # EXISTING FOOTSTATS BAYESIAN MODEL
    # -------------------------------------------------

    prediction = predict_match_bayesian(
        home_team,
        away_team,
        model_df,
    )

    # -------------------------------------------------
    # RETURN ONE CLEAN OBJECT TO STREAMLIT
    # -------------------------------------------------

    return {
        "home_team": home_team,
        "away_team": away_team,
        "prediction_date": prediction_date,

        "historical_matches": len(model_df),

        "prediction": prediction,

        "lambda_home": prediction["lambda_g"],
        "lambda_away": prediction["lambda_a"],

        "home_probability": prediction["pw"],
        "draw_probability": prediction["pr"],
        "away_probability": prediction["pa"],

        "home_matches_used": prediction["n_home"],
        "away_matches_used": prediction["n_away"],

        "model": prediction.get(
            "model",
            "Bayesian Poisson",
        ),
    }    }