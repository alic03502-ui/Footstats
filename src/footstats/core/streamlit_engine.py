"""
Streamlit integration engine.

Connects the Streamlit application to the
existing FootStats prediction engine.
"""

from __future__ import annotations

from typing import Any

from footstats.core.poisson import predict_match


def analyze_match(
    home_team: str,
    away_team: str,
    league_df,
    prediction_date,
) -> dict[str, Any]:
    """
    Main entry point for Streamlit.

    Uses only historical matches before the prediction date
    and sends them through the full FootStats prediction engine.
    """

    if league_df is None:
        raise ValueError(
            "Historical league data is required."
        )

    # -------------------------------------------------
    # PREVENT FUTURE-DATA LEAKAGE
    # -------------------------------------------------

    model_df = league_df[
        league_df["date"] < prediction_date
    ].copy()

    if model_df.empty:
        raise ValueError(
            "No historical matches available before "
            "the prediction date."
        )

    # -------------------------------------------------
    # KEEP HISTORY CHRONOLOGICAL
    # -------------------------------------------------

    model_df = (
        model_df
        .sort_values("date")
        .reset_index(drop=True)
    )

    # -------------------------------------------------
    # FULL FOOTSTATS PREDICTION ENGINE
    # -------------------------------------------------

    prediction = predict_match(
        home_team,
        away_team,
        model_df,
        use_xg=True,
        use_calibration=True,
    )

    # -------------------------------------------------
    # VALIDATION
    # -------------------------------------------------

    if prediction is None:
        raise ValueError(
            "The FootStats prediction engine could not "
            "generate a prediction from the available history."
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
        "home_probability": prediction["p_wygrana"],
        "draw_probability": prediction["p_remis"],
        "away_probability": prediction["p_przegrana"],
        "home_matches_used": prediction.get(
            "sila_at_g",
            0,
        ),
        "away_matches_used": prediction.get(
            "sila_at_a",
            0,
        ),
        "model": "FootStats Full Poisson + Dixon-Coles",
    }    }    