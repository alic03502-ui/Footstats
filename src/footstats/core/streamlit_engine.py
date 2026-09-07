"""
Streamlit integration engine.

This module connects the existing Footstats prediction components
into one clean interface for the Streamlit application.
"""

from __future__ import annotations

from typing import Any


def analyze_match(
    home_team: str,
    away_team: str,
    league_df,
    prediction_date,
) -> dict[str, Any]:
    """
    Main entry point for Streamlit.

    All existing Footstats model components will be connected here
    incrementally. The current working prediction remains the baseline.
    """

    if league_df is None:
        raise ValueError("Historical league data is required.")

    # Never allow future matches to enter the prediction.
    model_df = league_df[
        league_df["date"] < prediction_date
    ].copy()

    if model_df.empty:
        raise ValueError(
            "No historical matches available before the prediction date."
        )

    return {
        "home_team": home_team,
        "away_team": away_team,
        "prediction_date": prediction_date,
        "historical_matches": len(model_df),
        "status": "engine_ready",
    }