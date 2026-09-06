import streamlit as st
import math
st.set_page_config(
    page_title="FootStats Test",
    page_icon="⚽",
    layout="wide",
)
st.title("⚽ FootStats — Prediction Dashboard")
st.caption("Market interface test — calculations will be connected to the FootStats engine next.")
# ---------------------------------------------------------
# MATCH INPUT
# ---------------------------------------------------------
st.header("Match")
col1, col2 = st.columns(2)
with col1:
    home_team = st.text_input("Home Team", "Arsenal")
with col2:
    away_team = st.text_input("Away Team", "Chelsea")
st.divider()
# ---------------------------------------------------------
# DEMO MODEL PARAMETERS
# ---------------------------------------------------------
st.header("Model Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    home_xg = st.number_input(
        "Home Expected Goals",
        min_value=0.1,
        max_value=6.0,
        value=1.65,
        step=0.05,
    )
with col2:
    away_xg = st.number_input(
        "Away Expected Goals",
        min_value=0.1,
        max_value=6.0,
        value=1.15,
        step=0.05,
    )
with col3:
    max_goals = st.number_input(
        "Maximum Goals",
        min_value=5,
        max_value=12,
        value=8,
        step=1,
    )
# ---------------------------------------------------------
# POISSON
# ---------------------------------------------------------
def poisson_probability(goals, expected_goals):
    return (
        math.exp(-expected_goals)
        * expected_goals ** goals
        / math.factorial(goals)
    )
home_probs = [
    poisson_probability(i, home_xg)
    for i in range(max_goals + 1)
]
away_probs = [
    poisson_probability(i, away_xg)
    for i in range(max_goals + 1)
]
# Score matrix
score_matrix = {}
for h in range(max_goals + 1):
    for a in range(max_goals + 1):
        score_matrix[(h, a)] = home_probs[h] * away_probs[a]
# ---------------------------------------------------------
# 1X2
# ---------------------------------------------------------
home_win = sum(
    p for (h, a), p in score_matrix.items()
    if h > a
)
draw = sum(
    p for (h, a), p in score_matrix.items()
    if h == a
)
away_win = sum(
    p for (h, a), p in score_matrix.items()
    if h < a
)
# ---------------------------------------------------------
# TOTAL GOALS
# ---------------------------------------------------------
def total_goals_probability(line, over=True):
    probability = 0
    for (h, a), p in score_matrix.items():
        total = h + a
        if over:
            if total > line:
                probability += p
        else:
            if total < line:
                probability += p
    return probability
# ---------------------------------------------------------
# BTTS
# ---------------------------------------------------------
btts_yes = sum(
    p for (h, a), p in score_matrix.items()
    if h >= 1 and a >= 1
)
btts_no = 1 - btts_yes
# ---------------------------------------------------------
# DOUBLE CHANCE
# ---------------------------------------------------------
double_1x = home_win + draw
double_x2 = draw + away_win
double_12 = home_win + away_win
# ---------------------------------------------------------
# TEAM GOALS
# ---------------------------------------------------------
def team_goals_probability(team, line, over=True):
    probability = 0
    for (h, a), p in score_matrix.items():
        goals = h if team == "home" else a
        if over and goals > line:
            probability += p
        elif not over and goals < line:
            probability += p
    return probability
# ---------------------------------------------------------
# CLEAN SHEETS
# ---------------------------------------------------------
home_clean_sheet = sum(
    p for (h, a), p in score_matrix.items()
    if a == 0
)
away_clean_sheet = sum(
    p for (h, a), p in score_matrix.items()
    if h == 0
)
# ---------------------------------------------------------
# FAIR ODDS
# ---------------------------------------------------------
def fair_odds(probability):
    if probability <= 0:
        return float("inf")
    return 1 / probability
def market_row(name, probability):
    odds = fair_odds(probability)
    return {
        "Market": name,
        "Probability": f"{probability * 100:.2f}%",
        "Fair Odds": f"{odds:.2f}",
    }
# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------
st.divider()
st.header("📊 Prediction Markets")
# ---------------------------------------------------------
# 1X2
# ---------------------------------------------------------
st.subheader("1X2")
data_1x2 = [
    market_row(f"{home_team} Win", home_win),
    market_row("Draw", draw),
    market_row(f"{away_team} Win", away_win),
]
st.table(data_1x2)
# ---------------------------------------------------------
# DOUBLE CHANCE
# ---------------------------------------------------------
st.subheader("Double Chance")
data_dc = [
    market_row("1X — Home or Draw", double_1x),
    market_row("X2 — Draw or Away", double_x2),
    market_row("12 — Home or Away", double_12),
]
st.table(data_dc)
# ---------------------------------------------------------
# OVER / UNDER
# ---------------------------------------------------------
st.subheader("⚽ Total Goals — Over / Under")
ou_data = []
for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
    over_probability = total_goals_probability(line, True)
    under_probability = total_goals_probability(line, False)
    ou_data.append(
        market_row(f"Over {line}", over_probability)
    )
    ou_data.append(
        market_row(f"Under {line}", under_probability)
    )
st.table(ou_data)
# ---------------------------------------------------------
# BTTS
# ---------------------------------------------------------
st.subheader("Both Teams To Score")
btts_data = [
    market_row("BTTS — Yes", btts_yes),
    market_row("BTTS — No", btts_no),
]
st.table(btts_data)
# ---------------------------------------------------------
# TEAM GOALS
# ---------------------------------------------------------
st.subheader("Team Goals")
team_goal_data = []
for team_name, team_key in [
    (home_team, "home"),
    (away_team, "away"),
]:
    for line in [0.5, 1.5, 2.5]:
        over_probability = team_goals_probability(
            team_key,
            line,
            True,
        )
        under_probability = team_goals_probability(
            team_key,
            line,
            False,
        )
        team_goal_data.append(
            market_row(
                f"{team_name} Over {line}",
                over_probability,
            )
        )
        team_goal_data.append(
            market_row(
                f"{team_name} Under {line}",
                under_probability,
            )
        )
st.table(team_goal_data)
# ---------------------------------------------------------
# CLEAN SHEETS
# ---------------------------------------------------------
st.subheader("Clean Sheet")
clean_sheet_data = [
    market_row(
        f"{home_team} Clean Sheet",
        home_clean_sheet,
    ),
    market_row(
        f"{away_team} Clean Sheet",
        away_clean_sheet,
    ),
]
st.table(clean_sheet_data)
# ---------------------------------------------------------
# MOST LIKELY SCORES
# ---------------------------------------------------------
st.subheader("🎯 Most Likely Scores")
sorted_scores = sorted(
    score_matrix.items(),
    key=lambda x: x[1],
    reverse=True,
)
score_data = []
for (h, a), probability in sorted_scores[:10]:
    score_data.append(
        {
            "Score": f"{h}-{a}",
            "Probability": f"{probability * 100:.2f}%",
            "Fair Odds": f"{fair_odds(probability):.2f}",
        }
    )
st.table(score_data)
# ---------------------------------------------------------
# MODEL SUMMARY
# ---------------------------------------------------------
st.divider()
st.subheader("Model Summary")
summary_col1, summary_col2, summary_col3 = st.columns(3)
with summary_col1:
    st.metric(
        "Expected Goals",
        f"{home_xg + away_xg:.2f}",
    )
with summary_col2:
    st.metric(
        "Home Win",
        f"{home_win * 100:.1f}%",
    )
with summary_col3:
    st.metric(
        "Away Win",
        f"{away_win * 100:.1f}%",
    )
st.info(
    "⚠️ This is currently a Poisson demonstration. "
    "The next step is connecting these markets to the actual "
    "FootStats prediction engine and historical data."
)