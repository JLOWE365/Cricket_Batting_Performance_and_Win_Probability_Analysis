import sqlite3
import pandas as pd
import numpy as np


def load_batting_data(db_path="ashes.db"):
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query("""
    SELECT
        m.match_id,
        m.winner_team_id,
        b.batting_team_id,
        CAST(b.strike_rate AS FLOAT) AS strike_rate,
        b.player_id,
        CAST(b.runs AS INT) AS runs,
        CAST(b.balls_faced AS INT) AS balls_faced,
        b.innings,
        b.is_out
    FROM batting_scorecard b
    JOIN matches m ON b.match_id = m.match_id;
    """, conn)

    conn.close()

    # Basic typing
    df["winner_team_id"] = df["winner_team_id"].astype(int)
    df["innings"] = df["innings"].astype(int)
    df["dismissed"] = df["is_out"].astype(int)

    return df


def label_match_results(df):
    df = df.copy()

    df["result"] = np.where(
        df["batting_team_id"] == df["winner_team_id"],
        "win",
        "loss"
    )

    # Identify drawn matches (winner_team_id == 0)
    draw_matches = (
        df.groupby("match_id")["winner_team_id"]
          .unique()
          .apply(lambda x: 0 in x)
    )

    draw_ids = draw_matches[draw_matches].index
    df.loc[df["match_id"].isin(draw_ids), "result"] = "draw"

    return df
