import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from data_loader import load_batting_data, label_match_results
from feature_engineering import (
    add_batting_features,
    compute_quality_scores,
    label_innings_quality
)


def main():

    # -------------------------
    # Load and prepare data
    # -------------------------

    df = load_batting_data()
    df = label_match_results(df)

    df = add_batting_features(df)
    df = compute_quality_scores(df)
    df = label_innings_quality(df)

    # Remove draws for binary win modelling
    df = df[df["result"] != "draw"]

    # -------------------------
    # Poor innings per team
    # -------------------------

    team_poor = (
        df.groupby(["match_id", "batting_team_id"])
          .agg(poor_innings_count=("poor_innings", "sum"))
          .reset_index()
    )

    winner_map = (
        df[["match_id", "winner_team_id"]]
        .drop_duplicates()
    )

    team_poor = team_poor.merge(winner_map, on="match_id", how="left")

    # -------------------------
    # Pivot to match-level
    # -------------------------

    pivot = team_poor.pivot(
        index="match_id",
        columns="batting_team_id",
        values="poor_innings_count"
    )

    pivot = pivot.merge(
        winner_map.set_index("match_id"),
        left_index=True,
        right_index=True,
        how="left"
    )

    team_cols = [c for c in pivot.columns if str(c).isdigit()]
    pivot["winner_team_id"] = pivot["winner_team_id"].astype(str)

    pivot["winner_poor_count"] = pivot.apply(
        lambda r: r[r["winner_team_id"]],
        axis=1
    )

    pivot["loser_poor_count"] = pivot.apply(
        lambda r: r[team_cols].drop(r["winner_team_id"]).sum(),
        axis=1
    )

    pivot["poor_gap"] = pivot["loser_poor_count"] - pivot["winner_poor_count"]


    # -------------------------
    # Symmetric modelling data
    # -------------------------

    winners = pivot[
    pivot["winner_team_id"].notna() &
    pivot["winner_poor_count"].notna() &
    pivot["loser_poor_count"].notna()
    ].copy()

    winners["win"] = 1

    losers = winners.copy()
    losers["win"] = 0
    losers["poor_gap"] = -losers["poor_gap"]

    model_df = pd.concat([winners, losers], ignore_index=True)

    # -------------------------
    # Logistic regression
    # -------------------------

    X = sm.add_constant(model_df[["poor_gap"]])
    y = model_df["win"]

    logit = sm.Logit(y, X).fit()
    print(logit.summary())


    model_df["prob_win"] = logit.predict(X)


    # -------------------------
    # Visualisation
    # -------------------------

    (
        model_df
        .groupby("poor_gap")["prob_win"]
        .mean()
        .plot(
            title="Win Probability vs Poor Innings Gap",
            xlabel="Poor Innings Advantage",
            ylabel="Win Probability"
        )
    )

    plt.show()


if __name__ == "__main__":
    main()
