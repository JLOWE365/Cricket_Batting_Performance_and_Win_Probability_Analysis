import numpy as np
import pandas as pd


def add_batting_features(df):
    df = df.copy()

    df["run_rate"] = (df["runs"] / df["balls_faced"]) * 6
    df["effective_sr"] = df["strike_rate"] * np.log1p(df["balls_faced"])

    return df


def compute_quality_scores(df):
    df = df.copy()

    # Raw quality score
    df["quality_raw"] = (
        5 * df["runs"] +
        0.5 * df["effective_sr"] -
        15 * df["dismissed"]
    )

    # Match-level normalisation
    match_context = (
        df.groupby("match_id")["quality_raw"]
          .agg(match_mean="mean", match_std="std")
          .reset_index()
    )

    df = df.merge(match_context, on="match_id", how="left")

    df["quality_z"] = (
        (df["quality_raw"] - df["match_mean"]) /
        df["match_std"]
    )

    df.loc[df["match_std"] == 0, "quality_z"] = 0

    return df


def label_innings_quality(df):
    df = df.copy()

    df["great_innings"] = df["quality_z"] >= 2
    df["good_innings"] = df["quality_z"] >= 1.5
    df["decent_innings"] = df["quality_z"] >= 1
    df["poor_innings"] = df["quality_z"] < 0

    return df
