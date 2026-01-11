Cricket Batting Performance & Win Probability Analysis
Project Overview

This project explores how individual batting performances translate into team success in Test cricket, with a particular focus on whether poor batting performances exhibit diminishing returns at the team level.

Rather than relying on raw aggregates such as total runs, the analysis introduces a context-aware batting quality metric, normalised within each match, and examines how the accumulation of poor innings affects a team’s probability of winning.

The project combines SQL-based data extraction, feature engineering in Python, statistical modelling, and data visualisation.

Data:

Match, innings, and batting scorecard data stored in a SQLite database
Source data derived from structured cricket scorecards (Ashes series, later extended with additional Test series to increase sample size)

Key tables:

matches
batting_scorecard
players

Methodology
1. Data Extraction (SQL)

Used SQL queries (JOIN, GROUP BY) to combine match-level and player-level batting data.
Loaded query results directly into pandas DataFrames using pd.read_sql_query.

2. Feature Engineering

For each batting innings, engineered performance features including:
Runs scored
Strike rate and effective strike rate (scaled by balls faced)
Dismissal indicator
Innings number (to distinguish later-innings context)
These were combined into a raw batting quality score, then normalised within each match using z-scores to capture relative performance under the same conditions.

3. Defining Poor Innings

A poor innings is defined as one with a negative match-normalised quality score.
Poor innings were aggregated per team per match to create team-level indicators.

4. Team-Level Analysis

Batting data was pivoted so each match forms a single row, with columns representing each team’s poor-innings count.
For each match, the difference in poor innings between winning and losing teams was calculated.

5. Statistical Modelling

A logistic regression model was used to estimate the relationship between poor-innings advantage and win probability.
The model outputs were used to visualise how win probability changes as the poor-innings gap increases.

Key Findings:

Winning teams generally record fewer poor batting innings than losing teams.
The relationship between poor-innings advantage and win probability is positive and statistically significant.
Evidence suggests diminishing returns: beyond a certain point, further reductions in poor innings offer smaller marginal gains in win probability.

Visualisations:

Win probability vs poor-innings gap
Examples of highest innings by context

These plots are generated using matplotlib and are intended to communicate insights clearly rather than optimise aesthetics.

Tools & Technologies:

Python: pandas, NumPy, matplotlib, statsmodels
SQL: SQLite
Statistical Methods: logistic regression, z-score normalisation
Data Modelling: relational joins, pivot tables, aggregation

Notes & Limitations:

Sample size is limited to available Test matches; results should be interpreted directionally rather than conclusively.
Batting quality weights are heuristic and chosen for interpretability rather than optimisation.
Drawn matches are excluded from win–loss modelling.
