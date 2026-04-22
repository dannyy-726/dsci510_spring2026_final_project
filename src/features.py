import pandas as pd
from pathlib import Path
from config import FANTASY_SCORING

EWMA_SPANS = [5, 10]
EWMA_COLS = ["FANTASY_PTS", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]

# Maps the start-year portion of SEASON_ID to a season label
SEASON_ID_MAP = {2024: "2024-25", 2025: "2025-26"}


def build_features(data_dir, injuries_df=None):
    """
    Build the feature matrix from game logs and team stats.

    Steps:
      1. Calculate fantasy points per game
      2. Parse season label from SEASON_ID
      3. Parse home/away indicator and opponent abbreviation from MATCHUP
      4. Sort by player and date, then compute EWMA features (span 5 and 10)
         shifted by 1 to avoid data leakage, plus a back-to-back game flag
      5. Join opponent DRtg and Pace from Basketball Reference
      6. Add binary injury flag from Sleeper (if provided)

    Saves the result to data_dir/features.csv and returns the DataFrame.
    """
    data_dir = Path(data_dir)
    game_logs = pd.read_csv(data_dir / "game_logs_all.csv")
    team_stats = pd.read_csv(data_dir / "team_stats.csv")

    df = game_logs.copy()

    # 1. Fantasy points
    df["FANTASY_PTS"] = sum(
        df[col] * weight
        for col, weight in FANTASY_SCORING.items()
        if col in df.columns
    )

    # 2. Season label from SEASON_ID (e.g. 22024 -> "2024-25")
    df["SEASON"] = (
        df["SEASON_ID"].astype(str).str[-4:].astype(int).map(SEASON_ID_MAP)
    )

    # 3. Home/away and opponent abbreviation from MATCHUP
    # Format: "TEAM @ OPP" (away) or "TEAM vs. OPP" (home)
    df["IS_HOME"] = df["MATCHUP"].str.contains(r"vs\.", regex=True).astype(int)
    df["OPP_ABB"] = df["MATCHUP"].str.extract(r"(?:@ |vs\. )(\w+)")

    # 4. Sort by player and date for rolling calculations
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    # 5. Exponentially weighted moving averages (EWMA) shifted by 1 game to avoid leakage
    # EWMA gives more weight to recent games than a simple rolling average.
    # span=5 weights the last ~5 games with recency bias; span=10 does the same over ~10 games.
    print("Computing EWMA features...")
    for span in EWMA_SPANS:
        for col in EWMA_COLS:
            if col in df.columns:
                def compute_ewma(x):
                    return x.ewm(span=span, min_periods=1).mean().shift(1)
                df[f"{col}_EWMA{span}"] = (
                    df.groupby("PLAYER_ID")[col]
                    .transform(compute_ewma)
                )

    # 5b. Back-to-back game flag (IS_B2B=1 when days rest == 1)
    df["DAYS_REST"] = (
        df.groupby("PLAYER_ID")["GAME_DATE"].diff().dt.days.fillna(3)
    )
    df["IS_B2B"] = (df["DAYS_REST"] == 1).astype(int)

    # 6. Join opponent defensive stats by opponent team + season
    opp_stats = team_stats[["TEAM_ABB", "SEASON", "DRtg", "Pace"]].rename(
        columns={"TEAM_ABB": "OPP_ABB", "DRtg": "OPP_DRTG", "Pace": "OPP_PACE"}
    )
    df = df.merge(opp_stats, on=["OPP_ABB", "SEASON"], how="left")

    unmatched = df[df["OPP_DRTG"].isna()]["OPP_ABB"].nunique()
    if unmatched:
        print(f"Warning: {unmatched} opponent team(s) could not be matched to team stats.")

    # 7. Binary injury flag from Sleeper (current snapshot, player-level)
    if injuries_df is not None:
        injured_names = set(injuries_df["full_name"].dropna())
        df["IS_INJURED"] = df["PLAYER_NAME"].isin(injured_names).astype(int)
        print(f"Injury flag set: {df['IS_INJURED'].sum()} game rows flagged.")
    else:
        df["IS_INJURED"] = 0

    # 8. Join player's own team offensive rating (OWN_ORTG)
    # The first part of MATCHUP is always the player's own team (e.g. "LAC @ MIA" → LAC)
    df["OWN_TEAM_ABB"] = df["MATCHUP"].str.split(r" @| vs\.").str[0].str.strip()
    own_stats = team_stats[["TEAM_ABB", "SEASON", "ORtg"]].rename(
        columns={"TEAM_ABB": "OWN_TEAM_ABB", "ORtg": "OWN_ORTG"}
    )
    df = df.merge(own_stats, on=["OWN_TEAM_ABB", "SEASON"], how="left")

    unmatched_own = df[df["OWN_ORTG"].isna()]["OWN_TEAM_ABB"].nunique()
    if unmatched_own:
        print(f"Warning: {unmatched_own} own team(s) could not be matched to ORtg.")

    out_path = data_dir / "features.csv"
    df.to_csv(out_path, index=False)
    print(f"Features saved to {out_path} ({len(df)} rows, {df['PLAYER_ID'].nunique()} players)")

    return df
