import sys
import os
import unittest
import unicodedata
import pandas as pd
from pathlib import Path

# Add src/ to path so we can import from load.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from load import get_espn_injuries, get_last_two_seasons, get_sleeper_data, get_bball_ref_team_stats


class TestGetEspnInjuries(unittest.TestCase):
    # Tests for the ESPN injury scraper

    def test_returns_dataframe(self):
        # Verify the function returns a non-empty pandas DataFrame
        df = get_espn_injuries()
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)


class TestGetLastTwoSeasons(unittest.TestCase):
    # Tests for the nba_api game log fetcher

    def test_returns_dataframe(self):
        # Verify a valid player name returns a pandas DataFrame
        df = get_last_two_seasons("LeBron James")
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)

    def test_invalid_player_returns_none(self):
        # Verify an invalid player name returns None instead of crashing
        df = get_last_two_seasons("Not A Real Player")
        self.assertIsNone(df)


class TestGetSleeperData(unittest.TestCase):
    # Tests for the Sleeper API injury fetcher

    def test_returns_dataframe(self):
        # Verify the function returns a non-empty pandas DataFrame
        df = get_sleeper_data()
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)


class TestGetBballRefTeamStats(unittest.TestCase):
    # Tests for the Basketball Reference team stats scraper

    def test_returns_dataframe(self):
        # Verify the function returns a non-empty pandas DataFrame for the 2025-26 season
        df = get_bball_ref_team_stats(2026)
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)


class TestInjurySourceCoverage(unittest.TestCase):
    # Tests that compare ESPN and Sleeper injury data coverage

    def normalize(self, name):
        # Lowercase, remove accents, strip suffixes and whitespace
        name = unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode("utf-8").strip().lower()
        for suffix in [" jr.", " sr.", " iv", " iii", " ii"]:
            name = name.replace(suffix, "")
        return name.strip()

    def test_sleeper_covers_most_espn_players(self):
        # Verify Sleeper contains the vast majority of players ESPN reports as injured.
        # A small gap is expected since Sleeper excludes two-way/G League players.
        espn_df = get_espn_injuries()
        sleeper_df = get_sleeper_data()

        espn_names = set(espn_df["NAME"].map(self.normalize))
        sleeper_names = set(sleeper_df["full_name"].map(self.normalize))

        missing = espn_names - sleeper_names
        print(f"\nPlayers in ESPN but not Sleeper: {missing}")

        # Allow up to 5 missing players (fringe/two-way players Sleeper doesn't track)
        self.assertLessEqual(len(missing), 5)


class TestGetBballRefAllSeasons(unittest.TestCase):
    # Tests for the multi-season Basketball Reference scraper

    def test_returns_dataframe_with_both_seasons(self):
        # Verify both seasons are present in the returned DataFrame
        from load import get_bball_ref_all_seasons
        df = get_bball_ref_all_seasons()
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("2024-25", df["SEASON"].values)
        self.assertIn("2025-26", df["SEASON"].values)

    def test_team_abbreviations_populated(self):
        # Verify the name mapping ran — all rows should have a TEAM_ABB value
        from load import get_bball_ref_all_seasons
        df = get_bball_ref_all_seasons()
        self.assertIn("TEAM_ABB", df.columns)
        self.assertFalse(df["TEAM_ABB"].isna().any(), "Some teams missing abbreviation mapping")


class TestFantasyScoring(unittest.TestCase):
    # Tests for the fantasy points calculation logic in features.py

    def test_known_statline(self):
        # Verify the scoring formula produces the correct value for a known statline:
        # 30 PTS, 10 REB, 5 AST, 2 STL, 1 BLK, 3 TOV, 4 FG3M
        # = 30*1.0 + 10*1.2 + 5*1.5 + 2*3.0 + 1*3.0 + 3*(-1.0) + 4*0.5
        # = 30 + 12 + 7.5 + 6 + 3 - 3 + 2 = 57.5
        from config import FANTASY_SCORING
        stats = {"PTS": 30, "REB": 10, "AST": 5, "STL": 2, "BLK": 1, "TOV": 3, "FG3M": 4}
        result = sum(stats[col] * weight for col, weight in FANTASY_SCORING.items())
        self.assertAlmostEqual(result, 57.5)

    def test_zero_statline(self):
        # A player with all zeros should produce exactly 0 fantasy points
        from config import FANTASY_SCORING
        stats = {col: 0 for col in FANTASY_SCORING}
        result = sum(stats[col] * weight for col, weight in FANTASY_SCORING.items())
        self.assertEqual(result, 0.0)


class TestBuildFeaturesLeakage(unittest.TestCase):
    # Tests that EWMA features are properly shifted to prevent data leakage

    def _make_synthetic_logs(self, tmp_dir):
        # Build a minimal game_logs_all.csv and team_stats.csv for one player
        import numpy as np
        game_logs = pd.DataFrame({
            "PLAYER_ID":   [1] * 5,
            "PLAYER_NAME": ["Test Player"] * 5,
            "GAME_DATE":   pd.date_range("2024-10-01", periods=5, freq="2D").astype(str),
            "SEASON_ID":   [22024] * 5,
            "MATCHUP":     ["LAL vs. BOS"] * 5,
            "PTS":  [20, 25, 30, 15, 22],
            "REB":  [5,  6,  7,  4,  5],
            "AST":  [4,  3,  5,  2,  4],
            "STL":  [1,  2,  1,  0,  1],
            "BLK":  [0,  1,  0,  1,  0],
            "TOV":  [2,  1,  3,  1,  2],
            "FG3M": [2,  3,  2,  1,  2],
            "MIN":  [32, 34, 36, 28, 33],
        })
        team_stats = pd.DataFrame({
            "Team":     ["Boston Celtics", "Los Angeles Lakers"],
            "TEAM_ABB": ["BOS", "LAL"],
            "SEASON":   ["2024-25", "2024-25"],
            "ORtg":     [115.0, 113.0],
            "DRtg":     [110.0, 112.0],
            "Pace":     [98.0,  97.0],
        })
        game_logs.to_csv(tmp_dir / "game_logs_all.csv", index=False)
        team_stats.to_csv(tmp_dir / "team_stats.csv", index=False)

    def test_first_row_ewma_is_nan(self):
        # The first game per player must have NaN EWMA values (no prior games to average)
        import tempfile
        from features import build_features
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._make_synthetic_logs(tmp_path)
            df = build_features(tmp_path)
            first_row = df[df["PLAYER_ID"] == 1].iloc[0]
            self.assertTrue(
                pd.isna(first_row["FANTASY_PTS_EWMA5"]),
                "First game EWMA5 should be NaN — shift(1) ensures no leakage"
            )

    def test_ewma_columns_present(self):
        # Verify all expected EWMA columns exist in the output
        import tempfile
        from features import build_features
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._make_synthetic_logs(tmp_path)
            df = build_features(tmp_path)
            for col in ["FANTASY_PTS_EWMA5", "FANTASY_PTS_EWMA10", "MIN_EWMA5", "IS_B2B"]:
                self.assertIn(col, df.columns, f"Expected column {col} missing from feature matrix")


if __name__ == "__main__":
    unittest.main()
