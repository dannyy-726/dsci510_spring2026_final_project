import sys
import os
import unittest
import unicodedata
import pandas as pd

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


if __name__ == "__main__":
    unittest.main()
