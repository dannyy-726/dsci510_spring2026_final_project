from pathlib import Path

# --- Paths ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# --- Data Source URLs ---
ESPN_INJURIES_URL = "https://www.espn.com/nba/injuries"
SLEEPER_NBA_PLAYERS_URL = "https://api.sleeper.app/v1/players/nba"
BBALL_REF_SEASON_URL = "https://www.basketball-reference.com/leagues/NBA_{season_year}.html"

# --- NBA API Settings ---
NBA_API_SEASONS = ["2025-26", "2024-25"]
NBA_API_SLEEP_SECONDS = 1.5

# --- Sleeper Injury Columns ---
SLEEPER_INJURY_COLS = [
    "full_name",
    "team",
    "position",
    "injury_status",
    "injury_notes",
    "injury_start_date",
]

# --- Fantasy Scoring System ---
FANTASY_SCORING = {
    "PTS": 1.0,
    "REB": 1.2,
    "AST": 1.5,
    "STL": 3.0,
    "BLK": 3.0,
    "TOV": -1.0,
    "FG3M": 0.5,
}

# --- Team Name Mapping: Basketball Reference full name → nba_api abbreviation ---
# Basketball Reference appends '*' to teams that made the playoffs; strip it before lookup.
BBALL_REF_TO_NBA_API = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

#AI generated
# --- HTTP Headers ---
SCRAPER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}
