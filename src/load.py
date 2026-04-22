import pandas as pd
import requests
import io
import time
from pathlib import Path
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from config import (
    ESPN_INJURIES_URL,
    SLEEPER_NBA_PLAYERS_URL,
    BBALL_REF_SEASON_URL,
    NBA_API_SEASONS,
    NBA_API_SLEEP_SECONDS,
    SLEEPER_INJURY_COLS,
    SCRAPER_HEADERS,
    BBALL_REF_TO_NBA_API,
)

def get_espn_injuries():
    print("--- Fetching ESPN's NBA Injury Data ---")
    url = ESPN_INJURIES_URL
    headers = SCRAPER_HEADERS
    
    try:
        # 1. Get the page content
        response = requests.get(url, headers=headers, timeout=10)
        
        # 2. Wrap the text in StringIO so Pandas treats it as a file
        # This prevents the 'hanging' issue with large JSON strings
        html_data = io.StringIO(response.text)
        
        # 3. Read the tables
        tables = pd.read_html(html_data)
        
        if not tables:
            print("No tables found on the page.")
            return None

        # 4. Combine all tables into one, was divided by teams
        master_df = pd.concat(tables, ignore_index=True)
        
        print(f"Success! Found {len(master_df)} total injured players.")
        print(master_df.head(150)) # Show just the first 10 rows
        
        return master_df

    except Exception as e:
        print(f"Scrape failed: {e}")
        return None


#retrieve nba_api data
def get_last_two_seasons(player_name):
    # Use the built-in search tool instead of manual list comprehension
    player_dict = players.find_players_by_full_name(player_name)
    
    if not player_dict:
        # Check if maybe it's a spelling issue (e.g., Nikola Jokić vs Nikola Jokic)
        print(f"Player '{player_name}' not found. Check spelling or special characters.")
        return None
        
    player_id = player_dict[0]['id']
    print(f"Fetching data for {player_dict[0]['full_name']} (ID: {player_id})")

    all_logs = []

    for season in NBA_API_SEASONS:
        try:
            print(f"Loading {season}...")
            log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
            df = log.get_data_frames()[0]
            if not df.empty:
                all_logs.append(df)

            time.sleep(NBA_API_SLEEP_SECONDS)
        except Exception as e:
            print(f"Error fetching {season}: {e}")

    if not all_logs:
        return None

    return pd.concat(all_logs, ignore_index=True)


#retrieve sleeper api data
def get_sleeper_data():
    url = SLEEPER_NBA_PLAYERS_URL
    print("--- Requesting Sleeper NBA Data ---")
    
    try:
        # 1. Fetch the data
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            print("Failed to connect to Sleeper.")
            return None
        
        # 2. Parse the JSON dictionary
        player_dict = response.json()
        
        # 3. Convert to DataFrame (Keys are Sleeper IDs)
        # use .from_dict so the IDs become the index
        df = pd.DataFrame.from_dict(player_dict, orient='index')
       
        # This prints all column names in alphabetical order
        # print(sorted(df.columns.tolist()))
       
        # 4. Filter for Injury Reports
        # look for any row where 'injury_status' is NOT null
        # We also filter out players without a 'team' (Free Agents)
        injuries = df[df['injury_status'].notna() & df['team'].notna()].copy()
        
        # 5. Clean up the columns
        injuries = injuries[[c for c in SLEEPER_INJURY_COLS if c in injuries.columns]]
        
        # Sort by team
        injuries = injuries.sort_values(by='team')
        
        print(f"--- Load Complete: {len(injuries)} Injured Players Found ---")
        return injuries


    except Exception as e:
        print(f"Error loading Sleeper data: {e}")
        return None


def get_bball_ref_team_stats(season_year):

    url = BBALL_REF_SEASON_URL.format(season_year=season_year)
    print(f"--- Fetching Basketball Reference Team Stats ({season_year}) ---")

    try:
        tables = pd.read_html(url)

        advanced_table = None
        for table in tables:
            # Flatten multi-level column headers if present
            table.columns = [
                col[1] if isinstance(col, tuple) else col
                for col in table.columns
            ]
            if "DRtg" in table.columns:
                advanced_table = table.copy()
                break

        if advanced_table is None:
            print("Advanced stats table not found on page.")
            return None

        # Drop the league average summary row
        if "Team" in advanced_table.columns:
            advanced_table = advanced_table[advanced_table["Team"] != "League Average"]

        print(f"Success! Found {len(advanced_table)} teams.")
        return advanced_table

    except Exception as e:
        print(f"Basketball Reference scrape failed: {e}")
        return None


def get_player_positions():
    """
    Fetch position for all active NBA players from the Sleeper API.
    Returns a DataFrame with full_name and position columns.
    """
    url = SLEEPER_NBA_PLAYERS_URL
    print("--- Fetching player positions from Sleeper ---")

    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            print("Failed to connect to Sleeper.")
            return None

        df = pd.DataFrame.from_dict(response.json(), orient="index")
        df = (
            df[df["team"].notna() & df["position"].notna()]
            [["full_name", "position"]]
            .drop_duplicates("full_name")
            .query("position in ['PG', 'SG', 'SF', 'PF', 'C']")
            .reset_index(drop=True)
        )

        print(f"Found positions for {len(df)} players.")
        return df

    except Exception as e:
        print(f"Error fetching player positions: {e}")
        return None


def get_bball_ref_all_seasons():
    """
    Fetch Basketball Reference team advanced stats for both seasons.
    Strips playoff asterisks from team names and adds a TEAM_ABB column
    using BBALL_REF_TO_NBA_API so rows can be joined to nba_api game logs.
    Returns a combined DataFrame with a SEASON column (e.g. '2024-25').
    """
    season_map = {"2024-25": 2025, "2025-26": 2026}
    frames = []

    for season_label, season_year in season_map.items():
        df = get_bball_ref_team_stats(season_year)
        if df is None:
            print(f"Skipping {season_label} — fetch failed.")
            continue

        df = df.copy()
        df["Team"] = df["Team"].str.replace("*", "", regex=False).str.strip()
        df["TEAM_ABB"] = df["Team"].map(BBALL_REF_TO_NBA_API)
        df["SEASON"] = season_label

        unmatched = df[df["TEAM_ABB"].isna()]["Team"].tolist()
        if unmatched:
            print(f"Warning: {season_label} — no abbreviation found for: {unmatched}")

        frames.append(df)

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)


def get_all_players_game_logs(data_dir):
    """
    Fetch game logs for all active NBA players across NBA_API_SEASONS.

    Uses per-player CSV checkpoints in data_dir/game_logs/ so interrupted
    runs resume without re-fetching completed players. Combines all player
    files into data_dir/game_logs_all.csv at the end.

    Returns a combined DataFrame of all player game logs.
    """
    cache_dir = Path(data_dir) / "game_logs"
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_players = players.get_active_players()
    total = len(all_players)
    print(f"--- Fetching game logs for {total} active players ---")

    start_time = time.time()
    fetched = 0   # counts players actually fetched (not skipped)
    cached_count = 0
    skipped = {}  # {reason: [player_name, ...]}

    for i, player in enumerate(all_players, start=1):
        player_id = player["id"]
        player_name = player["full_name"]
        cache_file = cache_dir / f"{player_id}.csv"

        if cache_file.exists():
            cached_count += 1
            print(f"[{i}/{total}] Skipped (cached): {cached_count} players so far...", end="\r", flush=True)
            continue
        
        #AI generated - for visibility of time, and trouble shooting
        elapsed = time.time() - start_time
        pct = i / total * 100
        if fetched > 0:
            avg_per_player = elapsed / fetched
            eta_min = avg_per_player * (total - i) / 60
            status = f"[{i}/{total} | {pct:.1f}% | {elapsed/60:.1f}m elapsed | ~{eta_min:.1f}m left] Fetching: {player_name:<30}"
        else:
            status = f"[{i}/{total} | {pct:.1f}%] Fetching: {player_name:<30}"
        print(status, end="\r", flush=True)

        fetched += 1
        all_logs = []
        errors = []

        for season in NBA_API_SEASONS:
            try:
                log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
                df = log.get_data_frames()[0]
                if not df.empty:
                    all_logs.append(df)
                time.sleep(NBA_API_SLEEP_SECONDS)
            except Exception as e:
                errors.append(str(e))
                time.sleep(NBA_API_SLEEP_SECONDS)

        if all_logs:
            player_df = pd.concat(all_logs, ignore_index=True)
            player_df["PLAYER_NAME"] = player_name
            player_df["PLAYER_ID"] = player_id
            player_df.to_csv(cache_file, index=False)
        else:
            reason = errors[0] if errors else "No game logs returned for either season"
            skipped.setdefault(reason, []).append(player_name)

    # Combine all cached files into one master DataFrame
    files = sorted(cache_dir.glob("*.csv"))
    if not files:
        print("No game log files found.")
        return None

    print(f"\nCombining {len(files)} player files...")
    combined = pd.concat(
        (pd.read_csv(f) for f in files),
        ignore_index=True
    )

    out_path = Path(data_dir) / "game_logs_all.csv"
    combined.to_csv(out_path, index=False)
    print(f"Saved combined game logs to {out_path} ({len(combined)} rows, {combined['PLAYER_ID'].nunique()} players)")

    # Report players with no data
    missing_count = sum(len(names) for names in skipped.values())
    if missing_count:
        print(f"\n--- {missing_count} player(s) had no game log data ---")
        for reason, names in skipped.items():
            print(f"\nReason: {reason}")
            for name in names:
                print(f"  - {name}")

    if len(files) != total:
        print(f"\nSummary: {len(files)}/{total} players have game log data.")

    return combined


