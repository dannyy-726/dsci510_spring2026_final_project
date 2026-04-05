import pandas as pd
import requests
import io
import time
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


