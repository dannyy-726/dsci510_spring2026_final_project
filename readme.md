# Enhancing NBA Fantasy Point Prediction

## Introduction
This project builds a predictive model for NBA fantasy basketball points using two seasons of player-level game data (2024-25 and 2025-26). The model incorporates individual player performance metrics, opponent team defensive statistics, and injury reports to evaluate whether contextual factors improve prediction accuracy. A baseline Linear Regression model is compared against a full model that includes injury and defensive context, as well as a Random Forest Regressor and XGBoost model, to determine whether additional data sources or advanced models significantly improve forecasting performance.

## Data Sources

| Source | Description | Data Collected |
|--------|-------------|----------------|
| **nba_api** | Python client for NBA's stats.nba.com | Player name, team, game date, minutes played, points, rebounds, assists, steals, blocks, turnovers, FGA, FG%, 3PA, 3P%, plus/minus |
| **Basketball Reference** | Team advanced stats scraped via `pd.read_html()` | Team, ORtg, DRtg, Pace (per season) |
| **Sleeper API** | Free REST API for real-time NBA injury reports | Player name, team, injury status, injury notes, injury start date |
| **ESPN Injuries** | Injury data scraped via `pd.read_html()` (validation) | Player name, team, position, injury description, injury status, expected return date |

## Fantasy Scoring System

| Stat | Points |
|------|--------|
| Point | +1.0 |
| Rebound | +1.2 |
| Assist | +1.5 |
| Steal | +3.0 |
| Block | +3.0 |
| Turnover | -1.0 |
| 3-pointer made | +0.5 |

## Analysis
Fantasy points are calculated using the scoring system above applied to two seasons of player-level game logs. Feature engineering includes exponentially weighted moving averages (EWMA, span=5 and span=10) for all major stats, opponent defensive rating (DRtg), opponent pace, own team offensive rating (ORtg), home/away indicators, back-to-back game flag, and injury status. EWMA features are shifted by one game to prevent data leakage.

Models are trained on the 2024-25 season and tested on the 2025-26 season. Four models are evaluated:
- **Baseline Linear Regression** — rolling player stats only
- **Full Linear Regression** — rolling stats + opponent DRtg, pace, own ORtg, home/away, injury flag
- **Random Forest Regressor** — full feature set
- **XGBoost** — full feature set

Models are assessed using RMSE on the held-out test season and 5-fold cross-validation on the training season.

## Summary of Results

| Model | Features | Test RMSE | CV RMSE |
|-------|----------|-----------|---------|
| Linear Regression | Baseline | 10.177 | 10.400 |
| **Linear Regression** | **Full** | **10.146** | **10.359** |
| Random Forest | Full | 10.350 | 10.621 |
| XGBoost | Full | 10.209 | 10.525 |

**Key findings:**
- Exponentially weighted moving averages (EWMA) of recent player performance are the dominant predictors of fantasy points. EWMA outperformed simple rolling averages by giving more weight to the most recent games.
- Contextual features (opponent DRtg, pace, home/away, injury status, own team ORtg, back-to-back flag) improve the Linear Regression model by ~0.04 RMSE over the baseline.
- Linear Regression Full achieved the best test RMSE (10.146), outperforming both Random Forest and XGBoost — suggesting the relationships in the data are mostly linear once EWMA features are included.
- An RMSE of ~10 on a mean fantasy score of 23 points reflects the inherent noise in individual game performance (foul trouble, minutes changes, matchup variance).

## How to Run

### Requirements
Install dependencies:

```bash
pip install -r requirements.txt
```

This project does not require any API keys. All data sources are either open APIs or publicly accessible web pages.

### Running the pipeline

`main.py` supports two modes:

**Train mode** — fetches all data, builds features, trains all models, and saves them to `models/`:

```bash
python main.py --train
```

This will:
1. Fetch player game logs from nba_api for all active NBA players (2024-25 and 2025-26 seasons). Game logs are cached to `data/game_logs/` so interrupted runs resume automatically.
2. Fetch team advanced stats (DRtg, ORtg, Pace) from Basketball Reference for both seasons and save to `data/team_stats.csv`.
3. Fetch current injury reports from the Sleeper API.
4. Build the feature matrix and save to `data/features.csv`.
5. Train and evaluate all models, saving results to `results/model_comparison.csv` and trained model files to `models/`.

**Note:** Step 1 fetches ~530 players with a 1.5s rate limit delay and takes approximately 30-40 minutes on a fresh run. Subsequent runs skip already-cached players.

**Evaluation mode** — loads pre-trained models and evaluates on the 2025-26 test season.

> **Note:** `data/features.csv` must exist before running evaluation. Run `--train` first, or ensure the feature matrix has already been built.

```bash
python main.py --evaluation
```

To download a pre-trained model from Google Drive before evaluating:

```bash
python main.py --evaluation --model_link <google_drive_share_link>
```

### Reproducing the analysis
Open and run `results.ipynb` to see the full pipeline walkthrough, visualizations, and model comparison.

## Generative AI Usage

The following sections of this project were assisted by Claude (Anthropic) as noted below. All other code was written independently.

**1. `SCRAPER_HEADERS` — `src/config.py`**
The HTTP request headers used for scraping Basketball Reference were assisted by Claude. In the original project proposal, scraping Basketball Reference worked using the basic headers demonstrated in class. After submitting the proposal, the site began returning bot-detection errors and blocking automated requests. Claude was consulted to troubleshoot the issue and suggested a more complete browser-like User-Agent string, which resolved the blocking. The relevant constant is marked `# AI generated` in `config.py`.

**2. Live time tracking in `get_all_players_game_logs()` — `src/load.py`**
The elapsed time display, ETA calculation, and live status line printed during the bulk game log fetch were assisted by Claude. Fetching ~530 players takes 30–40 minutes on a fresh run, and the time visibility feature was added to make it easier to monitor progress and troubleshoot mid-run issues (e.g., identifying slow responses or stalls). The relevant block is marked `# AI generated` in `load.py`.

**3. `io.StringIO()` fix for ESPN scraping — `src/load.py`**
When scraping ESPN's injury page, passing `response.text` directly to `pd.read_html()` caused the program to hang indefinitely with no error or output. I consulted Claude to understand why this happens — `pd.read_html()` treats a raw string as a URL and attempts a second network request rather than parsing the HTML content. Claude suggested wrapping the response in `io.StringIO()` to force pandas to treat it as a file object. Once I understood the reason, I implemented the fix myself.

**4. Multi-level column header flattening — `src/load.py`**
Basketball Reference returns HTML tables with merged/grouped column headers (similar to merged cells in Excel). When pandas reads these, it represents them as tuples (e.g., `("Unnamed", "DRtg")`) rather than plain strings, which caused column lookups to fail silently. I consulted Claude to understand what multi-level headers are and how pandas encodes them. Claude explained the tuple structure and the concept of flattening. I then wrote the flattening logic myself: `col[1] if isinstance(col, tuple) else col`.

**5. EWMA with `.shift(1)` for data leakage prevention — `src/features.py`**
I knew the model needed rolling averages of recent player performance as features, but I was not aware of the concept of data leakage — the mistake of allowing the model to see the current game's stats when predicting that same game. I consulted Claude to understand this concept, and it explained that the standard prevention technique is shifting all rolling features forward by one game using `.shift(1)`, so each row's features only reflect games played before that game. Once I understood the concept, I implemented the EWMA calculation and shift myself.

**6. Filesystem checkpoint system in `get_all_players_game_logs()` — `src/load.py`**
Fetching game logs for ~530 players with a required 1.5-second delay between requests takes 30–40 minutes. When runs were interrupted mid-way (network drops, timeouts, closing the laptop), all progress was lost and the fetch had to restart from the beginning.

Other approaches commonly used in the nba_api community were considered but not adopted:
- **Randomized sleep intervals (2–6 seconds)** — adds time to an already slow process and does not solve the problem of interrupted runs losing all progress
- **Exponential backoff** — handles retry logic on errors but still requires restarting from the beginning if the program exits
- **Proxy/VPN rotation** — adds significant infrastructure complexity and cost, unreliable with public proxies, and unnecessary since rate limiting was not the core issue
- **Server-side caching (Redis/Memcached)** — requires running a separate database service, adding complexity well beyond the scope of a data collection script

I consulted Claude for a simpler, more reliable approach suited to a local script. It suggested a filesystem checkpoint pattern: save each player's data to an individual CSV immediately after fetching, and skip players whose file already exists on re-runs. This eliminated the need for any external infrastructure, made interrupted runs fully recoverable, and was straightforward to implement. I implemented the checkpoint logic, progress tracking, and file combining myself.
