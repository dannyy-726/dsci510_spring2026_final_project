# Enhancing NBA Fantasy Point Prediction

## Introduction
This project builds a predictive model for NBA fantasy basketball points using two seasons of player-level game data. The model incorporates individual player performance metrics, opponent team defensive statistics, and injury reports to evaluate whether contextual factors improve prediction accuracy. A baseline regression model is compared against models that include injury and defensive context to determine whether additional data sources significantly improve forecasting performance.

## Data Sources

- **nba_api** — Player name, team, game date, minutes played, points, rebounds, assists, steals, blocks, turnovers, FGA, FG%, 3PA, 3P%, plus/minus
- **Basketball Reference** — Team, ORtg, DRtg, Pace
- **Sleeper API** — Player name, team, injury status, injury notes, injury start date
- **ESPN Injuries** — Player name, team, position, injury description, injury status, expected return date

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
Fantasy points are calculated using standardized fantasy scoring rules applied to two seasons of player-level game logs. Feature engineering includes rolling averages (last 5 and 10 games), minutes played, usage-related metrics, opponent defensive rating, home/away indicators, and injury status. A baseline Linear Regression model is trained using only player stats, then compared against a full model using all features including injury and defensive context. A Random Forest Regressor is also evaluated. Models are assessed using RMSE and cross-validation to determine whether contextual features significantly improve prediction accuracy.

## Summary of Results
_To be updated upon project completion._

## How to Run
### Requirements
Install dependencies:

pip install -r requirements.txt

This project does not require any API keys. All data sources are either open APIs or publicly accessible web pages.

From the project root directory, run:

python main.py

This will fetch data from all sources, process features, train models, and save results to the `results/` folder. All fetched data will be stored in the `data/` folder.
