import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import DATA_DIR, RESULTS_DIR, MODELS_DIR
from load import get_all_players_game_logs, get_bball_ref_all_seasons, get_sleeper_data
from features import build_features
from model import train_and_evaluate, evaluate_from_saved


def download_model_from_drive(model_link, models_dir):
    """
    Download a model file from a Google Drive share link using gdown.
    Saves the downloaded .joblib file into models_dir.
    """
    try:
        import gdown
    except ImportError:
        print("gdown is required to download models from Google Drive.")
        print("Install it with: pip install gdown")
        sys.exit(1)

    models_dir.mkdir(parents=True, exist_ok=True)
    output = str(models_dir / "linear_regression_full.joblib")
    print("Downloading model from Google Drive...")
    gdown.download(model_link, output, fuzzy=True, quiet=False)
    print(f"Model downloaded to {output}")


def main():
    parser = argparse.ArgumentParser(
        description="NBA Fantasy Point Prediction Pipeline"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--train",
        action="store_true",
        help="Run full pipeline: fetch data, build features, train and evaluate all models.",
    )
    mode.add_argument(
        "--evaluation",
        action="store_true",
        help="Evaluate using pre-trained models (skips data fetch and training).",
    )
    parser.add_argument(
        "--model_link",
        type=str,
        default=None,
        help="Google Drive share link to download a pre-trained model before evaluation.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.train:
        # Step 1: Collect all player game logs
        game_logs = get_all_players_game_logs(DATA_DIR)
        if game_logs is None:
            print("Failed to collect game logs. Exiting.")
            return
        print(f"\nGame log collection complete: {len(game_logs)} rows")

        # Step 2: Collect Basketball Reference team stats (both seasons)
        team_stats = get_bball_ref_all_seasons()
        if team_stats is None:
            print("Failed to collect team stats. Exiting.")
            return
        team_stats.to_csv(DATA_DIR / "team_stats.csv", index=False)
        print(f"Team stats saved: {len(team_stats)} rows ({team_stats['SEASON'].nunique()} seasons)")

        # Step 3: Build feature matrix
        injuries = get_sleeper_data()
        features = build_features(DATA_DIR, injuries_df=injuries)
        if features is None:
            print("Failed to build features. Exiting.")
            return
        print(f"Feature matrix ready: {features.shape[1]} columns")

        # Step 4: Train, evaluate, and save all models
        results = train_and_evaluate(DATA_DIR, RESULTS_DIR, models_dir=MODELS_DIR)
        if results is None:
            print("Model training failed. Exiting.")
            return
        print("\nDone. Trained models saved to models/")

    elif args.evaluation:
        # Check that features.csv exists — it is required for evaluation
        features_path = DATA_DIR / "features.csv"
        if not features_path.exists():
            print("Error: data/features.csv not found.")
            print("Run --train first to fetch data and build the feature matrix.")
            return

        # Optionally download a model from Google Drive first
        if args.model_link:
            download_model_from_drive(args.model_link, MODELS_DIR)

        if not any(MODELS_DIR.glob("*.joblib")):
            print(f"No .joblib model files found in {MODELS_DIR}")
            print("Run with --train first, or provide a --model_link to download a model.")
            return

        results = evaluate_from_saved(DATA_DIR, RESULTS_DIR, MODELS_DIR)
        if results is None:
            print("Evaluation failed. Exiting.")
            return
        print("\nDone.")


if __name__ == "__main__":
    main()
