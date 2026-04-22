import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

TARGET = "FANTASY_PTS"

# Baseline: EWMA player stats only (no opponent or injury context)
BASELINE_FEATURES = [
    "FANTASY_PTS_EWMA5", "FANTASY_PTS_EWMA10",
    "MIN_EWMA5",         "MIN_EWMA10",
    "PTS_EWMA5",         "PTS_EWMA10",
    "REB_EWMA5",         "REB_EWMA10",
    "AST_EWMA5",         "AST_EWMA10",
    "STL_EWMA5",         "STL_EWMA10",
    "BLK_EWMA5",         "BLK_EWMA10",
    "TOV_EWMA5",         "TOV_EWMA10",
    "FG3M_EWMA5",        "FG3M_EWMA10",
]

# Full: baseline + opponent defensive context + own team offense + game situation + injury
FULL_FEATURES = BASELINE_FEATURES + [
    "OPP_DRTG",
    "OPP_PACE",
    "OWN_ORTG",
    "IS_HOME",
    "IS_INJURED",
    "IS_B2B",
]

MODELS = [
    ("Linear Regression", "Baseline", LinearRegression(), BASELINE_FEATURES),
    ("Linear Regression", "Full",     LinearRegression(), FULL_FEATURES),
    ("Random Forest",     "Full",     RandomForestRegressor(n_estimators=100, random_state=42), FULL_FEATURES),
    ("XGBoost",           "Full",     XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbosity=0), FULL_FEATURES),
]


def _model_filename(model_name, feature_set):
    return f"{model_name.lower().replace(' ', '_')}_{feature_set.lower()}.joblib"


def train_and_evaluate(data_dir, results_dir, models_dir=None):
    """
    Train and evaluate all models.

    Train/test split: train on 2024-25 season, test on 2025-26 season.
    Evaluates each model with test RMSE and 5-fold cross-validation RMSE.
    Saves a model_comparison.csv and per-model predictions to results_dir.
    If models_dir is provided, saves each trained model as a .joblib file.

    Returns a DataFrame summarizing model performance.
    """
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    if models_dir is not None:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_dir / "features.csv")

    train = df[df["SEASON"] == "2024-25"].copy()
    test  = df[df["SEASON"] == "2025-26"].copy()

    print(f"Train: {len(train)} rows | Test: {len(test)} rows")

    summary = []

    for model_name, feature_set, model, features in MODELS:
        label = f"{model_name} ({feature_set})"
        print(f"\nTraining {label}...")

        # Drop rows where any feature or target is NaN
        train_clean = train[features + [TARGET]].dropna()
        test_clean  = test[features + [TARGET]].dropna()

        X_train, y_train = train_clean[features], train_clean[TARGET]
        X_test,  y_test  = test_clean[features],  test_clean[TARGET]

        model.fit(X_train, y_train)

        # Save trained model to disk
        if models_dir is not None:
            model_path = models_dir / _model_filename(model_name, feature_set)
            joblib.dump(model, model_path)
            print(f"  Saved model to {model_path}")

        # Test RMSE
        preds = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, preds))

        # 5-fold cross-validation RMSE on training set
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring="neg_root_mean_squared_error"
        )
        cv_rmse_mean = -cv_scores.mean()
        cv_rmse_std  = cv_scores.std()

        print(f"  Test RMSE:  {test_rmse:.3f}")
        print(f"  CV RMSE:    {cv_rmse_mean:.3f} (+/- {cv_rmse_std:.3f})")

        summary.append({
            "Model":          model_name,
            "Features":       feature_set,
            "Test RMSE":      round(test_rmse, 3),
            "CV RMSE (mean)": round(cv_rmse_mean, 3),
            "CV RMSE (std)":  round(cv_rmse_std, 3),
            "Train rows":     len(X_train),
            "Test rows":      len(X_test),
        })

        # Save predictions vs actuals for this model
        pred_df = test[["PLAYER_NAME", "GAME_DATE", TARGET] + features].dropna().copy()
        pred_df["PREDICTED"] = preds
        pred_df["ERROR"] = pred_df["PREDICTED"] - pred_df[TARGET]
        pred_file = results_dir / f"predictions_{model_name.lower().replace(' ', '_')}_{feature_set.lower()}.csv"
        pred_df.to_csv(pred_file, index=False)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(results_dir / "model_comparison.csv", index=False)

    print("\n--- Model Comparison ---")
    print(summary_df[["Model", "Features", "Test RMSE", "CV RMSE (mean)", "CV RMSE (std)"]].to_string(index=False))

    return summary_df


def evaluate_from_saved(data_dir, results_dir, models_dir):
    """
    Load pre-trained models from models_dir and evaluate them on the 2025-26 test season.

    Expects one .joblib file per model, named by _model_filename().
    Saves results to results_dir/model_comparison.csv and per-model prediction CSVs.

    Returns a DataFrame summarizing model performance.
    """
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    models_dir = Path(models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_dir / "features.csv")
    test = df[df["SEASON"] == "2025-26"].copy()
    print(f"Test set: {len(test)} rows")

    summary = []

    for model_name, feature_set, _, features in MODELS:
        label = f"{model_name} ({feature_set})"
        model_path = models_dir / _model_filename(model_name, feature_set)

        if not model_path.exists():
            print(f"Skipping {label} — model file not found: {model_path}")
            continue

        print(f"\nEvaluating {label}...")
        model = joblib.load(model_path)

        test_clean = test[features + [TARGET]].dropna()
        X_test, y_test = test_clean[features], test_clean[TARGET]

        preds = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"  Test RMSE: {test_rmse:.3f}")

        summary.append({
            "Model":      model_name,
            "Features":   feature_set,
            "Test RMSE":  round(test_rmse, 3),
            "Test rows":  len(X_test),
        })

        pred_df = test[["PLAYER_NAME", "GAME_DATE", TARGET] + features].dropna().copy()
        pred_df["PREDICTED"] = preds
        pred_df["ERROR"] = pred_df["PREDICTED"] - pred_df[TARGET]
        pred_file = results_dir / f"predictions_{model_name.lower().replace(' ', '_')}_{feature_set.lower()}.csv"
        pred_df.to_csv(pred_file, index=False)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(results_dir / "model_comparison.csv", index=False)

    print("\n--- Evaluation Results ---")
    print(summary_df.to_string(index=False))

    return summary_df
