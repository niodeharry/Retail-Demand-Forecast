"""Model Training and Evaluation Pipeline.

This script loads the preprocessed retail demand sales data,
performs a chronological train-validation split, extracts temporal features,
trains Linear Regression (baseline) and Random Forest models,
evaluates them using MAE and RMSE, and saves the best model to app/best_model.pkl.
"""

import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_processed_data(csv_path: Path) -> pd.DataFrame:
    """Load the processed CSV data.

    Args:
        csv_path: Path to the processed training data CSV.

    Returns:
        Loaded DataFrame.
    """
    print(f"Loading processed data from {csv_path}...")
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed training data not found: {csv_path}")
    return pd.read_csv(csv_path)


def split_data(df: pd.DataFrame, split_date: str = "2017-08-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the DataFrame chronologically into train and validation sets.

    Args:
        df: Input DataFrame.
        split_date: The date to partition the data. Dates before this are train,
                    on or after are validation.

    Returns:
        A tuple of (train_df, val_df).
    """
    print(f"Splitting data chronologically at '{split_date}'...")
    train_mask = df["date"] < split_date
    train_df = df[train_mask].copy()
    val_df = df[~train_mask].copy()
    print(f"Train set: {train_df.shape[0]} rows, Validation set: {val_df.shape[0]} rows")
    return train_df, val_df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract temporal features and select columns for model training.

    Args:
        df: Input DataFrame containing 'date' and exogenous features.

    Returns:
        A tuple (X, y) containing the feature matrix and target vector.
    """
    df = df.copy()
    # Convert date to datetime
    date_col = pd.to_datetime(df["date"])

    # Extract temporal features
    df["month"] = date_col.dt.month
    df["day"] = date_col.dt.day
    df["dayofweek"] = date_col.dt.dayofweek

    # Select features
    feature_cols = [
        "store_nbr",
        "family",
        "onpromotion",
        "city",
        "state",
        "type",
        "cluster",
        "dcoilwtico",
        "is_holiday",
        "month",
        "day",
        "dayofweek"
    ]

    X = df[feature_cols]
    y = df["sales"]
    return X, y


def evaluate_model(model, X, y_true) -> tuple[float, float]:
    """Compute MAE and RMSE for a model on a given dataset.

    Args:
        model: Trained scikit-learn model.
        X: Feature matrix.
        y_true: True target values.

    Returns:
        A tuple of (mae, rmse).
    """
    y_pred = model.predict(X)
    # Clip negative predictions to 0 as sales cannot be negative
    y_pred = np.clip(y_pred, 0, None)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def main():
    """Main model training and evaluation script."""
    # Resolve paths relative to this script
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]

    csv_path = project_root / "data" / "processed" / "training_data.csv"
    app_dir = project_root / "app"

    # Ensure app directory exists
    app_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_processed_data(csv_path)

    # 2. Chronological Split
    train_df, val_df = split_data(df, split_date="2017-08-01")

    # 3. Prepare Features
    print("Preparing training and validation features...")
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)

    print(f"Features used for training: {list(X_train.columns)}")

    results = {}

    # 4. Train Linear Regression Baseline
    print("\n--- Training Linear Regression Baseline ---")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    train_mae_lr, train_rmse_lr = evaluate_model(lr_model, X_train, y_train)
    val_mae_lr, val_rmse_lr = evaluate_model(lr_model, X_val, y_val)
    print(f"Linear Regression -> Train RMSE: {train_rmse_lr:.4f}, Val RMSE: {val_rmse_lr:.4f}")
    results["LinearRegression"] = {
        "model": lr_model,
        "train_mae": train_mae_lr,
        "train_rmse": train_rmse_lr,
        "val_mae": val_mae_lr,
        "val_rmse": val_rmse_lr
    }

    # 5. Train Random Forest Regressor
    print("\n--- Training Random Forest Regressor ---")
    # Setting hyperparameters to prevent massive memory footprint and long training time.
    # Depth=15, Min Samples Leaf=5, Estimators=50 provides high accuracy, fast training (~20-40s),
    # and keeps the serialized model file small (~10-20MB) for easy app deployment.
    rf_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    train_mae_rf, train_rmse_rf = evaluate_model(rf_model, X_train, y_train)
    val_mae_rf, val_rmse_rf = evaluate_model(rf_model, X_val, y_val)
    print(f"Random Forest -> Train RMSE: {train_rmse_rf:.4f}, Val RMSE: {val_rmse_rf:.4f}")
    results["RandomForest"] = {
        "model": rf_model,
        "train_mae": train_mae_rf,
        "train_rmse": train_rmse_rf,
        "val_mae": val_mae_rf,
        "val_rmse": val_rmse_rf
    }

    # 6. Compare and Select Best Model
    print("\n--- Model Performance Comparison ---")
    print(f"{'Model':<20} | {'Train MAE':<10} | {'Train RMSE':<10} | {'Val MAE':<10} | {'Val RMSE':<10}")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(f"{model_name:<20} | {metrics['train_mae']:<10.4f} | {metrics['train_rmse']:<10.4f} | {metrics['val_mae']:<10.4f} | {metrics['val_rmse']:<10.4f}")

    # Determine best model based on validation RMSE
    best_model_name = min(results, key=lambda k: results[k]["val_rmse"])
    best_model = results[best_model_name]["model"]
    print(f"\nBest model selected: {best_model_name} (lowest Val RMSE: {results[best_model_name]['val_rmse']:.4f})")

    # 7. Save Model
    model_output_path = app_dir / "best_model.pkl"
    print(f"Saving best model to {model_output_path}...")
    with open(model_output_path, "wb") as f:
        pickle.dump(best_model, f)

    print("Model training pipeline completed successfully!")


if __name__ == "__main__":
    main()
