"""Data Preprocessing Pipeline for Retail Demand Forecasting.

This script loads raw data from data/raw/, performs downsampling,
merges exogenous variables (oil prices, holiday events, store metadata),
imputes missing values, encodes categorical variables, and saves
the processed dataset to data/processed/training_data.csv.
"""

import json
import os
from pathlib import Path
import pandas as pd


def load_raw_data(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSV files from the specified raw data directory.

    Args:
        raw_dir: Path to the directory containing raw CSV files.

    Returns:
        A tuple of DataFrames: (train_df, oil_df, holidays_df, stores_df)
    """
    print("Loading raw datasets...")
    train_path = raw_dir / "train.csv"
    oil_path = raw_dir / "oil.csv"
    holidays_path = raw_dir / "holidays_events.csv"
    stores_path = raw_dir / "stores.csv"

    # Check existence
    for path in [train_path, oil_path, holidays_path, stores_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required raw data file not found: {path}")

    # Load datasets
    train_df = pd.read_csv(train_path)
    oil_df = pd.read_csv(oil_path)
    holidays_df = pd.read_csv(holidays_path)
    stores_df = pd.read_csv(stores_path)

    print(f"Loaded train: {train_df.shape}, oil: {oil_df.shape}, holidays: {holidays_df.shape}, stores: {stores_df.shape}")
    return train_df, oil_df, holidays_df, stores_df


def preprocess_data(
    train_df: pd.DataFrame,
    oil_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    start_date: str = "2016-01-01"
) -> tuple[pd.DataFrame, dict]:
    """Process and merge the retail datasets.

    Args:
        train_df: Training transaction sales data.
        oil_df: Exogenous daily oil price data.
        holidays_df: Holiday and event calendar data.
        stores_df: Store metadata.
        start_date: Filter transaction records to be on or after this date (downsampling).

    Returns:
        A tuple containing:
            - The processed DataFrame.
            - A dictionary containing category-to-code mappings for categorical variables.
    """
    print(f"Starting preprocessing with start_date='{start_date}'...")

    # 1. Downsampling: filter train_df by date
    # Convert dates to string to ensure consistent string operations/formatting
    train_df["date"] = train_df["date"].astype(str)
    train_df = train_df[train_df["date"] >= start_date].copy()
    print(f"Train dataset size after downsampling: {train_df.shape}")

    # 2. Merge Store Metadata
    # Merge stores metadata first to get city/state for holiday scoping
    print("Merging store metadata...")
    train_df = train_df.merge(stores_df, on="store_nbr", how="left")

    # 3. Merge Oil Prices
    print("Merging oil prices...")
    oil_df["date"] = oil_df["date"].astype(str)
    train_df = train_df.merge(oil_df, on="date", how="left")

    # 4. Merge Holidays & Events
    print("Processing and merging holidays...")
    holidays_df["date"] = holidays_df["date"].astype(str)

    # Filter out transferred holidays and work days
    # (Transferred holidays are celebrated on the transfer day, which is a separate Transfer entry)
    # (Work Day is a day off swapped back to a regular work day, so NOT a holiday)
    holidays_filtered = holidays_df[
        (holidays_df["transferred"].astype(str).str.lower() != "true") &
        (holidays_df["type"] != "Work Day")
    ].copy()

    # Split holidays into national, regional, and local scopes to avoid row duplication on merge
    # National holidays apply to all stores
    national_holidays = holidays_filtered[holidays_filtered["locale"] == "National"][["date"]].drop_duplicates()
    national_holidays["is_holiday_nat"] = 1

    # Regional holidays apply to stores in the same state
    regional_holidays = holidays_filtered[holidays_filtered["locale"] == "Regional"][["date", "locale_name"]].drop_duplicates()
    regional_holidays = regional_holidays.rename(columns={"locale_name": "state"})
    regional_holidays["is_holiday_reg"] = 1

    # Local holidays apply to stores in the same city
    local_holidays = holidays_filtered[holidays_filtered["locale"] == "Local"][["date", "locale_name"]].drop_duplicates()
    local_holidays = local_holidays.rename(columns={"locale_name": "city"})
    local_holidays["is_holiday_loc"] = 1

    # Merge holiday indicators
    train_df = train_df.merge(national_holidays, on="date", how="left")
    train_df = train_df.merge(regional_holidays, on=["date", "state"], how="left")
    train_df = train_df.merge(local_holidays, on=["date", "city"], how="left")

    # Combine into a single binary is_holiday feature
    train_df["is_holiday"] = (
        (train_df["is_holiday_nat"] == 1) |
        (train_df["is_holiday_reg"] == 1) |
        (train_df["is_holiday_loc"] == 1)
    ).astype(int)

    # Drop temporary columns
    train_df = train_df.drop(columns=["is_holiday_nat", "is_holiday_reg", "is_holiday_loc"])

    # 5. Handle Missing Values
    print("Handling missing values...")
    # Oil price: forward fill then backward fill (for weekends and holidays)
    # Sort by date first to ensure ffill is chronologically correct
    train_df = train_df.sort_values(by=["date", "store_nbr", "family"]).reset_index(drop=True)
    train_df["dcoilwtico"] = train_df["dcoilwtico"].ffill().bfill()

    # is_holiday: fill NaNs with 0
    train_df["is_holiday"] = train_df["is_holiday"].fillna(0).astype(int)

    # 6. Categorical Variable Encoding
    print("Encoding categorical variables...")
    categorical_cols = ["family", "city", "state", "type"]
    categorical_mappings = {}

    for col in categorical_cols:
        train_df[col] = train_df[col].astype(str)
        # Sort values so mappings are deterministic and alphabetical
        categories = sorted(train_df[col].unique())
        mapping = {category: idx for idx, category in enumerate(categories)}
        categorical_mappings[col] = mapping

        # Map to integer codes
        train_df[col] = train_df[col].map(mapping)

    print("Preprocessing completed successfully!")
    return train_df, categorical_mappings


def main():
    """Main execution entry point."""
    # Resolve paths relative to this script: src/features/data_preprocessing.py -> project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]

    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    # Ensure processed directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load and preprocess
        train_df, oil_df, holidays_df, stores_df = load_raw_data(raw_dir)
        processed_df, mappings = preprocess_data(train_df, oil_df, holidays_df, stores_df)

        # Drop the row 'id' column as it is an arbitrary index and not useful for modeling
        if "id" in processed_df.columns:
            processed_df = processed_df.drop(columns=["id"])

        # Save processed dataset
        output_csv_path = processed_dir / "training_data.csv"
        print(f"Saving processed data to {output_csv_path}...")
        processed_df.to_csv(output_csv_path, index=False)

        # Save categorical mappings
        output_json_path = processed_dir / "categorical_mappings.json"
        print(f"Saving categorical mappings to {output_json_path}...")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(mappings, f, indent=4)

        print("\n--- Processing Summary ---")
        print(f"Total processed rows: {processed_df.shape[0]}")
        print(f"Total processed columns: {processed_df.shape[1]}")
        print(f"Columns: {list(processed_df.columns)}")
        print(f"Missing values count:\n{processed_df.isna().sum()}")
        print("Preprocessing task completed successfully!")

    except Exception as e:
        print(f"Error during preprocessing pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
