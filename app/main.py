"""FastAPI backend application for Retail Demand Forecasting.

This script runs the web server, pre-calculates sales forecasts on the test
dataset at startup, and defines router endpoints for the dashboard,
restock predictor form, and model evaluation metrics pages.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Initialize FastAPI App
app = FastAPI(title="Retail Demand Forecasting Dashboard")

# Paths Configuration
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Ensure static directory exists
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Global variables to hold model, mappings, and pre-calculated predictions
model = None
categorical_mappings = {}
test_predictions_df = None
available_families = []
available_dates = []


def preprocess_test_data(
    test_df: pd.DataFrame,
    oil_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    mappings: dict
) -> pd.DataFrame:
    """Preprocess and merge the test dataset using the same training pipeline logic.

    Args:
        test_df: Raw test dataset (Aug 16 - Aug 31, 2017).
        oil_df: Oil price dataset.
        holidays_df: Holidays and events dataset.
        stores_df: Store metadata.
        mappings: Categorical mapping dictionaries.

    Returns:
        Preprocessed test DataFrame with original name columns preserved.
    """
    test_df = test_df.copy()
    test_df["date"] = test_df["date"].astype(str)

    # Keep original names for display in frontend
    test_df["family_name"] = test_df["family"]

    # 1. Merge stores metadata to get state and city
    test_df = test_df.merge(stores_df, on="store_nbr", how="left")
    # Preserve original store characteristics for frontend display
    test_df["city_name"] = test_df["city"]
    test_df["state_name"] = test_df["state"]
    test_df["type_name"] = test_df["type"]

    # 2. Merge oil prices
    oil_df = oil_df.copy()
    oil_df["date"] = oil_df["date"].astype(str)
    test_df = test_df.merge(oil_df, on="date", how="left")

    # Fill missing oil prices (ffill, then bfill)
    test_df = test_df.sort_values(by=["date", "store_nbr", "family"]).reset_index(drop=True)
    test_df["dcoilwtico"] = test_df["dcoilwtico"].ffill().bfill()

    # 3. Merge holidays
    holidays_df["date"] = holidays_df["date"].astype(str)
    holidays_filtered = holidays_df[
        (holidays_df["transferred"].astype(str).str.lower() != "true") &
        (holidays_df["type"] != "Work Day")
    ].copy()

    national = holidays_filtered[holidays_filtered["locale"] == "National"][["date"]].drop_duplicates()
    national["is_holiday_nat"] = 1

    regional = holidays_filtered[holidays_filtered["locale"] == "Regional"][["date", "locale_name"]].drop_duplicates()
    regional = regional.rename(columns={"locale_name": "state"})
    regional["is_holiday_reg"] = 1

    local = holidays_filtered[holidays_filtered["locale"] == "Local"][["date", "locale_name"]].drop_duplicates()
    local = local.rename(columns={"locale_name": "city"})
    local["is_holiday_loc"] = 1

    test_df = test_df.merge(national, on="date", how="left")
    test_df = test_df.merge(regional, on=["date", "state"], how="left")
    test_df = test_df.merge(local, on=["date", "city"], how="left")

    test_df["is_holiday"] = (
        (test_df["is_holiday_nat"] == 1) |
        (test_df["is_holiday_reg"] == 1) |
        (test_df["is_holiday_loc"] == 1)
    ).astype(int)

    test_df = test_df.drop(columns=["is_holiday_nat", "is_holiday_reg", "is_holiday_loc"])

    # 4. Extract temporal features
    date_col = pd.to_datetime(test_df["date"])
    test_df["month"] = date_col.dt.month
    test_df["day"] = date_col.dt.day
    test_df["dayofweek"] = date_col.dt.dayofweek

    # 5. Apply categorical encoding mappings
    for col in ["family", "city", "state", "type"]:
        test_df[col] = test_df[col].map(mappings[col])

    return test_df


@app.on_event("startup")
def startup_event():
    """Startup routine to load files and generate predictions on the test set."""
    global model, categorical_mappings, test_predictions_df, available_families, available_dates

    model_path = BASE_DIR / "best_model.pkl"
    mapping_path = PROJECT_ROOT / "data" / "processed" / "categorical_mappings.json"
    raw_dir = PROJECT_ROOT / "data" / "raw"

    print("Startup: Initializing application databases...")

    # Check model file existence
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train_model.py first.")

    # Check mapping file existence
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found at {mapping_path}. Please run data_preprocessing.py first.")

    # Load Model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Startup: Model loaded successfully.")

    # Load Categorical Mappings
    with open(mapping_path, "r", encoding="utf-8") as f:
        categorical_mappings = json.load(f)
    print("Startup: Categorical mappings loaded.")

    # Load raw data required to build exogenous features for test predictions
    test_df = pd.read_csv(raw_dir / "test.csv")
    oil_df = pd.read_csv(raw_dir / "oil.csv")
    holidays_df = pd.read_csv(raw_dir / "holidays_events.csv")
    stores_df = pd.read_csv(raw_dir / "stores.csv")

    # Map the unique categories list for dropdowns
    # Sort family names alphabetically
    available_families = sorted(test_df["family"].unique())
    available_dates = sorted(test_df["date"].unique())

    # Preprocess test features
    print("Startup: Preprocessing test dataset...")
    preprocessed_test = preprocess_test_data(test_df, oil_df, holidays_df, stores_df, categorical_mappings)

    # Select features in the correct order as trained
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
    X_test = preprocessed_test[feature_cols]

    # Generate predictions
    print("Startup: Running forecast predictions for test set...")
    preds = model.predict(X_test)
    preprocessed_test["predicted_sales"] = np.clip(preds, 0, None)

    # Store final predictions globally
    test_predictions_df = preprocessed_test
    print(f"Startup: Successfully pre-calculated {len(test_predictions_df)} sales predictions.")


@app.get("/", response_class=HTMLResponse)
def read_dashboard(request: Request):
    """Dashboard homepage displaying high-level forecast summaries."""
    global test_predictions_df

    # Total predicted sales
    total_sales = float(test_predictions_df["predicted_sales"].sum())

    # Top store by sales
    store_sales = test_predictions_df.groupby(["store_nbr", "city_name"])["predicted_sales"].sum().reset_index()
    top_store_row = store_sales.sort_values(by="predicted_sales", ascending=False).iloc[0]
    top_store = {
        "store_nbr": int(top_store_row["store_nbr"]),
        "city_name": str(top_store_row["city_name"]),
        "sales": float(top_store_row["predicted_sales"])
    }

    # Top category by sales
    cat_sales = test_predictions_df.groupby("family_name")["predicted_sales"].sum().reset_index()
    top_cat_row = cat_sales.sort_values(by="predicted_sales", ascending=False).iloc[0]
    top_category = {
        "family_name": str(top_cat_row["family_name"]),
        "sales": float(top_cat_row["predicted_sales"])
    }

    # Top 5 Categories & Shares
    cat_sales_sorted = cat_sales.sort_values(by="predicted_sales", ascending=False)
    top_5_categories = []
    for _, row in cat_sales_sorted.head(5).iterrows():
        top_5_categories.append({
            "name": str(row["family_name"]),
            "sales": float(row["predicted_sales"]),
            "share": float((row["predicted_sales"] / total_sales) * 100) if total_sales > 0 else 0.0
        })

    # Top 5 Stores
    store_sales_sorted = store_sales.sort_values(by="predicted_sales", ascending=False)
    # Join with store state
    store_metadata = test_predictions_df[["store_nbr", "city_name", "state_name"]].drop_duplicates().set_index("store_nbr")
    top_5_stores = []
    for _, row in store_sales_sorted.head(5).iterrows():
        store_nbr = int(row["store_nbr"])
        top_5_stores.append({
            "store_nbr": store_nbr,
            "city": str(row["city_name"]),
            "state": str(store_metadata.loc[store_nbr, "state_name"]),
            "sales": float(row["predicted_sales"])
        })

    # Daily Forecast Trend (Chart.js)
    daily_sales = test_predictions_df.groupby("date")["predicted_sales"].sum().sort_index().reset_index()
    # Format labels e.g. "16 Aug"
    daily_sales["label"] = pd.to_datetime(daily_sales["date"]).dt.strftime("%d %b")
    chart_labels = daily_sales["label"].tolist()
    chart_data = daily_sales["predicted_sales"].tolist()

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "active_page": "dashboard",
            "total_sales": total_sales,
            "top_store": top_store,
            "top_category": top_category,
            "top_5_categories": top_5_categories,
            "top_5_stores": top_5_stores,
            "chart_labels": chart_labels,
            "chart_data": chart_data
        }
    )


@app.get("/predict", response_class=HTMLResponse)
def read_predict_form(request: Request):
    """Render prediction form with default initial prediction results."""
    global available_families, available_dates, test_predictions_df

    default_family = "BEVERAGES"
    default_date = "2017-08-16"

    # Prepopulate default list
    filtered_df = test_predictions_df[
        (test_predictions_df["family_name"] == default_family) &
        (test_predictions_df["date"] == default_date)
    ].copy()

    # Sort by store number
    filtered_df = filtered_df.sort_values(by="store_nbr")

    # Compute key stats for default display
    total_projected = float(filtered_df["predicted_sales"].sum())
    max_row = filtered_df.sort_values(by="predicted_sales", ascending=False).iloc[0]
    max_store = {
        "store_nbr": int(max_row["store_nbr"]),
        "sales": float(max_row["predicted_sales"])
    }
    oil_price = float(filtered_df.iloc[0]["dcoilwtico"])
    is_holiday = int(filtered_df.iloc[0]["is_holiday"]) == 1

    predictions = []
    for _, row in filtered_df.iterrows():
        predictions.append({
            "store_nbr": int(row["store_nbr"]),
            "city": str(row["city_name"]),
            "state": str(row["state_name"]),
            "type_name": str(row["type_name"]),
            "sales": float(row["predicted_sales"])
        })

    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "active_page": "predict",
            "families": available_families,
            "available_dates": available_dates,
            "selected_family": default_family,
            "selected_date": default_date,
            "predictions": predictions,
            "total_projected_sales": total_projected,
            "max_sales_store": max_store,
            "oil_price": oil_price,
            "is_holiday": is_holiday
        }
    )


@app.post("/predict", response_class=HTMLResponse)
def post_predict_form(
    request: Request,
    family: str = Form(...),
    date: str = Form(...)
):
    """Handle form submission, filter prediction database, and show recommendations."""
    global available_families, available_dates, test_predictions_df

    # Filter by selected parameters
    filtered_df = test_predictions_df[
        (test_predictions_df["family_name"] == family) &
        (test_predictions_df["date"] == date)
    ].copy()

    filtered_df = filtered_df.sort_values(by="store_nbr")

    # Compute statistics for page banner
    total_projected = float(filtered_df["predicted_sales"].sum())
    max_store = {"store_nbr": -1, "sales": 0.0}
    oil_price = 0.0
    is_holiday = False

    predictions = []
    if not filtered_df.empty:
        max_row = filtered_df.sort_values(by="predicted_sales", ascending=False).iloc[0]
        max_store = {
            "store_nbr": int(max_row["store_nbr"]),
            "sales": float(max_row["predicted_sales"])
        }
        oil_price = float(filtered_df.iloc[0]["dcoilwtico"])
        is_holiday = int(filtered_df.iloc[0]["is_holiday"]) == 1

        for _, row in filtered_df.iterrows():
            predictions.append({
                "store_nbr": int(row["store_nbr"]),
                "city": str(row["city_name"]),
                "state": str(row["state_name"]),
                "type_name": str(row["type_name"]),
                "sales": float(row["predicted_sales"])
            })

    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "active_page": "predict",
            "families": available_families,
            "available_dates": available_dates,
            "selected_family": family,
            "selected_date": date,
            "predictions": predictions,
            "total_projected_sales": total_projected,
            "max_sales_store": max_store,
            "oil_price": oil_price,
            "is_holiday": is_holiday
        }
    )


@app.get("/evaluation", response_class=HTMLResponse)
def read_evaluation(request: Request):
    """Render static model validation metrics and performance analysis page."""
    return templates.TemplateResponse(
        request=request,
        name="evaluation.html",
        context={
            "active_page": "evaluation"
        }
    )
