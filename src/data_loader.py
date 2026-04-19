"""
src/data_loader.py
Handles loading and cleaning of raw climate data.
"""

import pandas as pd
import numpy as np
import os


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV climate data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath, parse_dates=["date"])
    print(f"[INFO] Loaded data shape: {df.shape}")
    print(f"[INFO] Date range: {df['date'].min()} → {df['date'].max()}")
    return df


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Report missing values, dtype issues, and basic stats.
    Returns a quality report dictionary.
    """
    report = {}
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report["missing_values"] = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct
    })
    
    # Data types
    report["dtypes"] = df.dtypes
    
    # Basic stats
    report["stats"] = df.describe()
    
    print("\n[DATA QUALITY REPORT]")
    print(report["missing_values"])
    print(f"\nDate dtype: {df['date'].dtype}")
    
    return report


def clean_data(df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    1. Ensure correct dtypes
    2. Handle missing values via linear interpolation
    3. Clip physically impossible values
    4. Sort by date
    5. Reset index
    """
    df = df.copy()
    
    # ── 1. Parse dates if not already datetime ──
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    
    # ── 2. Sort by date ──
    df = df.sort_values("date").reset_index(drop=True)
    
    # ── 3. Log missing values before cleaning ──
    print(f"[INFO] Missing values BEFORE cleaning:\n{df.isnull().sum()}")
    
    # ── 4. Interpolate missing values ──
    numeric_cols = ["temperature_c", "rainfall_mm", "humidity_pct", "wind_speed_kmh"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
    
    # ── 5. Clip unrealistic values ──
    df["temperature_c"] = df["temperature_c"].clip(-20, 55)   # Physical bounds
    df["rainfall_mm"] = df["rainfall_mm"].clip(0, 500)
    df["humidity_pct"] = df["humidity_pct"].clip(0, 100)
    df["wind_speed_kmh"] = df["wind_speed_kmh"].clip(0, 200)
    
    # ── 6. Confirm no missing values remain ──
    remaining_nulls = df.isnull().sum().sum()
    print(f"[INFO] Missing values AFTER cleaning: {remaining_nulls}")
    
    # ── 7. Save if path given ──
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[SAVED] Cleaned data → {save_path}")
    
    return df


def load_and_prepare(
    raw_path: str = "data/raw/climate_raw.csv",
    processed_path: str = "data/processed/climate_cleaned.csv"
) -> pd.DataFrame:
    """Master function: load, check, clean, and return clean dataframe."""
    df_raw = load_raw_data(raw_path)
    check_data_quality(df_raw)
    df_clean = clean_data(df_raw, save_path=processed_path)
    return df_clean


if __name__ == "__main__":
    df = load_and_prepare()
    print("\n[CLEANED DATA SAMPLE]")
    print(df.head())
    print(df.tail())