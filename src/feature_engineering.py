"""
src/feature_engineering.py
Creates time-based and statistical features from cleaned climate data.
"""

import pandas as pd
import numpy as np


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract year, month, day, season, decade from date column.
    """
    df = df.copy()
    
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    
    # Season mapping (Northern Hemisphere)
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn"
    }
    df["season"] = df["month"].map(season_map)
    
    # Decade
    df["decade"] = (df["year"] // 10) * 10
    df["decade_label"] = df["decade"].astype(str) + "s"
    
    print("[FEATURES] Temporal features added: year, month, season, decade, etc.")
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling means and standard deviations for temperature and rainfall.
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    
    # 7-day rolling mean/std (short-term weather pattern)
    df["temp_7d_mean"] = df["temperature_c"].rolling(window=7, center=True, min_periods=1).mean().round(2)
    df["temp_30d_mean"] = df["temperature_c"].rolling(window=30, center=True, min_periods=1).mean().round(2)
    df["rain_7d_sum"] = df["rainfall_mm"].rolling(window=7, center=True, min_periods=1).sum().round(2)
    df["rain_30d_sum"] = df["rainfall_mm"].rolling(window=30, center=True, min_periods=1).sum().round(2)
    
    print("[FEATURES] Rolling features added: 7-day, 30-day means and sums.")
    return df


def add_monthly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add monthly and yearly aggregates as separate dataframes.
    Returns the original df unchanged — aggregates are returned separately.
    """
    # Monthly aggregates
    monthly = df.groupby(["year", "month"]).agg(
        temp_mean=("temperature_c", "mean"),
        temp_max=("temperature_c", "max"),
        temp_min=("temperature_c", "min"),
        rainfall_total=("rainfall_mm", "sum"),
        humidity_mean=("humidity_pct", "mean"),
        wind_mean=("wind_speed_kmh", "mean")
    ).reset_index().round(2)
    
    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str) + "-01"
    )
    
    # Yearly aggregates
    yearly = df.groupby("year").agg(
        temp_mean=("temperature_c", "mean"),
        temp_max=("temperature_c", "max"),
        temp_min=("temperature_c", "min"),
        rainfall_total=("rainfall_mm", "sum"),
        humidity_mean=("humidity_pct", "mean"),
        wind_mean=("wind_speed_kmh", "mean")
    ).reset_index().round(2)
    
    print("[FEATURES] Monthly and yearly aggregates computed.")
    return monthly, yearly


def engineer_all_features(df: pd.DataFrame):
    """
    Master function — applies all feature engineering steps.
    Returns: df (daily), monthly_df, yearly_df
    """
    df = add_temporal_features(df)
    df = add_rolling_features(df)
    monthly_df, yearly_df = add_monthly_aggregates(df)
    
    print(f"\n[COMPLETE] Feature engineering done.")
    print(f"  Daily records    : {len(df):,}")
    print(f"  Monthly records  : {len(monthly_df):,}")
    print(f"  Yearly records   : {len(yearly_df):,}")
    
    return df, monthly_df, yearly_df


if __name__ == "__main__":
    from data_loader import load_and_prepare
    df = load_and_prepare()
    df, monthly, yearly = engineer_all_features(df)
    print("\n[YEARLY SUMMARY]")
    print(yearly.head(10))