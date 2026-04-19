"""
tests/test_pipeline.py
Basic unit tests for the climate pipeline.
Run with: python -m pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))


def test_generate_dataset():
    """Test that synthetic dataset generates correctly."""
    from generate_dataset import generate_climate_data
    df = generate_climate_data(
        start_date="2000-01-01",
        end_date="2005-12-31",
        output_path="tests/test_data.csv"
    )
    assert len(df) > 0, "Dataset should not be empty"
    assert "date" in df.columns
    assert "temperature_c" in df.columns
    assert "rainfall_mm" in df.columns
    print("[PASS] generate_dataset test")


def test_data_loader():
    """Test data loading and cleaning."""
    from generate_dataset import generate_climate_data
    from data_loader import load_and_prepare
    
    generate_climate_data(
        start_date="2000-01-01",
        end_date="2002-12-31",
        output_path="tests/test_raw.csv"
    )
    
    df = load_and_prepare(
        raw_path="tests/test_raw.csv",
        processed_path="tests/test_clean.csv"
    )
    
    assert df.isnull().sum().sum() == 0, "No missing values after cleaning"
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    print("[PASS] data_loader test")


def test_feature_engineering():
    """Test feature engineering outputs."""
    from generate_dataset import generate_climate_data
    from data_loader import load_and_prepare
    from feature_engineering import engineer_all_features
    
    generate_climate_data("2000-01-01", "2002-12-31", "tests/fe_raw.csv")
    df = load_and_prepare("tests/fe_raw.csv", "tests/fe_clean.csv")
    df, monthly, yearly = engineer_all_features(df)
    
    assert "year" in df.columns
    assert "season" in df.columns
    assert "temp_7d_mean" in df.columns
    assert len(monthly) > 0
    assert len(yearly) > 0
    print("[PASS] feature_engineering test")


def test_anomaly_detection():
    """Test anomaly detection returns boolean flags."""
    from generate_dataset import generate_climate_data
    from data_loader import load_and_prepare
    from feature_engineering import engineer_all_features
    from anomaly_detection import detect_all_anomalies
    
    generate_climate_data("2000-01-01", "2005-12-31", "tests/anom_raw.csv")
    df = load_and_prepare("tests/anom_raw.csv", "tests/anom_clean.csv")
    df, monthly, yearly = engineer_all_features(df)
    df = detect_all_anomalies(df)
    
    assert "is_anomaly" in df.columns
    assert df["is_anomaly"].dtype == bool
    assert df["is_anomaly"].sum() > 0, "Should detect some anomalies"
    print("[PASS] anomaly_detection test")


def test_zscore_function():
    """Test Z-score anomaly function."""
    from anomaly_detection import zscore_anomaly
    
    s = pd.Series([20.0, 21.0, 22.0, 21.5, 50.0])  # 50 is clear outlier
    result = zscore_anomaly(s, threshold=2.0)
    assert result.iloc[-1] == True, "50.0 should be detected as anomaly"
    print("[PASS] zscore_function test")


if __name__ == "__main__":
    test_generate_dataset()
    test_data_loader()
    test_feature_engineering()
    test_anomaly_detection()
    test_zscore_function()
    
    # Cleanup test files
    import glob
    for f in glob.glob("tests/*.csv"):
        os.remove(f)
    
    print("\n✅ All tests passed!")