"""
src/anomaly_detection.py
Detects climate anomalies using three methods:
1. Z-Score (statistical deviation)
2. IQR (interquartile range)
3. Isolation Forest (machine learning)
Combines all methods into a consensus flag.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def zscore_anomaly(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Flag anomalies where |Z-score| > threshold (default 3σ).
    Returns boolean Series: True = anomaly
    """
    mean = series.mean()
    std = series.std()
    z_scores = np.abs((series - mean) / std)
    return z_scores > threshold


def iqr_anomaly(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Flag anomalies outside [Q1 - k*IQR, Q3 + k*IQR].
    Returns boolean Series: True = anomaly
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (series < lower_bound) | (series > upper_bound)


def isolation_forest_anomaly(df: pd.DataFrame, features: list, contamination: float = 0.03) -> pd.Series:
    """
    Isolation Forest for multivariate anomaly detection.
    contamination: expected fraction of anomalies (3% of data)
    Returns boolean Series: True = anomaly
    """
    X = df[features].dropna()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    preds = iso_forest.fit_predict(X_scaled)
    
    # -1 = anomaly, 1 = normal
    anomaly_flag = pd.Series(preds == -1, index=X.index)
    return anomaly_flag


def detect_all_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all three anomaly detection methods on daily climate data.
    Adds anomaly flag columns and a consensus column.
    """
    df = df.copy()
    
    # ── Z-Score anomalies ──────────────────────────────────────────
    df["anomaly_zscore_temp"] = zscore_anomaly(df["temperature_c"], threshold=3.0)
    df["anomaly_zscore_rain"] = zscore_anomaly(df["rainfall_mm"], threshold=3.0)
    
    # ── IQR anomalies ──────────────────────────────────────────────
    df["anomaly_iqr_temp"] = iqr_anomaly(df["temperature_c"], multiplier=2.5)
    df["anomaly_iqr_rain"] = iqr_anomaly(df["rainfall_mm"], multiplier=2.5)
    
    # ── Isolation Forest (multivariate) ────────────────────────────
    iso_cols = ["temperature_c", "rainfall_mm", "humidity_pct", "wind_speed_kmh"]
    iso_anomaly = isolation_forest_anomaly(df, iso_cols, contamination=0.03)
    df["anomaly_iforest"] = False
    df.loc[iso_anomaly.index, "anomaly_iforest"] = iso_anomaly
    
    # ── Consensus: flagged by at least 2 methods ───────────────────
    df["anomaly_vote"] = (
        df["anomaly_zscore_temp"].astype(int) +
        df["anomaly_iqr_temp"].astype(int) +
        df["anomaly_iforest"].astype(int)
    )
    df["is_anomaly"] = df["anomaly_vote"] >= 1
    
    # Summary
    total_anomalies = df["is_anomaly"].sum()
    pct = total_anomalies / len(df) * 100
    print(f"\n[ANOMALY DETECTION SUMMARY]")
    print(f"  Z-Score temperature anomalies : {df['anomaly_zscore_temp'].sum()}")
    print(f"  IQR temperature anomalies     : {df['anomaly_iqr_temp'].sum()}")
    print(f"  Isolation Forest anomalies    : {df['anomaly_iforest'].sum()}")
    print(f"  Consensus anomalies (≥2 votes): {total_anomalies} ({pct:.2f}% of data)")
    
    return df


def get_anomaly_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and format all detected anomalies into a readable report.
    """
    anomalies = df[df["is_anomaly"] == True][
        ["date", "year", "month", "temperature_c", "rainfall_mm",
         "humidity_pct", "anomaly_vote", "season"]
    ].copy()
    
    anomalies = anomalies.sort_values("date")
    anomalies["severity"] = anomalies["anomaly_vote"].map({2: "High", 3: "Extreme"})
    
    print(f"\n[ANOMALY REPORT] Total: {len(anomalies)} events")
    print(anomalies.head(10).to_string(index=False))
    
    return anomalies


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_loader import load_and_prepare
    from src.feature_engineering import engineer_all_features
    
    df = load_and_prepare()
    df, monthly, yearly = engineer_all_features(df)
    df = detect_all_anomalies(df)
    anomaly_report = get_anomaly_report(df)
    
    # Save anomaly report
    anomaly_report.to_csv("outputs/tables/anomaly_report.csv", index=False)
    print("\n[SAVED] anomaly_report.csv")