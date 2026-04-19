"""
main.py
Master pipeline for Climate Trend Analyzer.
Runs all modules end-to-end:
  generate → load → clean → engineer → trend → anomaly → forecast → visualize
"""

import os
import sys
import time
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

from data_loader import load_and_prepare
from feature_engineering import engineer_all_features
from trend_analysis import run_full_trend_analysis
from anomaly_detection import detect_all_anomalies, get_anomaly_report
from forecasting import run_full_forecasting
from visualization import generate_all_plots

# Create output directories
for d in ["data/raw", "data/processed", "outputs/figures", "outputs/tables", "outputs/reports", "models"]:
    os.makedirs(d, exist_ok=True)


def main():
    print("=" * 70)
    print("   🌍  CLIMATE TREND ANALYZER — FULL PIPELINE")
    print("=" * 70)
    start_time = time.time()
    
    # ── STEP 1: Generate / Load Data ──────────────────────────────────────────
    print("\n[STEP 1] Generating synthetic climate dataset...")
    from generate_dataset import generate_climate_data
    generate_climate_data()
    
    # ── STEP 2: Load and Clean ─────────────────────────────────────────────────
    print("\n[STEP 2] Loading and cleaning data...")
    df = load_and_prepare(
        raw_path="data/raw/climate_raw.csv",
        processed_path="data/processed/climate_cleaned.csv"
    )
    
    # ── STEP 3: Feature Engineering ───────────────────────────────────────────
    print("\n[STEP 3] Engineering features...")
    df, monthly_df, yearly_df = engineer_all_features(df)
    
    # ── STEP 4: Trend Analysis ─────────────────────────────────────────────────
    print("\n[STEP 4] Running trend analysis...")
    trend_results = run_full_trend_analysis(yearly_df, monthly_df)
    decadal_df = trend_results["decadal"]
    stl_temp = trend_results["stl_temp"]
    stl_rain = trend_results["stl_rain"]
    
    # Save decadal summary
    decadal_df.to_csv("outputs/tables/decadal_summary.csv", index=False)
    
    # ── STEP 5: Anomaly Detection ──────────────────────────────────────────────
    print("\n[STEP 5] Detecting climate anomalies...")
    df = detect_all_anomalies(df)
    anomaly_report = get_anomaly_report(df)
    anomaly_report.to_csv("outputs/tables/anomaly_report.csv", index=False)
    
    # ── STEP 6: Forecasting ────────────────────────────────────────────────────
    print("\n[STEP 6] Running forecasting models...")
    forecast_results = run_full_forecasting(yearly_df, monthly_df)
    
    if "arima_temp" in forecast_results:
        forecast_results["arima_temp"].to_csv("outputs/tables/arima_temp_forecast.csv", index=False)
    if "arima_rain" in forecast_results:
        forecast_results["arima_rain"].to_csv("outputs/tables/arima_rain_forecast.csv", index=False)
    
    # ── STEP 7: Visualization ──────────────────────────────────────────────────
    print("\n[STEP 7] Generating all visualizations...")
    generate_all_plots(df, monthly_df, yearly_df, decadal_df,
                       stl_temp, stl_rain, forecast_results)
    
    # ── STEP 8: Final Summary ──────────────────────────────────────────────────
    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print("   ✅  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"   Total runtime         : {elapsed}s")
    print(f"   Daily records         : {len(df):,}")
    print(f"   Anomalies detected    : {df['is_anomaly'].sum()}")
    print(f"   Charts saved          : outputs/figures/")
    print(f"   Tables saved          : outputs/tables/")
    print(f"   Dashboard             : outputs/reports/interactive_dashboard.html")
    print("=" * 70)
    
    # Print key insights
    trend_stats = trend_results["trend_stats"]
    print("\n📊 KEY CLIMATE INSIGHTS")
    print("-" * 40)
    print(f"  Temperature trend    : +{trend_stats['temp_mean']['trend_per_decade']:.3f}°C per decade")
    print(f"  Total warming (34yr) : +{trend_stats['temp_mean']['total_change']:.2f}°C")
    print(f"  Rainfall trend       : {trend_stats['rainfall_total']['trend_per_decade']:.1f} mm per decade")
    mk = trend_results["mann_kendall"]
    if mk.get("temperature"):
        print(f"  Temp trend (MK test) : {mk['temperature']['trend'].upper()} (p={mk['temperature']['p_value']})")
    print("-" * 40)


if __name__ == "__main__":
    main()