"""
src/trend_analysis.py
Performs statistical trend analysis on climate data:
- Yearly temperature and rainfall trends
- Mann-Kendall trend test (statistically validates trend direction)
- STL decomposition (separates trend from seasonality)
- Decadal comparison
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings("ignore")

try:
    import pymannkendall as mk
    MK_AVAILABLE = True
except ImportError:
    MK_AVAILABLE = False
    print("[WARN] pymannkendall not installed. Mann-Kendall test will be skipped.")


def compute_yearly_trends(yearly_df: pd.DataFrame) -> dict:
    """
    Compute linear trends for temperature and rainfall over years.
    Returns slope, intercept, and R² for each variable.
    """
    from numpy.polynomial import polynomial as P
    
    years = yearly_df["year"].values
    results = {}
    
    for col in ["temp_mean", "rainfall_total"]:
        y = yearly_df[col].values
        
        # Fit linear trend
        coeffs = np.polyfit(years, y, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        # R² calculation
        y_pred = np.polyval(coeffs, years)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        results[col] = {
            "slope": round(slope, 4),
            "intercept": round(intercept, 4),
            "r_squared": round(r_squared, 4),
            "trend_per_decade": round(slope * 10, 4),
            "total_change": round(slope * (years.max() - years.min()), 4)
        }
        
        print(f"\n[TREND] {col}:")
        print(f"  Slope (per year)  : {results[col]['slope']}")
        print(f"  Change per decade : {results[col]['trend_per_decade']}")
        print(f"  Total change      : {results[col]['total_change']}")
        print(f"  R²                : {results[col]['r_squared']}")
    
    return results


def mann_kendall_test(series: pd.Series, variable_name: str = "") -> dict:
    """
    Apply Mann-Kendall monotonic trend test.
    H0: No monotonic trend
    If p < 0.05 → statistically significant trend
    """
    if not MK_AVAILABLE:
        print("[SKIP] pymannkendall not available.")
        return {}
    
    result = mk.original_test(series.dropna())
    
    print(f"\n[MANN-KENDALL TEST] Variable: {variable_name}")
    print(f"  Trend     : {result.trend}")
    print(f"  P-value   : {result.p:.4f}")
    print(f"  Tau       : {result.Tau:.4f}")
    print(f"  Significant: {'YES' if result.p < 0.05 else 'NO'}")
    
    return {
        "variable": variable_name,
        "trend": result.trend,
        "p_value": round(result.p, 4),
        "tau": round(result.Tau, 4),
        "significant": result.p < 0.05
    }


def stl_decomposition(monthly_df: pd.DataFrame, variable: str = "temp_mean") -> dict:
    """
    STL (Seasonal and Trend decomposition using Loess) on monthly data.
    Decomposes series into: Trend + Seasonality + Residual
    """
    # Build time-indexed series
    series = monthly_df.set_index("date")[variable].dropna()
    series = series.sort_index()
    series.index = pd.DatetimeIndex(series.index, freq="MS")  # Monthly start freq
    
    # Fit STL
    stl = STL(series, period=12, robust=True)
    result = stl.fit()
    
    decomp_df = pd.DataFrame({
        "date": series.index,
        "observed": result.observed.values,
        "trend": result.trend.values,
        "seasonal": result.seasonal.values,
        "residual": result.resid.values
    })
    
    print(f"\n[STL DECOMPOSITION] Variable: {variable}")
    print(f"  Trend range  : {decomp_df['trend'].min():.2f} → {decomp_df['trend'].max():.2f}")
    print(f"  Seasonal amp : ±{decomp_df['seasonal'].std():.2f}")
    print(f"  Residual std : {decomp_df['residual'].std():.2f}")
    
    return {"decomp_df": decomp_df, "stl_result": result}


def decadal_analysis(yearly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare climate averages across decades.
    """
    yearly_df = yearly_df.copy()
    yearly_df["decade"] = (yearly_df["year"] // 10) * 10
    
    decadal = yearly_df.groupby("decade").agg(
        temp_mean=("temp_mean", "mean"),
        temp_max=("temp_max", "mean"),
        temp_min=("temp_min", "mean"),
        rainfall_total=("rainfall_total", "mean"),
        humidity_mean=("humidity_mean", "mean")
    ).reset_index().round(2)
    
    decadal["decade_label"] = decadal["decade"].astype(str) + "s"
    
    print("\n[DECADAL ANALYSIS]")
    print(decadal.to_string(index=False))
    
    return decadal


def run_full_trend_analysis(yearly_df: pd.DataFrame, monthly_df: pd.DataFrame) -> dict:
    """Master function for trend analysis."""
    print("=" * 60)
    print("RUNNING FULL TREND ANALYSIS")
    print("=" * 60)
    
    # 1. Linear trends
    trend_stats = compute_yearly_trends(yearly_df)
    
    # 2. Mann-Kendall tests
    mk_temp = mann_kendall_test(yearly_df["temp_mean"], "Temperature")
    mk_rain = mann_kendall_test(yearly_df["rainfall_total"], "Rainfall")
    
    # 3. STL decomposition
    stl_temp = stl_decomposition(monthly_df, "temp_mean")
    stl_rain = stl_decomposition(monthly_df, "rainfall_total")
    
    # 4. Decadal analysis
    decadal = decadal_analysis(yearly_df)
    
    return {
        "trend_stats": trend_stats,
        "mann_kendall": {"temperature": mk_temp, "rainfall": mk_rain},
        "stl_temp": stl_temp,
        "stl_rain": stl_rain,
        "decadal": decadal
    }


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_loader import load_and_prepare
    from src.feature_engineering import engineer_all_features
    
    df = load_and_prepare()
    df, monthly, yearly = engineer_all_features(df)
    results = run_full_trend_analysis(yearly, monthly)