"""
src/forecasting.py
Climate forecasting using:
1. ARIMA (statsmodels) — classical time series model
2. Facebook Prophet — handles seasonality and trend shifts well
Forecasts annual temperature, rainfall, humidity, and wind speed 5 years ahead.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ── ARIMA imports ──
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# ── Prophet imports ──
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("[WARN] Prophet not installed. Prophet forecasting will be skipped.")


def check_stationarity(series: pd.Series, variable_name: str = "") -> dict:
    """
    Augmented Dickey-Fuller test for stationarity.
    A stationary series is needed for ARIMA.
    """
    result = adfuller(series.dropna())
    adf_stat = result[0]
    p_value = result[1]
    is_stationary = p_value < 0.05
    
    print(f"\n[STATIONARITY TEST] {variable_name}")
    print(f"  ADF Statistic : {adf_stat:.4f}")
    print(f"  P-value       : {p_value:.4f}")
    print(f"  Stationary    : {'YES' if is_stationary else 'NO - differencing needed'}")
    
    return {"adf_stat": adf_stat, "p_value": p_value, "is_stationary": is_stationary}


def arima_forecast(yearly_df: pd.DataFrame, variable: str, forecast_years: int = 5) -> pd.DataFrame:
    """
    Fit ARIMA model on yearly data and forecast next N years.
    
    Auto-selects d based on stationarity test.
    Uses ARIMA(1,1,1) as a reasonable default.
    """
    series = yearly_df[variable].values
    years = yearly_df["year"].values
    
    # Check stationarity
    stat = check_stationarity(yearly_df[variable], variable)
    d = 0 if stat["is_stationary"] else 1
    
    # Fit ARIMA
    model = ARIMA(series, order=(1, d, 1))
    fitted = model.fit()
    
    print(f"\n[ARIMA({1},{d},{1})] {variable}")
    print(f"  AIC: {fitted.aic:.2f}")
    
    # Forecast
    forecast = fitted.get_forecast(steps=forecast_years)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05)
    
    future_years = np.arange(years.max() + 1, years.max() + forecast_years + 1)
    
    result_df = pd.DataFrame({
        "year": future_years,
        "forecast": np.round(forecast_mean, 3),
        "lower_95": np.round(conf_int[:, 0], 3),
        "upper_95": np.round(conf_int[:, 1], 3),
        "model": "ARIMA"
    })
    
    print(f"\n[ARIMA FORECAST] {variable} — Next {forecast_years} years:")
    print(result_df.to_string(index=False))
    
    return result_df


def prophet_forecast(monthly_df: pd.DataFrame, variable: str, forecast_months: int = 60) -> pd.DataFrame:
    """
    Fit Prophet model on monthly data and forecast forward.
    Prophet requires columns: ds (date) and y (value)
    """
    if not PROPHET_AVAILABLE:
        print("[SKIP] Prophet not available.")
        return pd.DataFrame(), None, None
    
    # Prepare Prophet-compatible dataframe
    prophet_df = monthly_df[["date", variable]].rename(
        columns={"date": "ds", variable: "y"}
    ).dropna()
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    
    # Initialize and fit Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,  # Controls trend flexibility
        seasonality_prior_scale=10.0,
        interval_width=0.95
    )
    
    model.fit(prophet_df)
    
    # Make future dataframe
    future = model.make_future_dataframe(periods=forecast_months, freq="MS")
    forecast = model.predict(future)
    
    # Extract forecast portion
    forecast_only = forecast[forecast["ds"] > prophet_df["ds"].max()][[
        "ds", "yhat", "yhat_lower", "yhat_upper", "trend"
    ]].rename(columns={
        "ds": "date",
        "yhat": "forecast",
        "yhat_lower": "lower_95",
        "yhat_upper": "upper_95"
    })
    
    forecast_only["model"] = "Prophet"
    
    print(f"\n[PROPHET FORECAST] {variable} — Next {forecast_months} months:")
    print(forecast_only.head(12).to_string(index=False))
    
    return forecast_only, forecast, model


def run_full_forecasting(yearly_df: pd.DataFrame, monthly_df: pd.DataFrame, save_csv: bool = True) -> dict:
    """Master forecasting function that loops through all variables."""
    print("=" * 60)
    print("RUNNING FORECASTING MODULE (ALL VARIABLES)")
    print("=" * 60)
    
    results = {}
    
    # Map the dataframe columns to the exact prefixes the Streamlit app expects
    targets = {
        "temp_mean": "temp",
        "rainfall_total": "rain",
        "humidity_mean": "humidity",
        "wind_mean": "wind"
    }
    
    if save_csv:
        os.makedirs("outputs/tables", exist_ok=True)
        
    for col, prefix in targets.items():
        # --- 1. ARIMA Forecasts (Yearly) ---
        if col in yearly_df.columns:
            arima_fc = arima_forecast(yearly_df, col, forecast_years=5)
            results[f"arima_{prefix}"] = arima_fc
            
            # Save CSV for Streamlit dashboard
            if save_csv:
                out_path = f"outputs/tables/arima_{prefix}_forecast.csv"
                arima_fc.to_csv(out_path, index=False)
                print(f"[SAVE] Wrote ARIMA forecast to {out_path}")
        else:
            print(f"[WARN] Missing column {col} in yearly data. Skipping ARIMA.")

        # --- 2. Prophet Forecasts (Monthly) ---
        if PROPHET_AVAILABLE and col in monthly_df.columns:
            prophet_fc, prophet_full, prophet_model = prophet_forecast(monthly_df, col, 60)
            
            if not prophet_fc.empty:
                results[f"prophet_{prefix}"] = prophet_fc
                results[f"prophet_{prefix}_full"] = prophet_full
                
                # Optional: Save Prophet CSVs too in case you ever want to plot them in Streamlit
                if save_csv:
                    prophet_out_path = f"outputs/tables/prophet_{prefix}_forecast.csv"
                    prophet_fc.to_csv(prophet_out_path, index=False)
                    print(f"[SAVE] Wrote Prophet forecast to {prophet_out_path}")

    return results


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_loader import load_and_prepare
    from src.feature_engineering import engineer_all_features
    
    df = load_and_prepare()
    df, monthly, yearly = engineer_all_features(df)
    
    # Running this directly will now loop through all 4 variables and generate the CSVs
    forecast_results = run_full_forecasting(yearly, monthly, save_csv=True)
    print("\n✅ All forecasting completed and saved successfully!")