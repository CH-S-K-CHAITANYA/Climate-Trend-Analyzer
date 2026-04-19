"""
generate_dataset.py
Generates a synthetic climate dataset simulating 34 years of daily
temperature, rainfall, humidity, and wind speed data for a fictional city.
Includes realistic trends, seasonality, and injected anomalies.
"""

import pandas as pd
import numpy as np
import os

def generate_climate_data(
    start_date="1990-01-01",
    end_date="2023-12-31",
    seed=42,
    output_path="data/raw/climate_raw.csv"
):
    """
    Generate synthetic daily climate data.
    
    Includes:
    - Long-term warming trend (temperature rises ~1.5°C over 34 years)
    - Seasonal cycles (sinusoidal pattern)
    - Random noise
    - Injected anomalies (heat waves, drought years)
    """
    np.random.seed(seed)
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)
    
    print(f"[INFO] Generating {n} daily records from {start_date} to {end_date}...")
    
    # ─── TEMPERATURE ──────────────────────────────────────────────────────
    # Base seasonal cycle (peaks in summer ~June-July for Northern Hemisphere)
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    seasonal_temp = 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    # Long-term warming trend: +1.5°C over 34 years
    years_elapsed = np.array([(d.year - 1990) + d.timetuple().tm_yday / 365 for d in dates])
    warming_trend = 1.5 * (years_elapsed / 34)
    
    # Random noise
    noise_temp = np.random.normal(0, 2.5, n)
    
    # Base temperature
    temperature = 22 + seasonal_temp + warming_trend + noise_temp
    
    # ─── RAINFALL ─────────────────────────────────────────────────────────
    # Monsoon pattern: higher rainfall June-September
    rainfall_seasonal = np.clip(
        5 * np.sin(2 * np.pi * (day_of_year - 150) / 365) + 3, 0, None
    )
    rainfall_noise = np.abs(np.random.gamma(2, 2, n))
    rainfall = rainfall_seasonal + rainfall_noise
    
    # Long-term rainfall decline: -10% over 34 years
    rainfall_trend = 1 - 0.10 * (years_elapsed / 34)
    rainfall = rainfall * rainfall_trend
    
    # ─── HUMIDITY ─────────────────────────────────────────────────────────
    humidity_seasonal = 20 * np.sin(2 * np.pi * (day_of_year - 160) / 365) + 60
    humidity_noise = np.random.normal(0, 5, n)
    humidity = np.clip(humidity_seasonal + humidity_noise, 20, 100)
    
    # ─── WIND SPEED ───────────────────────────────────────────────────────
    wind_speed = np.abs(np.random.normal(12, 5, n))
    
    # ─── INJECT ANOMALIES ─────────────────────────────────────────────────
    # Heat wave: Summer 1998 (El Niño year) — +6°C spike
    mask_1998 = (pd.DatetimeIndex(dates).year == 1998) & \
                (pd.DatetimeIndex(dates).month.isin([5, 6, 7]))
    temperature[mask_1998] += np.random.uniform(4, 7, mask_1998.sum())
    
    # Extreme heat: 2015 June — +8°C spike
    mask_2015 = (pd.DatetimeIndex(dates).year == 2015) & \
                (pd.DatetimeIndex(dates).month == 6)
    temperature[mask_2015] += np.random.uniform(5, 9, mask_2015.sum())
    
    # Drought year 2002: rainfall drops to near zero March-May
    mask_drought = (pd.DatetimeIndex(dates).year == 2002) & \
                   (pd.DatetimeIndex(dates).month.isin([3, 4, 5]))
    rainfall[mask_drought] *= 0.1
    
    # Cold snap: December 2010
    mask_cold = (pd.DatetimeIndex(dates).year == 2010) & \
                (pd.DatetimeIndex(dates).month == 12)
    temperature[mask_cold] -= np.random.uniform(3, 6, mask_cold.sum())
    
    # Heavy rain anomaly: 2017 August
    mask_rain = (pd.DatetimeIndex(dates).year == 2017) & \
                (pd.DatetimeIndex(dates).month == 8)
    rainfall[mask_rain] *= 3.5
    
    # ─── INTRODUCE MISSING VALUES (realistic ~2%) ──────────────────────────
    missing_idx = np.random.choice(n, size=int(0.02 * n), replace=False)
    temperature[missing_idx[:len(missing_idx)//4]] = np.nan
    rainfall[missing_idx[len(missing_idx)//4:len(missing_idx)//2]] = np.nan
    humidity[missing_idx[len(missing_idx)//2:3*len(missing_idx)//4]] = np.nan
    
    # ─── ASSEMBLE DATAFRAME ────────────────────────────────────────────────
    df = pd.DataFrame({
        "date": dates,
        "temperature_c": np.round(temperature, 2),
        "rainfall_mm": np.round(np.clip(rainfall, 0, None), 2),
        "humidity_pct": np.round(humidity, 1),
        "wind_speed_kmh": np.round(wind_speed, 1)
    })
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Dataset saved to: {output_path}")
    print(f"[INFO] Shape: {df.shape}")
    print(f"\n{df.head()}")
    return df


if __name__ == "__main__":
    df = generate_climate_data()
    print("\n[DATASET STATS]")
    print(df.describe())