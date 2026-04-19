"""
src/visualization.py
Generates publication-quality climate visualization plots.
All charts are saved as PNG files to outputs/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings("ignore")

# ── Style settings ──
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12
})

FIGURES_DIR = "outputs/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def save_fig(filename: str):
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[SAVED] {path}")


# ── PLOT 1: Temperature Over Time ─────────────────────────────────────────────
def plot_temperature_trend(daily_df: pd.DataFrame, yearly_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Daily (light)
    ax.plot(daily_df["date"], daily_df["temp_30d_mean"],
            color="#a8d8ea", linewidth=0.8, label="30-day Rolling Mean", alpha=0.8)
    
    # Yearly mean
    ax.plot(yearly_df["year"], yearly_df["temp_mean"],
            color="#d62728", linewidth=2.5, label="Yearly Mean", marker="o", markersize=4)
    
    # Linear trend line
    z = np.polyfit(yearly_df["year"], yearly_df["temp_mean"], 1)
    p = np.poly1d(z)
    ax.plot(yearly_df["year"], p(yearly_df["year"]),
            "k--", linewidth=1.5, label=f"Trend (+{z[0]*10:.2f}°C/decade)")
    
    ax.set_title("Annual Temperature Trend (1990–2023)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    save_fig("01_temperature_trend.png")


# ── PLOT 2: Rainfall Trend ─────────────────────────────────────────────────────
def plot_rainfall_trend(yearly_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 5))
    
    bars = ax.bar(yearly_df["year"], yearly_df["rainfall_total"],
                  color=plt.cm.Blues(np.linspace(0.3, 0.9, len(yearly_df))),
                  edgecolor="grey", linewidth=0.5, label="Annual Rainfall")
    
    # Trend line
    z = np.polyfit(yearly_df["year"], yearly_df["rainfall_total"], 1)
    p = np.poly1d(z)
    ax.plot(yearly_df["year"], p(yearly_df["year"]),
            "r--", linewidth=2, label=f"Trend ({z[0]*10:.1f} mm/decade)")
    
    ax.set_title("Annual Rainfall Trend (1990–2023)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Annual Rainfall (mm)")
    ax.legend()
    save_fig("02_rainfall_trend.png")


# ── PLOT 3: Seasonal Temperature Boxplot ──────────────────────────────────────
def plot_seasonal_temperature(daily_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    
    sns.boxplot(
        data=daily_df,
        x="season", y="temperature_c",
        order=season_order,
        palette=["#3498db", "#2ecc71", "#e74c3c", "#f39c12"],
        ax=ax,
        width=0.6
    )
    
    ax.set_title("Temperature Distribution by Season (1990–2023)", fontweight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Temperature (°C)")
    save_fig("03_seasonal_temperature.png")


# ── PLOT 4: Monthly Heatmap ───────────────────────────────────────────────────
def plot_monthly_heatmap(monthly_df: pd.DataFrame):
    pivot = monthly_df.pivot_table(values="temp_mean", index="year", columns="month")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        pivot,
        cmap="RdYlBu_r",
        annot=False,
        linewidths=0.3,
        fmt=".1f",
        ax=ax,
        cbar_kws={"label": "Mean Temperature (°C)"}
    )
    ax.set_title("Monthly Mean Temperature Heatmap (1990–2023)", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    save_fig("04_monthly_temp_heatmap.png")


# ── PLOT 5: Anomaly Detection Chart ───────────────────────────────────────────
def plot_anomaly_detection(daily_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    
    # Panel 1: Temperature anomalies
    ax1 = axes[0]
    ax1.plot(daily_df["date"], daily_df["temp_30d_mean"],
             color="#3498db", linewidth=0.8, alpha=0.7, label="30d Temp")
    
    anomalies = daily_df[daily_df["is_anomaly"] == True]
    ax1.scatter(anomalies["date"], anomalies["temperature_c"],
                color="#e74c3c", s=20, zorder=5, label="Anomaly", alpha=0.8)
    
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Temperature Anomalies Detected (1990–2023)", fontweight="bold")
    ax1.legend(loc="upper left")
    
    # Panel 2: Rainfall anomalies
    ax2 = axes[1]
    ax2.bar(daily_df["date"], daily_df["rainfall_mm"],
            color="#85c1e9", alpha=0.5, width=1, label="Daily Rainfall")
    
    rain_anomalies = daily_df[daily_df["anomaly_zscore_rain"] == True]
    ax2.scatter(rain_anomalies["date"], rain_anomalies["rainfall_mm"],
                color="#e67e22", s=20, zorder=5, label="Rain Anomaly")
    
    ax2.set_ylabel("Rainfall (mm)")
    ax2.set_xlabel("Year")
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    save_fig("05_anomaly_detection.png")


# ── PLOT 6: Decadal Comparison ────────────────────────────────────────────────
def plot_decadal_comparison(decadal_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Temperature by decade
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"][:len(decadal_df)]
    axes[0].bar(decadal_df["decade_label"], decadal_df["temp_mean"],
                color=colors, edgecolor="black", linewidth=0.7)
    axes[0].set_title("Mean Temperature by Decade", fontweight="bold")
    axes[0].set_ylabel("Temperature (°C)")
    
    for i, v in enumerate(decadal_df["temp_mean"]):
        axes[0].text(i, v + 0.05, f"{v:.2f}°", ha="center", fontsize=10)
    
    # Rainfall by decade
    axes[1].bar(decadal_df["decade_label"], decadal_df["rainfall_total"],
                color=colors, edgecolor="black", linewidth=0.7)
    axes[1].set_title("Mean Annual Rainfall by Decade", fontweight="bold")
    axes[1].set_ylabel("Rainfall (mm)")
    
    for i, v in enumerate(decadal_df["rainfall_total"]):
        axes[1].text(i, v + 5, f"{v:.0f}", ha="center", fontsize=10)
    
    plt.tight_layout()
    save_fig("06_decadal_comparison.png")


# ── PLOT 7: STL Decomposition ─────────────────────────────────────────────────
def plot_stl_decomposition(stl_result_dict: dict, variable: str = "Temperature"):
    decomp_df = stl_result_dict["decomp_df"]
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    titles = ["Observed", "Trend", "Seasonal", "Residual"]
    cols = ["observed", "trend", "seasonal", "residual"]
    colors = ["#2c3e50", "#e74c3c", "#3498db", "#27ae60"]
    
    for ax, title, col, color in zip(axes, titles, cols, colors):
        ax.plot(decomp_df["date"], decomp_df[col], color=color, linewidth=0.8)
        ax.set_ylabel(title)
        ax.set_title(f"{variable} — {title}", fontsize=11)
    
    axes[-1].set_xlabel("Date")
    plt.suptitle(f"STL Decomposition: {variable} (1990–2023)", y=1.01,
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig("07_stl_decomposition.png")


# ── PLOT 8: Forecast Chart ─────────────────────────────────────────────────────
def plot_forecast(yearly_df: pd.DataFrame, arima_forecast_df: pd.DataFrame, variable: str = "temp_mean"):
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Historical
    ax.plot(yearly_df["year"], yearly_df[variable],
            color="#2c3e50", linewidth=2, label="Historical", marker="o", markersize=4)
    
    # Forecast
    ax.plot(arima_forecast_df["year"], arima_forecast_df["forecast"],
            color="#e74c3c", linewidth=2, linestyle="--", label="ARIMA Forecast", marker="s", markersize=6)
    
    # Confidence interval
    ax.fill_between(
        arima_forecast_df["year"],
        arima_forecast_df["lower_95"],
        arima_forecast_df["upper_95"],
        color="#e74c3c", alpha=0.2, label="95% Confidence Interval"
    )
    
    # Vertical divider
    ax.axvline(x=yearly_df["year"].max(), color="grey", linestyle=":", linewidth=1.5)
    ax.text(yearly_df["year"].max() + 0.1, ax.get_ylim()[0] + 0.2, "Forecast →", fontsize=9, color="grey")
    
    label = "Temperature (°C)" if "temp" in variable else "Rainfall (mm)"
    ax.set_title(f"Climate Forecast — {label} (ARIMA)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel(label)
    ax.legend()
    
    fname = "08_temp_forecast.png" if "temp" in variable else "09_rain_forecast.png"
    save_fig(fname)


# ── PLOT 9: Correlation Heatmap ───────────────────────────────────────────────
def plot_correlation_heatmap(daily_df: pd.DataFrame):
    cols = ["temperature_c", "rainfall_mm", "humidity_pct", "wind_speed_kmh"]
    corr = daily_df[cols].corr()
    
    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        mask=mask,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("Climate Variables Correlation Matrix", fontweight="bold")
    save_fig("10_correlation_heatmap.png")


# ── PLOT 10: Interactive Plotly Dashboard ─────────────────────────────────────
def create_interactive_dashboard(daily_df: pd.DataFrame, yearly_df: pd.DataFrame, anomaly_df: pd.DataFrame):
    """Creates a multi-panel interactive Plotly chart saved as HTML."""
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Annual Temperature Trend",
            "Annual Rainfall Trend",
            "Temperature Anomalies",
            "Monthly Rainfall Distribution",
            "Seasonal Temperature Comparison",
            "Humidity vs Temperature"
        ),
        vertical_spacing=0.12
    )
    
    # 1. Annual temperature
    fig.add_trace(
        go.Scatter(x=yearly_df["year"], y=yearly_df["temp_mean"],
                   mode="lines+markers", name="Yearly Temp",
                   line=dict(color="#e74c3c", width=2)),
        row=1, col=1
    )
    
    # 2. Annual rainfall
    fig.add_trace(
        go.Bar(x=yearly_df["year"], y=yearly_df["rainfall_total"],
               name="Rainfall", marker_color="#3498db"),
        row=1, col=2
    )
    
    # 3. Temperature anomalies
    non_anomaly = daily_df[daily_df["is_anomaly"] == False]
    fig.add_trace(
        go.Scatter(x=non_anomaly["date"], y=non_anomaly["temp_30d_mean"],
                   mode="lines", name="Normal",
                   line=dict(color="#95a5a6", width=0.5), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=anomaly_df["date"], y=anomaly_df["temperature_c"],
                   mode="markers", name="Anomaly",
                   marker=dict(color="#e74c3c", size=5)),
        row=2, col=1
    )
    
    # 4. Monthly rainfall box
    monthly_means = daily_df.groupby("month")["rainfall_mm"].mean().reset_index()
    fig.add_trace(
        go.Bar(x=monthly_means["month"], y=monthly_means["rainfall_mm"],
               name="Monthly Rain Avg", marker_color="#2980b9"),
        row=2, col=2
    )
    
    # 5. Seasonal temp
    season_means = daily_df.groupby(["season", "year"])["temperature_c"].mean().reset_index()
    for season, color in zip(["Winter","Spring","Summer","Autumn"],
                              ["#3498db","#2ecc71","#e74c3c","#f39c12"]):
        s_data = season_means[season_means["season"] == season]
        fig.add_trace(
            go.Scatter(x=s_data["year"], y=s_data["temperature_c"],
                       mode="lines", name=season,
                       line=dict(color=color, width=1.5)),
            row=3, col=1
        )
    
    # 6. Humidity vs Temperature scatter
    sample = daily_df.sample(2000, random_state=42)
    fig.add_trace(
        go.Scatter(x=sample["temperature_c"], y=sample["humidity_pct"],
                   mode="markers", name="Humidity vs Temp",
                   marker=dict(size=3, color="#9b59b6", opacity=0.5)),
        row=3, col=2
    )
    
    fig.update_layout(
        height=1100,
        title_text="<b>Climate Trend Analyzer — Interactive Dashboard</b>",
        title_font_size=18,
        showlegend=True,
        template="plotly_white"
    )
    
    output_path = "outputs/reports/interactive_dashboard.html"
    fig.write_html(output_path)
    print(f"[SAVED] Interactive dashboard → {output_path}")
    
    return fig


def generate_all_plots(daily_df, monthly_df, yearly_df, decadal_df, stl_temp, stl_rain, arima_results):
    """Run all visualization functions."""
    print("\n[VISUALIZATION] Generating all charts...")
    
    plot_temperature_trend(daily_df, yearly_df)
    plot_rainfall_trend(yearly_df)
    plot_seasonal_temperature(daily_df)
    plot_monthly_heatmap(monthly_df)
    plot_anomaly_detection(daily_df)
    plot_decadal_comparison(decadal_df)
    plot_stl_decomposition(stl_temp, "Temperature")
    plot_stl_decomposition(stl_rain, "Rainfall")
    
    if "arima_temp" in arima_results:
        plot_forecast(yearly_df, arima_results["arima_temp"], "temp_mean")
    if "arima_rain" in arima_results:
        plot_forecast(yearly_df, arima_results["arima_rain"], "rainfall_total")
    
    plot_correlation_heatmap(daily_df)
    
    anomaly_df = daily_df[daily_df["is_anomaly"] == True]
    create_interactive_dashboard(daily_df, yearly_df, anomaly_df)
    
    print(f"\n[COMPLETE] All charts saved to {FIGURES_DIR}/")