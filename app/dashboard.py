"""
app/dashboard.py
Streamlit web dashboard for Climate Trend Analyzer.
Run with: streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.abspath("src"))

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Climate Trend Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Styling ───────────────────────────────────────────────────────────────
# ── CSS Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    :root {
        --bg-primary: #080c14; --bg-secondary: #0d1526; --bg-card: #111827;
        --teal: #00d4aa; --teal-dim: rgba(0, 212, 170, 0.12); --teal-glow: 0 0 24px rgba(0, 212, 170, 0.25);
        --text-primary: #f0f6ff; --text-secondary: #94a3b8; --font-main: 'Space Grotesk', sans-serif;
    }
    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(ellipse 80% 50% at 20% 0%, rgba(0,212,170,0.06) 0%, transparent 60%), var(--bg-primary) !important;
        color: var(--text-primary) !important; font-family: var(--font-main) !important;
    }
    [data-testid="stSidebar"] { background: var(--bg-secondary) !important; border-right: 1px solid rgba(0,212,170,0.1) !important; }
    [data-testid="stTabs"] [role="tablist"] { background: var(--bg-card) !important; border: 1px solid rgba(0,212,170,0.1) !important; border-radius: 10px; padding: 4px; }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] { background: var(--teal-dim) !important; color: var(--teal) !important; border: 1px solid rgba(0,212,170,0.4) !important; }
    [data-testid="stMetric"] { background: var(--bg-card); border: 1px solid rgba(0,212,170,0.1); border-radius: 12px; padding: 15px; }
    /* Hide Default Chrome */
    #MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load cleaned data with caching for performance."""
    if not os.path.exists("data/processed/climate_cleaned.csv"):
        from generate_dataset import generate_climate_data
        from data_loader import load_and_prepare
        from feature_engineering import engineer_all_features
        from anomaly_detection import detect_all_anomalies
        
        generate_climate_data()
        df = load_and_prepare()
        df, monthly, yearly = engineer_all_features(df)
        df = detect_all_anomalies(df)
        return df, monthly, yearly
    
    df = pd.read_csv("data/processed/climate_cleaned.csv", parse_dates=["date"])
    
    if "year" not in df.columns:
        from feature_engineering import engineer_all_features
        from anomaly_detection import detect_all_anomalies
        df, monthly, yearly = engineer_all_features(df)
        df = detect_all_anomalies(df)
    else:
        monthly = df.groupby(["year","month"]).agg(
            temp_mean=("temperature_c","mean"),
            rainfall_total=("rainfall_mm","sum")
        ).reset_index()
        monthly["date"] = pd.to_datetime(
            monthly["year"].astype(str) + "-" + monthly["month"].astype(str) + "-01"
        )
        yearly = df.groupby("year").agg(
            temp_mean=("temperature_c","mean"),
            temp_max=("temperature_c","max"),
            temp_min=("temperature_c","min"),
            rainfall_total=("rainfall_mm","sum"),
            humidity_mean=("humidity_pct","mean"),
            wind_mean=("wind_speed_kmh", "mean")
        ).reset_index()
    
    return df, monthly, yearly


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
with st.spinner("Crunching historical climate data..."):
    df, monthly, yearly = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    #st.image("https://img.icons8.com/emoji/96/globe-showing-europe-africa.png", width=60)
    st.markdown("<div style='font-size: 60px; line-height: 1.2; margin-bottom: 10px;'>🌍</div>", unsafe_allow_html=True)
    st.title("Settings & Filters")
    st.markdown("---")
    
    st.subheader("📊 Primary Focus")
    variable = st.selectbox(
        "Select Variable to Analyze", 
        ["Temperature", "Rainfall", "Humidity", "Wind Speed"],
        help="This changes the primary variable shown across all tabs."
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📅 Timeframe & Season")
    min_yr, max_yr = int(df["year"].min()), int(df["year"].max())
    year_range = st.slider("Select Year Range", min_value=min_yr, max_value=max_yr, value=(min_yr, max_yr))
    
    seasons = st.multiselect(
        "Filter by Season", 
        ["Winter", "Spring", "Summer", "Autumn"], 
        default=["Winter", "Spring", "Summer", "Autumn"]
    )
    
    st.markdown("---")
    st.markdown("**About this App**")
    st.caption("A full-stack data pipeline visualizing synthetic global climate patterns, anomalies, and forecasting.")

# ── FILTER DATA ───────────────────────────────────────────────────────────────
mask = (df["year"] >= year_range[0]) & (df["year"] <= year_range[1])
if "season" in df.columns:
    mask &= df["season"].isin(seasons)
df_filtered = df[mask]
yearly_filtered = yearly[(yearly["year"] >= year_range[0]) & (yearly["year"] <= year_range[1])]

col_map_daily = {"Temperature": "temperature_c", "Rainfall": "rainfall_mm", "Humidity": "humidity_pct", "Wind Speed": "wind_speed_kmh"}
col_map_yearly = {"Temperature": "temp_mean", "Rainfall": "rainfall_total", "Humidity": "humidity_mean", "Wind Speed": "wind_mean"}

daily_var = col_map_daily[variable]
yearly_var = col_map_yearly[variable]

agg_func = "sum" if variable == "Rainfall" else "mean"
unit = "mm" if variable == "Rainfall" else "°C" if variable == "Temperature" else "%" if variable == "Humidity" else "km/h"

# ── MAIN HEADER ───────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🌍 Climate Intelligence Dashboard</p>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">Currently analyzing <b>{variable}</b> trends from <b>{year_range[0]} to {year_range[1]}</b> across {len(seasons)} seasons.</p>', unsafe_allow_html=True)

# ── METRICS ROW (Dynamic Deltas) ──────────────────────────────────────────────
# Calculate deltas (Trend between first and last year in selection)
try:
    t_delta = yearly_filtered['temp_mean'].iloc[-1] - yearly_filtered['temp_mean'].iloc[0]
    r_delta = yearly_filtered['rainfall_total'].iloc[-1] - yearly_filtered['rainfall_total'].iloc[0]
    h_delta = yearly_filtered['humidity_mean'].iloc[-1] - yearly_filtered['humidity_mean'].iloc[0]
except IndexError:
    t_delta = r_delta = h_delta = 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Temperature", f"{df_filtered['temperature_c'].mean():.1f}°C", f"{t_delta:+.1f}°C trend")
col2.metric("Avg Annual Rainfall", f"{yearly_filtered['rainfall_total'].mean():,.0f} mm", f"{r_delta:+.0f} mm trend")
col3.metric("Avg Humidity", f"{df_filtered['humidity_pct'].mean():.1f}%", f"{h_delta:+.1f}% trend", delta_color="off")
col4.metric("Anomalies Detected", f"{df_filtered.get('is_anomaly', pd.Series(False, index=df_filtered.index)).sum():,}", "Requires review", delta_color="inverse")
st.markdown("<br>", unsafe_allow_html=True)

# ── TAB LAYOUT ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trend Analysis", "🌦️ Seasonal Patterns", "🚨 Anomaly Detection", "🔮 Forecast", "📊 Raw Data"])

# ── TAB 1: TREND ANALYSIS ─────────────────────────────────────────────────────
with tab1:
    st.subheader(f"Macro {variable} Trends")
    c1, c2 = st.columns([3, 2]) # Make the line chart slightly wider than the bar chart
    
    with c1:
        fig_trend = px.line(
            yearly_filtered, x="year", y=yearly_var,
            labels={yearly_var: f"{variable} ({unit})", "year": ""},
            color_discrete_sequence=["#e74c3c" if variable == "Temperature" else "#3498db"]
        )
        z = np.polyfit(yearly_filtered["year"], yearly_filtered[yearly_var].fillna(0), 1)
        p = np.poly1d(z)
        trend_val = z[0]*10
        trend_sign = "+" if trend_val > 0 else ""
        
        fig_trend.add_scatter(x=yearly_filtered["year"], y=p(yearly_filtered["year"]),
                             mode="lines", name=f"10-yr Trend ({trend_sign}{trend_val:.2f})",
                             line=dict(color="#2C3E50", dash="dash", width=2))
                             
        # UX Upgrade: Crosshair hover mode
        fig_trend.update_layout(height=380, template="plotly_white", hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with c2:
        fig_bar = px.bar(
            yearly_filtered, x="year", y=yearly_var,
            labels={yearly_var: f"{variable} ({unit})", "year": ""},
            color=yearly_var,
            color_continuous_scale="Blues" if variable == "Rainfall" else "Reds"
        )
        fig_bar.update_layout(height=380, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader(f"Micro-Trends: 30-day Rolling Average")
    with st.expander("ℹ️ How to read this chart"):
        st.write("This chart smooths out daily fluctuations by averaging the data over a rolling 30-day window. This makes it much easier to spot mid-term climate shifts and seasonal peaks without the noise of daily weather spikes.")
        
    roll_df = df_filtered[["date", daily_var]].dropna().sort_values("date")
    roll_df[f"rolling_{daily_var}"] = roll_df[daily_var].rolling(window=30).mean()
    
    sample = roll_df.sample(min(5000, len(roll_df)), random_state=42).sort_values("date")
    fig_roll = px.line(sample, x="date", y=f"rolling_{daily_var}",
                       labels={f"rolling_{daily_var}": f"{variable} ({unit})", "date": ""},
                       color_discrete_sequence=["#1ABC9C"])
    fig_roll.update_layout(height=300, template="plotly_white", hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_roll, use_container_width=True)


# ── TAB 2: SEASONAL PATTERNS ──────────────────────────────────────────────────
with tab2:
    st.subheader(f"Seasonal Distribution ({variable})")
    
    if "season" in df_filtered.columns:
        c1, c2 = st.columns(2)
        
        with c1:
            fig_box = px.box(
                df_filtered, x="season", y=daily_var,
                color="season",
                category_orders={"season": ["Winter","Spring","Summer","Autumn"]},
                color_discrete_map={"Winter":"#3498db","Spring":"#2ecc71","Summer":"#e74c3c","Autumn":"#f39c12"},
                labels={daily_var: f"{variable} ({unit})", "season": ""}
            )
            fig_box.update_layout(height=400, showlegend=False, template="plotly_white", margin=dict(t=20))
            st.plotly_chart(fig_box, use_container_width=True)
        
        with c2:
            month_avg = df_filtered.groupby("month").agg(val=(daily_var, agg_func)).reset_index()
            # Map month numbers to names for better UX
            month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
            month_avg['month_name'] = month_avg['month'].map(month_dict)
            
            fig_monthly = px.bar(
                month_avg, x="month_name", y="val",
                labels={"month_name":"", "val": f"{'Total' if agg_func == 'sum' else 'Avg'} {variable} ({unit})"},
                color="val", color_continuous_scale="Tealgrn" if variable == "Humidity" else "Sunset"
            )
            fig_monthly.update_layout(height=400, template="plotly_white", margin=dict(t=20), coloraxis_showscale=False)
            st.plotly_chart(fig_monthly, use_container_width=True)


# ── TAB 3: ANOMALY DETECTION ──────────────────────────────────────────────────
with tab3:
    st.subheader(f"Anomaly Detection: Isolation Forest")
    with st.expander("ℹ️ What is an Anomaly here?"):
        st.write(f"The machine learning model scans the historical dataset for multivariate outliers. A red 'X' indicates a day where the combination of temperature, rainfall, and wind speed was statistically highly unusual compared to historical norms for that season.")

    if "is_anomaly" in df_filtered.columns:
        anomalies = df_filtered[df_filtered["is_anomaly"] == True]
        normal = df_filtered[df_filtered["is_anomaly"] == False]
        
        fig_anom = go.Figure()
        fig_anom.add_trace(go.Scatter(
            x=normal["date"].sample(min(3000, len(normal)), random_state=42).sort_values(),
            y=normal.loc[normal.index.isin(normal["date"].sample(min(3000, len(normal)), random_state=42).index), daily_var],
            mode="lines", name="Normal Weather",
            line=dict(color="#CBD5E1", width=1),
            hoverinfo="skip" # Clean up hover UX
        ))
        fig_anom.add_trace(go.Scatter(
            x=anomalies["date"],
            y=anomalies[daily_var],
            mode="markers", name="Statistically Anomalous",
            marker=dict(color="#E74C3C", size=8, symbol="x", line=dict(width=1, color="darkred")),
            hovertemplate="Date: %{x}<br>Value: %{y}<extra></extra>"
        ))
        fig_anom.update_layout(
            height=450, template="plotly_white",
            yaxis_title=f"{variable} ({unit})",
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_anom, use_container_width=True)


# ── TAB 4: FORECAST ───────────────────────────────────────────────────────────
with tab4:
    st.subheader(f"Predictive Forecast (ARIMA): {variable}")
    
    file_prefix_map = {"Temperature": "temp", "Rainfall": "rain", "Humidity": "humidity", "Wind Speed": "wind"}
    prefix = file_prefix_map[variable]
    arima_path = f"outputs/tables/arima_{prefix}_forecast.csv" 
    
    if os.path.exists(arima_path):
        arima_fc = pd.read_csv(arima_path)
        
        fig_fc = go.Figure()
        # Historical Trace
        fig_fc.add_trace(go.Scatter(
            x=yearly_filtered["year"], y=yearly_filtered[yearly_var],
            mode="lines+markers", name=f"Historical",
            line=dict(color="#334155", width=2),
            marker=dict(size=6)
        ))
        # Forecast Trace
        fig_fc.add_trace(go.Scatter(
            x=arima_fc["year"], y=arima_fc["forecast"],
            mode="lines+markers", name="Forecast",
            line=dict(color="#3B82F6", dash="dash", width=3),
            marker=dict(size=8)
        ))
        # Confidence Interval
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([arima_fc["year"], arima_fc["year"][::-1]]),
            y=pd.concat([arima_fc["upper_95"], arima_fc["lower_95"][::-1]]),
            fill="toself", fillcolor="rgba(59, 130, 246, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% Confidence Interval",
            hoverinfo="skip"
        ))
        fig_fc.update_layout(
            height=450, template="plotly_white", hovermode="x unified",
            yaxis_title=f"{variable} ({unit})",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_fc, use_container_width=True)
        
        # Display as a styled dataframe underneath
        st.caption("Forecast Data Output")
        st.dataframe(arima_fc.style.format({"forecast": "{:.2f}", "lower_95": "{:.2f}", "upper_95": "{:.2f}"}), use_container_width=True)
        
    else:
        st.info(f"No forecast data found for {variable}. The app is looking for `{arima_path}`.")
        if st.button("🚀 Run Backend ML Pipeline", key=f"run_pipeline_{variable}"):
            with st.spinner("Training models and generating forecasts..."):
                os.system("python main.py")
            st.success("Done! Refresh the page.")

# ── TAB 5: RAW DATA ───────────────────────────────────────────────────────────
with tab5:
    st.subheader("Data Explorer")
    
    col_search, col_dl = st.columns([3, 1])
    with col_search:
        search_year = st.number_input("Filter Data by Specific Year", min_value=min_yr, max_value=max_yr, value=max_yr)
    
    year_data = df_filtered[df_filtered["year"] == search_year] if "year" in df_filtered.columns else df_filtered
    display_cols = [c for c in ["date","temperature_c","rainfall_mm","humidity_pct","wind_speed_kmh","season", "is_anomaly"] if c in year_data.columns]
    
    # UX Upgrade: Use st.dataframe with custom configuration
    st.dataframe(
        year_data[display_cols].reset_index(drop=True), 
        height=400, 
        use_container_width=True,
        hide_index=True
    )
    
    with col_dl:
        st.markdown("<br>", unsafe_allow_html=True)
        csv = year_data[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download {search_year} CSV",
            data=csv,
            file_name=f"climate_data_{search_year}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div style='text-align: center; color: #94A3B8; font-size: 0.9rem;'>Built with Python, Plotly, and Streamlit</div>", unsafe_allow_html=True)