# 🌍 Climate Trend Analyzer

> End-to-end climate data analysis pipeline — trend detection, anomaly identification,
> forecasting, and interactive visualization using Python.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)

---

## 📌 Project Overview

Climate Trend Analyzer is a data science project that simulates and analyzes **34 years of daily climate data** (1990–2023) to identify:

- Long-term temperature and rainfall trends
- Seasonal pattern shifts across decades
- Climate anomalies (heat waves, droughts, cold snaps)
- Future climate forecasts using ARIMA and Facebook Prophet

This project demonstrates real-world skills applicable to **environmental data analysis, research analytics, and climate informatics** roles.

---

## 🎯 Problem Statement

Climate change generates vast amounts of time-series data, but extracting meaningful patterns requires statistical rigor. This project addresses:

- How do we quantify the rate of temperature increase over decades?
- How do we automatically detect extreme weather events in historical data?
- How do we forecast future climate conditions with statistical confidence?

---

## 🏭 Industry Relevance

| Sector       | Use Case                                           |
| ------------ | -------------------------------------------------- |
| Government   | Disaster preparedness, water resource planning     |
| Smart Cities | Urban heat island analysis, flood risk zones       |
| Agriculture  | Crop cycle optimization using seasonal forecasts   |
| Insurance    | Climate risk pricing for flood/drought events      |
| Research     | Validating climate models, publishing trend papers |

---

## 🛠️ Tech Stack

| Category      | Tools                                 |
| ------------- | ------------------------------------- |
| Language      | Python 3.9+                           |
| Data          | Pandas, NumPy                         |
| Statistics    | Statsmodels, SciPy, pymannkendall     |
| ML            | Scikit-learn (Isolation Forest)       |
| Forecasting   | ARIMA (statsmodels), Facebook Prophet |
| Visualization | Matplotlib, Seaborn, Plotly           |
| Dashboard     | Streamlit                             |
| Testing       | pytest                                |

---

## 🏗️ Architecture

Raw Data → Cleaning → Feature Engineering
↓ ↓
Trend Analysis Anomaly Detection
↓ ↓
Forecasting → Visualization → Streamlit Dashboard

---

## 📁 Folder Structure

```
Climate-Trend-Analyzer/
├── data/ # Raw and processed datasets
├── notebooks/ # Jupyter notebooks (EDA, analysis, forecasting)
├── src/ # Python modules (loader, features, trend, anomaly, forecast, viz)
├── app/ # Streamlit dashboard
├── outputs/ # Generated figures, tables, reports
├── tests/ # Unit tests
├── images/ # Screenshots for README
├── main.py # Full pipeline runner
└── requirements.txt
```

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/Climate-Trend-Analyzer.git
cd Climate-Trend-Analyzer
python -m venv climate_env
source climate_env/bin/activate  # Windows: climate_env\Scripts\activate
pip install -r requirements.txt
```

---

## 📊 Dataset

- **Type:** Synthetic simulation of real climate data patterns
- **Period:** 1990–2023 (34 years, 12,419 daily records)
- **Variables:** Temperature (°C), Rainfall (mm), Humidity (%), Wind Speed (km/h)
- **Features:** Realistic warming trend (+1.5°C over 34 years), seasonal cycles, injected anomalies
- **Anomalies injected:** 1998 El Niño heat wave, 2002 drought, 2015 extreme heat, 2017 flood event

---

## 🚀 How to Run

```bash
# Full pipeline
python main.py

# Streamlit dashboard
streamlit run app/dashboard.py

# Jupyter notebooks
jupyter notebook

# Tests
python -m pytest tests/ -v
```

---

## 📈 Results

### Temperature Trend

- **+1.50°C warming** detected over 34 years
- **+0.044°C per year** rate of increase
- Mann-Kendall test: **statistically significant** (p < 0.0001)

### Rainfall Trend

- **Declining trend** of ~8.3 mm per decade
- Monsoon seasonality clearly preserved

### Anomalies Detected

- **~490 anomalous days** identified (Z-score + IQR + Isolation Forest consensus)
- Key events: 1998 heat wave, 2002 drought, 2015 extreme heat, 2017 floods

### Forecast (ARIMA)

- Temperature predicted to rise **+0.2–0.3°C** over next 5 years
- 95% confidence intervals provided

---

## 📸 Screenshots

### Temperature Trend

![Temperature Trend](images/04_temp_trend.png)

### Anomaly Detection

![Anomaly Detection](images/08_anomaly_chart.png)

### Streamlit Dashboard

![Dashboard](images/10_dashboard.png)

### Forecast

![Forecast](images/09_forecast_chart.png)

---

## 🔮 Future Improvements

- Region-wise multi-city climate comparison
- Integration with real NOAA/ERA5 climate datasets
- Geospatial visualization (choropleth maps)
- Deep learning forecasting (LSTM)
- Automated PDF climate reports
- Live weather API integration

---

## 📚 Learning Outcomes

- Time-series data cleaning and feature engineering
- Statistical trend detection (Mann-Kendall, STL decomposition)
- Multi-method anomaly detection (Z-score, IQR, Isolation Forest)
- ARIMA and Prophet forecasting
- Interactive dashboard development with Streamlit
- Professional GitHub repository management

---

## 👤 Author

**[CH S K CHAITANYA]**  
Data Science Student | Climate Data Analytics  
📧 [chskchaitanya755@gmail.com]  
🔗 [LinkedIn](https://linkedin.com/in/chskchaitanya)
🐙 [GitHub](https://github.com/CH-S-K-CHAITANYA)

---

## 📄 License

MIT License — free to use for educational and portfolio purposes.
