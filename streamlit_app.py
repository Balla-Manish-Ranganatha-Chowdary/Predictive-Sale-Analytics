import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Predictive Sales Analytics",
    layout="wide"
)

st.title("📈 Predictive Sales Analytics")
st.subheader("Interactive sales forecasting using ML models")

# -------------------------------------------------
# Load Model & Feature Order
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("best_sales_forecast_model.pkl")
    feature_order = joblib.load("model_features.pkl")
    return model, feature_order

model, model_features = load_model()

# -------------------------------------------------
# Sidebar Controls
# -------------------------------------------------
st.sidebar.header("Forecast Settings")

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=1,
    max_value=30,
    value=7
)

st.sidebar.markdown("---")
st.sidebar.info("Adjust forecast horizon to update predictions")

# -------------------------------------------------
# Load Historical Data
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("clean_daily_sales.csv", parse_dates=["date"])
    return df.sort_values("date")

data = load_data()

# -------------------------------------------------
# Feature Engineering (Inference)
# -------------------------------------------------
def prepare_features(df):
    features = pd.DataFrame({
        "day": df["date"].dt.day,
        "month": df["date"].dt.month,
        "year": df["date"].dt.year,
        "lag_1": df["sales"].shift(1),
        "lag_7": df["sales"].shift(7),
        "rolling_7": df["sales"].rolling(7).mean(),
    })
    return features.dropna()

# -------------------------------------------------
# Recursive Forecasting (CORRECT WAY)
# -------------------------------------------------
window = data.tail(14).copy()
features_df = prepare_features(window)

# enforce training feature order
features_df = features_df[model_features]

current_row = features_df.tail(1).copy()
forecast = []
forecast_dates = []

last_date = window["date"].iloc[-1]

for i in range(forecast_horizon):
    # predict next value
    pred = model.predict(current_row)[0]
    forecast.append(pred)

    # next date
    next_date = last_date + pd.Timedelta(days=1)
    forecast_dates.append(next_date)

    # update rolling window
    window = pd.concat(
        [window, pd.DataFrame({"date": [next_date], "sales": [pred]})],
        ignore_index=True
    )

    # recompute features for next step
    new_features = prepare_features(window).tail(1)
    new_features = new_features[model_features]

    current_row = new_features
    last_date = next_date

forecast = pd.Series(forecast)
forecast_dates = pd.to_datetime(forecast_dates)

# -------------------------------------------------
# Visualization
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    data["date"].tail(60),
    data["sales"].tail(60),
    label="Historical",
    linewidth=2
)

ax.plot(
    forecast_dates,
    forecast,
    label="Forecast",
    linestyle="--",
    marker="o"
)

ax.set_title("Sales Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()

st.pyplot(fig)

# -------------------------------------------------
# KPI Metrics
# -------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Daily Sales",
    f"{data['sales'].mean():.2f}"
)

col2.metric(
    "Forecast Horizon",
    f"{forecast_horizon} days"
)

col3.metric(
    "Latest Predicted Sales",
    f"{forecast.iloc[-1]:.2f}"
)
