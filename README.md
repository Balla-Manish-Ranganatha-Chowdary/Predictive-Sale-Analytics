# 📈 Predictive Sales Analytics

An interactive machine learning application for forecasting daily sales using time series analysis and regression models. Built with Streamlit for real-time visualization and predictions.

## 🎯 Overview

This project implements a sales forecasting system that predicts future daily sales based on historical data from a retail store-item demand dataset. The application uses feature engineering techniques including lag features and rolling averages to capture temporal patterns and trends.

## ✨ Features

- **Interactive Dashboard**: Real-time sales forecasting with adjustable prediction horizons (1-30 days)
- **ML-Powered Predictions**: Trained regression model using time-based and lag features
- **Recursive Forecasting**: Multi-step ahead predictions with dynamic feature updates
- **Visual Analytics**: Historical vs. predicted sales visualization with matplotlib
- **KPI Metrics**: Key performance indicators including average daily sales and forecast summaries

## 🛠️ Tech Stack

- **Python 3.13**
- **Streamlit**: Interactive web application framework
- **Pandas**: Data manipulation and time series handling
- **Scikit-learn**: Machine learning model training and prediction
- **Matplotlib**: Data visualization
- **Joblib**: Model serialization

## 📊 Dataset

The project uses the Store Item Demand Forecasting Challenge dataset from Kaggle, containing:
- **913,000 records** of daily sales data
- **10 stores** and **50 items**
- **Date range**: January 2013 - December 2017
- **Aggregated daily sales** for simplified forecasting

## 🚀 Getting Started

### Prerequisites

```bash
pip install streamlit pandas joblib matplotlib scikit-learn
```

### Installation

1. Clone the repository
2. Ensure the following files are in the project directory:
   - `best_sales_forecast_model.pkl` - Trained ML model
   - `model_features.pkl` - Feature order for predictions
   - `clean_daily_sales.csv` - Historical sales data

### Running the Application

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
.
├── streamlit_app.py                    # Main Streamlit application
├── predictive sale analysis.ipynb      # Model training and analysis notebook
├── best_sales_forecast_model.pkl       # Serialized trained model
├── model_features.pkl                  # Feature names and order
├── clean_daily_sales.csv               # Preprocessed daily sales data
└── README.md                           # Project documentation
```

## 🔧 Model Features

The forecasting model uses the following engineered features:

- **Temporal Features**: Day, month, year
- **Lag Features**: 
  - `lag_1`: Previous day's sales
  - `lag_7`: Sales from 7 days ago
- **Rolling Statistics**: 
  - `rolling_7`: 7-day moving average

## 📈 How It Works

1. **Data Loading**: Historical sales data is loaded and preprocessed
2. **Feature Engineering**: Time-based and lag features are computed
3. **Model Prediction**: Trained model generates forecasts
4. **Recursive Forecasting**: Each prediction feeds into the next step for multi-day forecasts
5. **Visualization**: Results are displayed with interactive charts and metrics

## 🎮 Usage

1. Launch the Streamlit app
2. Use the sidebar slider to adjust the forecast horizon (1-30 days)
3. View the forecast visualization showing historical and predicted sales
4. Monitor KPI metrics for quick insights

## 📝 Model Training

The model was trained using:
- Historical sales data from 2013-2017
- Feature engineering with lag and rolling window techniques
- Regression algorithm (details in the Jupyter notebook)
- Train-test split for validation

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

This project is open source and available for educational and commercial use.

## 🙏 Acknowledgments

- Dataset: [Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only) on Kaggle
- Built with Streamlit for rapid prototyping and deployment

---

**Note**: Ensure all required pickle files and CSV data are present before running the application.
