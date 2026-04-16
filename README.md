🏦 Bank Liquidity Anomaly Detection System

📌 Overview

This project is an end-to-end machine learning system designed to detect unusual patterns in banking transactions using unsupervised learning.

It processes transaction data, engineers time-series features, trains anomaly detection models, and provides a real-time interactive interface for predictions.

---

🚀 Features

- 📊 Time-series based anomaly detection
- ⚙️ Advanced feature engineering (lag, rolling statistics, cyclic time features)
- 🤖 Multiple ML models:
  - Isolation Forest (Primary)
  - One-Class SVM
  - Local Outlier Factor (LOF)
- 📈 Exploratory Data Analysis (EDA)
- 🌐 Streamlit-based interactive UI
- 🔍 Model comparison and agreement analysis
- ⚡ Real-time feature generation using history buffer (advanced system)

---

🧠 How It Works

1. Data Processing

- Raw banking transactions are aggregated into hourly format
- Core feature:
  - "net_flow = deposit_amount - withdrawal_amount"

---

2. Feature Engineering

📌 Time-Series Features

- "lag_1", "lag_24" → past transaction dependency
- "rolling_mean_24", "rolling_std_24" → trend & volatility

📌 Change Features

- "pct_change" → percentage variation
- "net_flow_diff" → absolute change

📌 Time Features

- "hour_sin", "hour_cos" → cyclic encoding
- "is_weekend"

📌 External Feature

- "interest_rate"

---

3. Model Training

Three unsupervised models were trained:

- Isolation Forest (Selected Model)
  - Efficient and robust for anomaly detection
- One-Class SVM
  - Learns boundary of normal data
- Local Outlier Factor (LOF)
  - Detects anomalies based on local density

---

4. Model Comparison

- Compared predictions across models
- Created agreement metrics:
  - "IF_SVM"
  - "IF_LOF"
  - "ALL_THREE" (high-confidence anomalies)

👉 Isolation Forest was selected due to stability and performance

---

5. Exploratory Data Analysis (EDA)

- Correlation heatmap
- Time-series visualization
- Distribution & outlier analysis

---

🧠 Intelligent Feature Engineering Upgrade (Advanced System)

⚠️ Problem (Earlier)

The initial system generated features without historical context:

- "lag = current value"
- "rolling_std = 0"
- Unrealistic inputs → incorrect predictions

---

✅ Solution Implemented

A real-time history buffer system was introduced:

- Maintains last 50 transaction values
- Dynamically computes:
  - "lag_1", "lag_24"
  - "rolling_mean_24", "rolling_std_24"
  - "pct_change", "net_flow_diff"
- Uses hybrid statistical approach combining:
  - live user input
  - historical transaction patterns

---

🚀 Result

- Realistic feature generation
- Improved anomaly detection accuracy
- System now mimics real-world banking behavior

---

🌐 Deployment (Streamlit UI)

User Inputs:

- Transaction Amount
- Transaction Type (Deposit / Withdrawal)
- Transaction Hour
- Weekend Indicator
- Interest Rate

Backend Process:

1. Convert input → "net_flow"
2. Update history buffer
3. Generate time-series features dynamically
4. Apply scaler
5. Predict using Isolation Forest

Output:

- ✅ Normal Transaction
- 🚨 Anomaly Detected

---

🗂️ Project Structure

ML_Project/
│
├── DATA/
│   ├── RAW/
│   └── Processed_data/
│       └── df_final.csv
│
├── models/
│   ├── isolation_forest.pkl
│   └── scaler.pkl
│
├── Notebooks/
│   ├── EDA.ipynb
│   ├── model_analysis.ipynb
│   └── model_comparison.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
│
├── app/
│   └── app.py   # Streamlit UI
│
└── main.py

---

▶️ How to Run

1. Install Dependencies

pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib

2. Run the Application

cd app
streamlit run app.py

---

🎯 Conclusion

This project demonstrates how combining time-series feature engineering with unsupervised learning can effectively detect anomalies in financial systems.

With the addition of a real-time history-based feature generation system, it now provides a more realistic and production-ready anomaly detection pipeline.

---

🔮 Future Scope

- 📈 Forecasting integration (ARIMA / LSTM / Prophet)
- ⚡ Real-time streaming pipeline
- 🧠 Explainable AI (anomaly reasoning)
- 🌐 Cloud deployment (AWS / GCP)

---

👨‍💻 Author

Ayush Phutane
