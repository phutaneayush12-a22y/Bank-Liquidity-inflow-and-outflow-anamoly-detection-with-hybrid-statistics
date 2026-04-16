💰 Bank Liquidity Anomaly Detection System
Hybrid Statistical and Machine Learning Approach
📌 Overview

This project presents a hybrid anomaly detection system designed to identify unusual patterns in banking liquidity transactions. It combines statistical methods (IQR) with machine learning (Isolation Forest) to improve anomaly detection accuracy and robustness.

The system processes large-scale transaction data and converts it into time-series format for effective modeling and analysis.

⚙️ Tech Stack
Programming Language: Python
Libraries: Pandas, NumPy, Scikit-learn, Matplotlib
🚀 Key Features
🔍 Hybrid anomaly detection using IQR + Isolation Forest
📊 Time-series transformation of ~5 million raw transactions
🧠 Advanced feature engineering (11 features) including:
Lag variables
Rolling statistics
Percentage change
Temporal encodings
🧹 Data preprocessing:
Cleaning
Resampling
Normalization
Feature standardization
⚙️ Hybrid decision logic using OR-gate combination of statistical and ML outputs
📈 Visualization of liquidity patterns and anomalies
🏗️ System Workflow
Data Collection (Raw Transactions)
Data Preprocessing & Cleaning
Time-Series Aggregation (Hourly)
Feature Engineering
Model Training (Isolation Forest)
Statistical Analysis (IQR Method)
Hybrid Anomaly Detection
Visualization & Insights
📊 Dataset
~5 million raw banking transaction records
Aggregated into ~300K time-series data points
🎯 Objective

To build a scalable and accurate anomaly detection system that improves detection reliability by combining statistical and machine learning techniques.

🚀 Future Enhancements
Real-time anomaly detection (streaming data)
Integration with alert systems
Dashboard for monitoring anomalies
Model performance optimization
👨‍💻 Author

Ayush Phutane
