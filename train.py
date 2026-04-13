import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


features = [
    "deposit_amount",
    "withdrawal_amount",
    "net_flow",
    "net_flow_lag1",
    "net_flow_lag24",
    "rolling_mean_24",
    "rolling_std_24",
    "pct_change",
    "hour",
    "day_of_week",
    "is_weekend",
    "inflation_rate",
    "interest_rate"
]


df = joblib.load(".pkl")  

X = df[features]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


model = IsolationForest(
    n_estimators=100,
    contamination=0.01,
    random_state=42
)

model.fit(X_scaled)


joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(features, "features.pkl")

print("Model trained and saved")