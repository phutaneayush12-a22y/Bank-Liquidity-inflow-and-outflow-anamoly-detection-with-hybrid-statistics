import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


def train_model(df):
    features = [
        'net_flow', 'lag_1', 'lag_24',
        'rolling_mean_24', 'rolling_std_24',
        'pct_change', 'net_flow_diff',
        'hour_sin', 'hour_cos',
        'is_weekend', 'interest_rate'
    ]

    X = df[features]

    split = int(len(X) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X_train_scaled)

    joblib.dump(model, "models/isolation_forest.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    return model, scaler, X_test_scaled, df.iloc[split:]