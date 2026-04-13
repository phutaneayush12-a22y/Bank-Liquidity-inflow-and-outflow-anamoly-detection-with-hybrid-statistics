import joblib


def load_model():
    model = joblib.load("models/isolation_forest.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


def predict(data):
    model, scaler = load_model()

    data_scaled = scaler.transform(data)

    preds = model.predict(data_scaled)

    preds = [1 if x == -1 else 0 for x in preds]

    return preds