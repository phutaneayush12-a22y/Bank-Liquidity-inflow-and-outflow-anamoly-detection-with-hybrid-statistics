from src.preprocessing import load_and_clean_data, resample_data
from src.feature_engineering import (
    create_time_features,
    create_lag_features,
    create_rolling_features,
    create_extra_features,
    add_macro_features,
    finalize_dataset
)
from src.train import train_model


def run_pipeline():
    df = load_and_clean_data("DATA/RAW/financial_fraud_detection_dataset(1).csv")

    df_hourly = resample_data(df)

    df_hourly = create_time_features(df_hourly)
    df_hourly = create_lag_features(df_hourly)
    df_hourly = create_rolling_features(df_hourly)
    df_hourly = create_extra_features(df_hourly)
    df_hourly = add_macro_features(df_hourly)

    df_final = finalize_dataset(df_hourly)

    model, scaler, X_test_scaled, test_df = train_model(df_final)

    print(" Model trained and saved successfully!")


if __name__ == "__main__":
    run_pipeline()