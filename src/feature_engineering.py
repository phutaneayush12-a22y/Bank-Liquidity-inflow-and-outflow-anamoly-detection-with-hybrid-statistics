import pandas as pd
import numpy as np


def create_time_features(df):
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    return df


def create_lag_features(df):
    df['lag_1'] = df['net_flow'].shift(1)
    df['lag_24'] = df['net_flow'].shift(24)
    return df


def create_rolling_features(df):
    df['rolling_mean_24'] = df['net_flow'].rolling(24).mean()
    df['rolling_std_24'] = df['net_flow'].rolling(24).std()
    return df


def create_extra_features(df):
    df['net_flow_diff'] = df['net_flow'].diff()
    df['pct_change'] = df['net_flow'].pct_change()
    df.replace([np.inf, -np.inf], 0, inplace=True)
    return df


def add_macro_features(df):
    df['year'] = df.index.year
    df['interest_rate'] = 6.5
    return df


def finalize_dataset(df):
    df = df.ffill().bfill()

    df = df.dropna(subset=[
        'lag_1',
        'lag_24',
        'rolling_mean_24',
        'rolling_std_24'
    ])

    return df