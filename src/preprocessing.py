import pandas as pd
import numpy as np


def load_and_clean_data(path):
    df = pd.read_csv(path)

    df = df[['timestamp', 'amount', 'transaction_type']]

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format='mixed')
    df = df.dropna(subset=['timestamp'])

    df.rename(columns={'timestamp': 'date'}, inplace=True)

    df = df.sort_values('date')

    df['deposit_amount'] = np.where(df['transaction_type'] == 'deposit', df['amount'], 0)
    df['withdrawal_amount'] = np.where(df['transaction_type'] != 'deposit', df['amount'], 0)

    return df


def resample_data(df):
    df = df.copy()

    df.set_index('date', inplace=True)

    df_hourly = df.resample('h').agg({
        'deposit_amount': 'sum',
        'withdrawal_amount': 'sum'
    })

    df_hourly['net_flow'] = df_hourly['deposit_amount'] - df_hourly['withdrawal_amount']

    return df_hourly