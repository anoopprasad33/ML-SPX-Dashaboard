# features.py
import pandas as pd
import numpy as np

def add_features(df):
    df = df.copy()
    df['SMA200'] = df['Close'].rolling(200, min_periods=50).mean()
    df['Momentum_30'] = df['Close'].pct_change(30)
    df['Momentum_90'] = df['Close'].pct_change(90)
    df['Volatility_30'] = df['Close'].pct_change().rolling(30).std()
    df['Returns_1'] = df['Close'].pct_change(1)
    df['Returns_5'] = df['Close'].pct_change(5)
    # EarningsYield = 1 / CAPE (CAPE should be monthly, forward-filled daily)
    df['EarningsYield'] = 1.0 / df['CAPE']
    df = df.dropna()
    return df

def make_targets(df, horizon_days=30):
    df = df.copy()
    df['Target_up'] = (df['Close'].shift(-horizon_days) > df['Close']).astype(int)
    # for regression target: next horizon return
    df['Target_return'] = df['Close'].shift(-horizon_days) / df['Close'] - 1.0
    df = df.dropna()
    return df
