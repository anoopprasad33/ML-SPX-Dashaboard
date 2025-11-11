# backtest.py
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from datetime import timedelta

def compute_perf(equity_series):
    equity = equity_series.dropna()
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    max_dd = drawdown.min()
    # approx sharpe (daily)
    returns = equity.pct_change().dropna()
    if returns.std() == 0:
        sharpe = np.nan
    else:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    return {'total_ret': total_ret, 'cagr': cagr, 'max_dd': max_dd, 'sharpe': sharpe}

def apply_strategy(df, xgb_model, lstm_model=None, scaler=None, lstm_features=None, threshold=0.5, ensemble='and'):
    df = df.copy()
    features_xgb = xgb_model.feature_names_in_ if hasattr(xgb_model, 'feature_names_in_') else xgb_model.get_booster().feature_names
    X = df[list(features_xgb)]
    proba = xgb_model.predict_proba(X)[:,1]
    pred_xgb = (proba > threshold).astype(int)
    df['proba_xgb'] = proba

    if lstm_model is not None:
        # prepare last sequences for LSTM prediction (requires scaler & lookback)
        # Here we do a rolling sequence predict for each time where we have enough history
        lookback = lstm_model.input_shape[1]
        lstm_preds = []
        for i in range(len(df)):
            if i < lookback:
                lstm_preds.append(np.nan)
                continue
            seq = df[lstm_features].iloc[i-lookback:i].values
            seq_scaled = scaler.transform(seq)  # scaler saved was fit on features shape
            seq_scaled = seq_scaled.reshape(1, lookback, seq_scaled.shape[1])
            p = lstm_model.predict(seq_scaled, verbose=0)[0,0]
            lstm_preds.append(p)
        df['proba_lstm'] = lstm_preds
        df['pred_lstm'] = (df['proba_lstm'] > threshold).astype(int)
    else:
        df['proba_lstm'] = np.nan
        df['pred_lstm'] = np.nan

    if lstm_model is not None and ensemble == 'and':
        df['final_signal'] = ((pred_xgb==1) & (df['pred_lstm']==1)).astype(int)
    elif lstm_model is not None and ensemble == 'or':
        df['final_signal'] = ((pred_xgb==1) | (df['pred_lstm']==1)).astype(int)
    else:
        df['final_signal'] = pred_xgb

    # Combine with rule-based filter (SMA + CAPE conditions)
    # Create the rule-based strategy column (example)
    cape_median = df['CAPE'].median()
    df['rule_filter'] = ((df['Close'] > df['SMA200']) & (df['CAPE'] < cape_median) & (df['BondYield'] < df['EarningsYield'])).astype(int)
    df['trade_signal'] = df['final_signal'] * df['rule_filter']

    # Backtest (equally weighted single asset)
    df['market_ret'] = df['Close'].pct_change()
    df['strat_ret'] = df['market_ret'] * df['trade_signal'].shift(1)
    df['equity'] = (1 + df['strat_ret'].fillna(0)).cumprod()
    df['market_equity'] = (1 + df['market_ret'].fillna(0)).cumprod()

    perf = compute_perf(df['equity'])
    perf_market = compute_perf(df['market_equity'])
    return df, perf, perf_market
