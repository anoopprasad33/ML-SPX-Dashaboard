# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib, os
from data_loader import load_price, load_macro, merge_price_macro
from features import add_features, make_targets
from walkforward import walk_forward_train
from lstm_model import train_lstm
from backtest import apply_strategy
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="ML+Macro Strategy Dashboard")

st.title("ML + Macro Strategy — Backtest & Deploy")

with st.sidebar:
    ticker = st.text_input("Ticker (Yahoo)", value="^GSPC")
    macro_csv = st.file_uploader("Upload macro_data.csv (Date,CAPE,BondYield)", type=['csv'])
    run_mode = st.radio("Mode", ['Train Walk-Forward XGBoost', 'Train LSTM', 'Backtest using saved models', 'Show Equity Curves'])

if macro_csv is None:
    st.warning("Please upload macro_data.csv to proceed (monthly Date,CAPE,BondYield).")
    st.stop()

macro_path = os.path.join("uploaded_macro.csv")
with open(macro_path, "wb") as f:
    f.write(macro_csv.getbuffer())

price = load_price(ticker)
macro = load_macro(macro_path)
df = merge_price_macro(price, macro)
df = add_features(df)
df = make_targets(df)

st.write("Data loaded:", df.shape)
st.dataframe(df.tail(5))

if run_mode == 'Train Walk-Forward XGBoost':
    st.info("Running walk-forward training (this may take a while)...")
    features = ['SMA200','Momentum_30','Momentum_90','Volatility_30','CAPE','BondYield','EarningsYield']
    results_df, all_oos = walk_forward_train(df, features, start_train_period_years=15, test_period_days=365, step_days=180)
    st.write(results_df)
    st.line_chart(all_oos.fillna(method='ffill'))
    st.success("Walk-forward complete. Models saved to models_xgb/")

if run_mode == 'Train LSTM':
    st.info("Training LSTM (this could take long depending on epochs)...")
    features_lstm = ['Returns_1','Momentum_30','Momentum_90','Volatility_30','CAPE','BondYield','EarningsYield']
    model = train_lstm(df, features_lstm, lookback=60, epochs=20, batch_size=64, outdir='models_lstm')
    st.success("LSTM saved to models_lstm/")

if run_mode == 'Backtest using saved models':
    st.info("Loading latest XGB model (models_xgb/) and LSTM (if present)")
    xgb_dir = 'models_xgb'
    lstm_dir = 'models_lstm'
    if not os.path.exists(xgb_dir):
        st.error("No XGBoost models found in models_xgb/")
    else:
        latest = sorted(os.listdir(xgb_dir))[-1]
        xgb_model = joblib.load(os.path.join(xgb_dir, latest))
        lstm_model = None
        scaler = None
        lstm_features = None
        if os.path.exists(lstm_dir):
            from tensorflow.keras.models import load_model
            lstm_model = load_model(os.path.join(lstm_dir, 'lstm_model.h5'))
            scaler = joblib.load(os.path.join(lstm_dir, 'scaler.pkl'))
            lstm_features = ['Returns_1','Momentum_30','Momentum_90','Volatility_30','CAPE','BondYield','EarningsYield']
        df_bt, perf, perf_market = apply_strategy(df, xgb_model, lstm_model, scaler, lstm_features, threshold=0.5, ensemble='and')
        st.write("Strategy perf:", perf)
        st.write("Market perf:", perf_market)
        st.line_chart(pd.concat([df_bt['equity'], df_bt['market_equity']], axis=1).fillna(method='ffill'))

if run_mode == 'Show Equity Curves':
    st.line_chart(pd.concat([df['Close'].pct_change().cumsum().apply(np.exp),], axis=1).fillna(0))
