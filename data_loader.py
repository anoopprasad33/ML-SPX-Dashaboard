# data_loader.py
import yfinance as yf
import pandas as pd
import numpy as np

def load_price(ticker, start="1925-01-01", end=None):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.index = pd.to_datetime(df.index)
    return df[['Open','High','Low','Close','Adj Close','Volume']]

def load_macro(csv_path):
    # expects Date,CAPE,BondYield monthly (or at least monthly frequency)
    macro = pd.read_csv(csv_path, parse_dates=['Date'])
    macro = macro.set_index('Date').sort_index()
    # forward fill missing macro monthly values
    macro = macro.resample('M').ffill()
    return macro

def merge_price_macro(price_df, macro_df):
    # align on business days: reindex price to business days, then forward-fill price?
    # We'll daily-price join monthly macro (ffill macro)
    price = price_df.copy()
    # ensure macro is available daily by forward-filling each day in month
    macro_daily = macro_df.reindex(price.index, method='ffill')
    df = price.join(macro_daily, how='inner')
    return df
