# lstm_model.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import joblib
import os

def build_lstm(input_shape, units=64, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def prepare_sequences(df, features, lookback=60):
    Xs = []
    ys = []
    for i in range(lookback, len(df)):
        Xs.append(df[features].iloc[i-lookback:i].values)
        ys.append(df['Target_up'].iloc[i])
    Xs = np.array(Xs)
    ys = np.array(ys)
    return Xs, ys

def train_lstm(df, features, lookback=60, epochs=50, batch_size=32, outdir='models_lstm'):
    os.makedirs(outdir, exist_ok=True)
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df_scaled[features])
    joblib.dump(scaler, os.path.join(outdir, 'scaler.pkl'))

    X, y = prepare_sequences(df_scaled, features, lookback=lookback)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=2)

    model.save(os.path.join(outdir, 'lstm_model.h5'))
    return model
