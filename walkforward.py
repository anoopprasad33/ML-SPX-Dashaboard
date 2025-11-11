# walkforward.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
from datetime import timedelta

def walk_forward_train(df, features, target_col='Target_up',
                       start_train_period_years=20, test_period_days=365,
                       step_days=90, outdir='models_xgb'):
    """
    Rolling walk-forward:
    - initial training window = start_train_period_years (years)
    - then iteratively: train on available historical window, test next test_period_days,
      then expand window by step_days and repeat.
    Saves each model and aggregates OOS predictions.
    """
    os.makedirs(outdir, exist_ok=True)
    df = df.copy().sort_index()
    dates = df.index

    start_train_days = int(start_train_period_years * 365)
    i0 = 0
    results = []
    all_oos = pd.DataFrame(index=df.index)

    while True:
        train_start = dates[0]
        train_end_idx = i0 + start_train_days
        if train_end_idx >= len(dates) - test_period_days:
            break
        train_end = dates[train_end_idx]
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_period_days)

        train_mask = (df.index <= train_end)
        test_mask = (df.index > train_end) & (df.index <= test_end)
        if df.loc[test_mask].empty:
            break

        X_train = df.loc[train_mask, features]
        y_train = df.loc[train_mask, target_col]
        X_test = df.loc[test_mask, features]
        y_test = df.loc[test_mask, target_col]

        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
            eval_metric='logloss', verbosity=0
        )
        model.fit(X_train, y_train)

        preds_proba = model.predict_proba(X_test)[:,1]
        preds = (preds_proba > 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, preds_proba)

        results.append({
            'train_end': train_end, 'test_start': test_start, 'test_end': test_end,
            'acc': acc, 'auc': auc, 'n_train': len(X_train), 'n_test': len(X_test)
        })

        # save model with train_end timestamp
        fname = os.path.join(outdir, f"xgb_{train_end.strftime('%Y%m%d')}.joblib")
        joblib.dump(model, fname)

        # store oos probs
        s = pd.Series(preds_proba, index=X_test.index, name=f"proba_{train_end.strftime('%Y%m%d')}")
        all_oos = all_oos.join(s, how='outer')

        # step forward
        i0 += step_days

    results_df = pd.DataFrame(results)
    return results_df, all_oos
