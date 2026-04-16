# train.py
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from features import compute_aggregates
import os
import numpy as np

MIN_ROWS_TO_TRAIN = 20  # conservative; you can lower but model quality will be poor

def train(input_csv, outdir):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input canonical CSV not found: {input_csv}. Run ETL with real SAP exports first.")

    df = pd.read_csv(input_csv)
    if df.shape[0] < MIN_ROWS_TO_TRAIN:
        print(f"Insufficient historical rows to train a reliable model: found {df.shape[0]} rows, need at least {MIN_ROWS_TO_TRAIN}.")
        print("Please provide more executed shipment history (12+ months recommended) exported from SAP.")
        return

    df = compute_aggregates(df)

    # simple cleaning: drop NA target
    if 'total_settlement_cost' not in df.columns:
        raise ValueError("Canonical file missing 'total_settlement_cost' column required for training.")
    df = df.dropna(subset=['total_settlement_cost', 'quantity_shipped'])

    # target: cost per tonne (use numeric)
    df['cost_per_tonne'] = pd.to_numeric(df['total_settlement_cost'], errors='coerce') / pd.to_numeric(df['quantity_shipped'], errors='coerce')
    df = df.dropna(subset=['cost_per_tonne'])

    # features and target
    features = [
        'vendor_mean_transit','vendor_on_time_pct','route_mean_transit','route_mean_wait',
        'quantity_shipped'
    ]
    # ensure features exist
    for f in features:
        if f not in df.columns:
            df[f] = 0.0

    X = df[features].fillna(0.0)
    y = df['cost_per_tonne'].astype(float)

    # use time-based split if we have a date-like column; otherwise simple split
    tscv = TimeSeriesSplit(n_splits=3)
    models = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        params = {'objective':'regression','metric':'mae','verbosity':-1}
        model = lgb.train(params, train_data, valid_sets=[val_data], early_stopping_rounds=10, num_boost_round=200)
        y_pred = model.predict(X_val)
        print('val MAE:', mean_absolute_error(y_val, y_pred))
        models.append(model)

    os.makedirs(outdir, exist_ok=True)
    # persist the last model
    joblib.dump(models[-1], f"{outdir}/ats_cost_model.joblib")
    print('Saved model to', f"{outdir}/ats_cost_model.joblib")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/executed_shipments_canonical.csv')
    parser.add_argument('--outdir', default='src/models')
    args = parser.parse_args()
    train(args.input, args.outdir)
