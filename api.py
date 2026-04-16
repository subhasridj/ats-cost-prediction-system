# api.py
from fastapi import FastAPI, File, UploadFile
import pandas as pd
import joblib
import io
import os
from features import compute_aggregates

app = FastAPI(title='ATS MVP API')

MODEL_PATH = 'src/models/ats_cost_model.joblib'
CANONICAL_PATH = 'data/executed_shipments_canonical.csv'

# In-memory aggregates loaded at startup (if canonical exists)
vendor_agg = None
route_agg = None
model = None

@app.on_event('startup')
def startup_event():
    global model, vendor_agg, route_agg
    # load model if available
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            model = None
            print("Warning: could not load model:", e)
    else:
        model = None
        print("No trained model found at", MODEL_PATH)

    # load canonical executed shipments to compute aggregates (if available)
    if os.path.exists(CANONICAL_PATH):
        try:
            df = pd.read_csv(CANONICAL_PATH)
            if df.shape[0] > 0:
                df = compute_aggregates(df)
                # create simple lookup tables
                vendor_agg = df.drop_duplicates('vendor_code').set_index('vendor_code')[[
                    'vendor_avg_cost_per_tonne','vendor_total_volume','vendor_mean_transit','vendor_on_time_pct'
                ]]
                route_agg = df.drop_duplicates(['load_port','discharge_port']).set_index(['load_port','discharge_port'])[
                    ['route_mean_transit','route_mean_wait']
                ]
                print("Loaded historical aggregates for enrichment (vendors, routes).")
            else:
                vendor_agg = None
                route_agg = None
                print("Canonical file exists but is empty.")
        except Exception as e:
            vendor_agg = None
            route_agg = None
            print("Warning: failed to compute aggregates from canonical file:", e)
    else:
        vendor_agg = None
        route_agg = None
        print("No canonical executed shipments file found at", CANONICAL_PATH, "- enrichment will be limited.")

def safe_lookup_vendor(vendor_code: str):
    """
    Return dict of vendor aggregates or None if not available.
    """
    global vendor_agg
    if vendor_agg is None or vendor_code not in vendor_agg.index:
        return None
    row = vendor_agg.loc[vendor_code]
    return {
        'vendor_avg_cost_per_tonne': row.get('vendor_avg_cost_per_tonne'),
        'vendor_total_volume': row.get('vendor_total_volume'),
        'vendor_mean_transit': row.get('vendor_mean_transit'),
        'vendor_on_time_pct': row.get('vendor_on_time_pct')
    }

def safe_lookup_route(load_port: str, discharge_port: str):
    global route_agg
    if route_agg is None:
        return None
    key = (load_port, discharge_port)
    try:
        row = route_agg.loc[key]
        return {'route_mean_transit': row.get('route_mean_transit'), 'route_mean_wait': row.get('route_mean_wait')}
    except Exception:
        return None

@app.post('/score_tender')
async def score_tender(file: UploadFile = File(...)):
    """
    Accepts a tender_bids CSV with columns:
      bid_id,vendor_code,quoted_price,quantity,load_port,discharge_port,vessel_class_planned

    Returns ranked bids and enrichment used. Does NOT invent synthetic defaults.
    """
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        return {'error': f'Failed to parse uploaded CSV: {e}'}

    required = {'bid_id','vendor_code','quoted_price','quantity','load_port','discharge_port'}
    missing = required - set(df.columns.tolist())
    if missing:
        return {'error': f'Uploaded tender CSV missing required columns: {sorted(list(missing))}'}

    # Convert numeric
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['quoted_price'] = pd.to_numeric(df['quoted_price'], errors='coerce')

    # Enrich per-row using historical aggregates if available
    enrich_rows = []
    # compute global fallback means from vendor_agg / route_agg if present
    # These are data-driven fallbacks only if historical data exists; otherwise left as NA and model may not run.
    vendor_mean_transit_global = None
    vendor_on_time_pct_global = None
    route_mean_transit_global = None
    route_mean_wait_global = None

    if vendor_agg is not None:
        vendor_mean_transit_global = vendor_agg['vendor_mean_transit'].dropna().mean()
        vendor_on_time_pct_global = vendor_agg['vendor_on_time_pct'].dropna().mean()
    if route_agg is not None:
        route_mean_transit_global = route_agg['route_mean_transit'].dropna().mean()
        route_mean_wait_global = route_agg['route_mean_wait'].dropna().mean()

    enriched_feature_rows = []
    for _, r in df.iterrows():
        v = safe_lookup_vendor(r['vendor_code'])
        rt = safe_lookup_route(r['load_port'], r['discharge_port'])

        # Use historical values if available, otherwise fallback to data-driven global means; if those are None, use NaN.
        vendor_mean_transit = v.get('vendor_mean_transit') if v is not None else vendor_mean_transit_global
        vendor_on_time_pct = v.get('vendor_on_time_pct') if v is not None else vendor_on_time_pct_global
        route_mean_transit = rt.get('route_mean_transit') if rt is not None else route_mean_transit_global
        route_mean_wait = rt.get('route_mean_wait') if rt is not None else route_mean_wait_global

        enriched_feature_rows.append({
            'bid_id': r['bid_id'],
            'vendor_code': r['vendor_code'],
            'quoted_price': r['quoted_price'],
            'quantity_shipped': r['quantity'],
            'vendor_mean_transit': vendor_mean_transit,
            'vendor_on_time_pct': vendor_on_time_pct,
            'route_mean_transit': route_mean_transit,
            'route_mean_wait': route_mean_wait
        })

    feat_df = pd.DataFrame(enriched_feature_rows)
    # If model missing, inform user (do not invent predictions)
    if model is None:
        return {
            'error': 'No trained model found. Train model first using real canonical executed shipments (see README).',
            'enriched_bids_preview': feat_df.head(20).to_dict(orient='records')
        }

    # Prepare feature matrix for model - require that no crucial features are entirely NaN
    feature_cols = ['vendor_mean_transit','vendor_on_time_pct','route_mean_transit','route_mean_wait','quantity_shipped']
    X = feat_df[feature_cols].copy()

    # If any column is fully null, we cannot reliably predict — report to caller
    fully_null_cols = [c for c in feature_cols if X[c].isna().all()]
    if fully_null_cols:
        return {
            'error': 'Insufficient historical aggregates to compute features required by the model. Missing columns:',
            'missing_feature_columns': fully_null_cols,
            'note': 'Provide more executed shipment history in data/executed_shipments_canonical.csv or increase coverage of vendor/route history.'
        }

    # Fill remaining NaNs with column means (data-driven) - still not synthetic: based on available historical data
    X = X.fillna(X.mean())

    # Predict cost per tonne using model
    try:
        preds = model.predict(X)
    except Exception as e:
        return {'error': f'Model prediction failed: {e}'}

    feat_df['pred_cost_per_tonne'] = preds
    feat_df['expected_total_sea_leg_cost'] = feat_df['pred_cost_per_tonne'] * feat_df['quantity_shipped']

    # Final score: combine quoted price with expected sea-leg cost (both should be provided). If quoted_price is NaN, mark row.
    def combine_score(row):
        qp = row.get('quoted_price')
        sea_cost = row.get('expected_total_sea_leg_cost')
        if pd.isna(qp) or pd.isna(sea_cost):
            return pd.NA
        return qp + sea_cost

    feat_df['final_score'] = feat_df.apply(combine_score, axis=1)

    # Sort by final score where it's available; keep other rows at the bottom
    ranked = feat_df.copy()
    ranked = ranked.sort_values(by=['final_score'], na_position='last').reset_index(drop=True)

    # Return ranked bids and the enrichment used (small sample)
    return {
        'ranked_bids': ranked[['bid_id','vendor_code','quoted_price','pred_cost_per_tonne','expected_total_sea_leg_cost','final_score']].to_dict(orient='records'),
        'notes': 'All enrichments are derived from historical executed shipments (data/executed_shipments_canonical.csv). No synthetic defaults were used.'
    }
