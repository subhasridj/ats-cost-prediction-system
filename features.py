# features.py
import pandas as pd

VENDOR_AGG_COLS = ['vendor_code']

def compute_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute vendor and route aggregates. Requires that df has at least one executed shipment row.
    Returns df enriched with vendor_* and route_* columns.
    """
    if df is None or df.shape[0] == 0:
        raise ValueError("compute_aggregates requires a non-empty executed shipments dataframe with real data.")

    # ensure numeric
    if 'total_settlement_cost' in df.columns:
        df['total_settlement_cost'] = pd.to_numeric(df['total_settlement_cost'], errors='coerce')
    else:
        df['total_settlement_cost'] = pd.NA

    if 'quantity_shipped' in df.columns:
        df['quantity_shipped'] = pd.to_numeric(df['quantity_shipped'], errors='coerce')
    else:
        df['quantity_shipped'] = pd.NA

    # per-vendor aggregates (use safe aggregations)
    vendor = df.groupby('vendor_code', dropna=True).agg(
        vendor_total_cost=('total_settlement_cost', 'sum'),
        vendor_total_volume=('quantity_shipped','sum'),
        vendor_mean_transit=('sea_transit_days','mean'),
        vendor_on_time_pct=('ata', lambda s: float(s.notna().sum()) / max(1, len(s)))
    ).reset_index()

    # compute vendor_avg_cost_per_tonne safely
    def safe_cost_per_tonne(row):
        try:
            vol = float(row['vendor_total_volume'] or 0)
            if vol <= 0:
                return pd.NA
            return float(row['vendor_total_cost']) / vol
        except Exception:
            return pd.NA

    vendor['vendor_avg_cost_per_tonne'] = vendor.apply(safe_cost_per_tonne, axis=1)

   + # route aggregates
    route = df.groupby(['load_port','discharge_port'], dropna=True).agg(
        route_mean_transit=('sea_transit_days','mean'),
        route_mean_wait=('port_wait_days','mean')
    ).reset_index()

    # join back to main df
    df = df.merge(vendor[['vendor_code','vendor_avg_cost_per_tonne','vendor_total_volume','vendor_mean_transit','vendor_on_time_pct']],
                  on='vendor_code', how='left')
    df = df.merge(route, on=['load_port','discharge_port'], how='left')

    return df
