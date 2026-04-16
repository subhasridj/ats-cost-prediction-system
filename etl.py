# etl.py
import argparse
import pandas as pd
import os
from utils import parse_dates, read_csv_safe

PO_COLS = ['po_number','vendor_code','material_code','incoterm','quantity','currency','po_date']
SHIP_COLS = ['shipment_id','po_number','vessel_name','vessel_type','vessel_dwt_class',
             'load_port','discharge_port','etd','eta','atd','ata','quantity_shipped']
FREIGHT_COLS = ['shipment_id','ocean_freight_amount','surcharges_total','demurrage_amount','total_settlement_cost']

def build_canonical(po_path, ship_path, freight_path, out_file):
    # Read and validate
    if not os.path.exists(po_path) or not os.path.exists(ship_path) or not os.path.exists(freight_path):
        missing = [p for p in [po_path, ship_path, freight_path] if not os.path.exists(p)]
        raise FileNotFoundError(f"Required CSVs missing: {missing}. Place real SAP-export CSVs at these paths.")

    po, missing = read_csv_safe(po_path, required_cols=PO_COLS, nrows=10)
    if missing:
        raise ValueError(f"po_header.csv is missing columns: {missing}")

    ship, missing = read_csv_safe(ship_path, required_cols=SHIP_COLS, nrows=10)
    if missing:
        raise ValueError(f"shipments.csv is missing columns: {missing}")

    freight, missing = read_csv_safe(freight_path, required_cols=FREIGHT_COLS, nrows=10)
    if missing:
        raise ValueError(f"freight_settlement.csv is missing columns: {missing}")

    # Parse dates
    po = parse_dates(po)
    ship = parse_dates(ship)

    # Basic joins
    df = ship.merge(po, on='po_number', how='left', validate='m:1')
    df = df.merge(freight, on='shipment_id', how='left', validate='1:1')

    # Ensure required numeric columns exist and are numeric
    for col in ['quantity_shipped', 'demurrage_amount', 'total_settlement_cost']:
        if col not in df.columns:
            df[col] = pd.NA
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # compute safe canonical features (use safe operations)
    df['sea_transit_days'] = None
    try:
        df.loc[:, 'sea_transit_days'] = (df['ata'] - df['atd']).dt.total_seconds() / (3600*24)
    except Exception:
        # timestamps might be missing; keep NaN
        df['sea_transit_days'] = pd.NA

    df['port_wait_days'] = None
    try:
        df.loc[:, 'port_wait_days'] = (df['atd'] - df['eta']).dt.total_seconds() / (3600*24)
    except Exception:
        df['port_wait_days'] = pd.NA

    # Avoid division by zero or NA
    def dem_per_1000(row):
        try:
            q = float(row.get('quantity_shipped') or 0)
            dem = float(row.get('demurrage_amount') or 0)
            if q <= 0:
                return pd.NA
            return dem / (q / 1000)
        except Exception:
            return pd.NA

    df['demurrage_per_1000t'] = df.apply(dem_per_1000, axis=1)

    # final check: write only columns existing
    df.to_csv(out_file, index=False)
    print(f'Wrote canonical file to {out_file} (rows: {len(df)})')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='data/sample')
    parser.add_argument('--out-file', default='data/executed_shipments_canonical.csv')
    args = parser.parse_args()

    po_path = os.path.join(args.input_dir, "po_header.csv")
    ship_path = os.path.join(args.input_dir, "shipments.csv")
    freight_path = os.path.join(args.input_dir, "freight_settlement.csv")

    build_canonical(po_path, ship_path, freight_path, args.out_file)
