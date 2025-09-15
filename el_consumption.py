import requests
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import math

# --- Helper: robust UTC datetime parsing ---


def smart_to_datetime(ts):
    """
    Convert a timestamp (ISO string or UNIX epoch in s/ms) to pandas UTC datetime.
    Returns pandas.NaT if parsing fails.
    """
    if isinstance(ts, str):
        return pd.to_datetime(ts, utc=True, errors="coerce")
    if isinstance(ts, (int, float)) and not math.isnan(ts):
        unit = "ms" if ts > 10_000_000_000 else "s"
        return pd.to_datetime(ts, unit=unit, utc=True, errors="coerce")
    return pd.NaT

# --- JSON parser: collect only (timestamp, consumption) pairs ---


def extract_timestamp_and_consumption(obj):
    """
    Recursively walk the JSON and collect (timestamp, consumption) tuples.
    Expects fields exactly named 'timestamp' and 'consumption'.
    """
    rows = []

    def visit(node):
        if isinstance(node, dict):
            if "timestamp" in node and "consumption" in node:
                rows.append((node["timestamp"], node["consumption"]))
            for v in node.values():
                visit(v)
        elif isinstance(node, list):
            for item in node:
                visit(item)
    visit(obj)
    return rows

# --- Fetch one time chunk with simple retry/backoff ---


def fetch_chunk(start_dt_utc: datetime, end_dt_utc: datetime, retries: int = 3, backoff: float = 2.0) -> pd.DataFrame:
    """
    Fetch [start, end) from the Elering API and return a DataFrame with columns:
      - sum_cons_time (UTC datetime)
      - sum_el_hourly_value (numeric)
    """
    start_str = start_dt_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end_str = end_dt_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    url = f"https://dashboard.elering.ee/api/system/with-plan?start={start_str}&end={end_str}"

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            pairs = extract_timestamp_and_consumption(data)
            df = pd.DataFrame(
                pairs, columns=["sum_cons_time", "sum_el_hourly_value"])

            if not df.empty:
                # Parse types
                df["sum_cons_time"] = df["sum_cons_time"].apply(
                    smart_to_datetime)
                df["sum_el_hourly_value"] = pd.to_numeric(
                    df["sum_el_hourly_value"], errors="coerce")
                # Drop rows with invalid timestamps
                df = df.dropna(subset=["sum_cons_time"]).reset_index(drop=True)
            return df

        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (2 ** attempt))  # exponential backoff

# --- Imputation: fill missing values as mean(prev_valid, next_valid) ---


def impute_missing_mean_of_neighbors(df: pd.DataFrame) -> pd.DataFrame:
    """
    For rows with NaN in sum_el_hourly_value, fill using the average of the
    previous and next non-NaN values (chronological order).
    Adds a boolean 'imputed' column indicating which rows were filled.
    Edge/lonely gaps (no valid neighbor on one side) remain NaN.
    """
    if df.empty:
        df["imputed"] = False
        return df

    df = df.sort_values("sum_cons_time").reset_index(drop=True)

    prev_valid = df["sum_el_hourly_value"].ffill()
    next_valid = df["sum_el_hourly_value"].bfill()

    to_impute = df["sum_el_hourly_value"].isna(
    ) & prev_valid.notna() & next_valid.notna()
    fill_values = (prev_valid[to_impute] + next_valid[to_impute]) / 2.0

    df.loc[to_impute, "sum_el_hourly_value"] = fill_values
    df["imputed"] = False
    df.loc[to_impute, "imputed"] = True
    return df

# --- Data quality report for the hourly dataframe ---


def quality_report(hourly_df: pd.DataFrame):
    """
    Print a data quality report for sum_hourly_el_consumption:
    - total rows, present/missing counts
    - duplicate timestamps
    - date range
    - imputation stats (how many filled, how many still missing)
    """
    total = len(hourly_df)
    nonnull_ts = hourly_df["sum_cons_time"].notna().sum() if total else 0
    nonnull_val = hourly_df["sum_el_hourly_value"].notna(
    ).sum() if total else 0
    null_ts = total - nonnull_ts
    null_val = total - nonnull_val
    dup = hourly_df.duplicated(
        subset=["sum_cons_time"], keep=False).sum() if total else 0
    imputed_count = int(hourly_df["imputed"].sum()) if (
        "imputed" in hourly_df.columns and total) else 0
    still_missing = int(
        hourly_df["sum_el_hourly_value"].isna().sum()) if total else 0

    print("\n=== Data Quality Report (hourly) ===")
    print(f"Total rows: {total}")
    print(f"sum_cons_time present: {nonnull_ts} | missing: {null_ts}")
    print(f"sum_el_hourly_value present: {nonnull_val} | missing: {null_val}")
    print(f"Duplicate timestamps: {dup}")
    if total:
        print(
            f"Date range: {hourly_df['sum_cons_time'].min()} … {hourly_df['sum_cons_time'].max()}")
    print(f"Imputed (mean of neighbors): {imputed_count}")
    print(f"Remaining missing after imputation: {still_missing}")


# === Define 24-month window split into 12 + 12 ===
end_all = datetime.utcnow()
mid_point = end_all - relativedelta(months=12)
start_all = end_all - relativedelta(months=24)

# Fetch both 12-month chunks
df1 = fetch_chunk(start_all, mid_point)
df2 = fetch_chunk(mid_point, end_all)

# === Build the hourly dataframe (requested name) ===
sum_hourly_el_consumption = (
    pd.concat([df1, df2], ignore_index=True)
    if (df1 is not None and df2 is not None)
    else pd.DataFrame(columns=["sum_cons_time", "sum_el_hourly_value"])
)

if not sum_hourly_el_consumption.empty:
    sum_hourly_el_consumption = (
        sum_hourly_el_consumption
        .drop_duplicates(subset=["sum_cons_time"])
        .sort_values("sum_cons_time")
        .reset_index(drop=True)
    )

# Impute hourly missing values using neighbor mean
sum_hourly_el_consumption = impute_missing_mean_of_neighbors(
    sum_hourly_el_consumption)

# Report quality on hourly data
quality_report(sum_hourly_el_consumption)

# --- Optional save (commented out as requested) ---
# out_path = "elering_consumption_hourly_last24months_imputed.csv"
# sum_hourly_el_consumption.to_csv(out_path, index=False)
# print(f"Saved hourly dataset: {out_path}")

# === Build the daily dataframe (requested name) ========================
if sum_hourly_el_consumption.empty:
    sum_daily_el_consumption = pd.DataFrame(
        columns=["sum_cons_date", "sum_el_daily_value"])
else:
    # Set time index for daily resampling
    hourly_idx = sum_hourly_el_consumption.set_index("sum_cons_time")

    # Sum hourly consumption per calendar day (UTC); min_count=1 keeps all-NaN days as NaN
    sum_daily_el_consumption = (
        hourly_idx["sum_el_hourly_value"]
        .resample("D")
        .sum(min_count=1)
        .rename("sum_el_daily_value")
        .reset_index()
    )

    # Keep only date part in its own column
    sum_daily_el_consumption["sum_cons_date"] = sum_daily_el_consumption["sum_cons_time"].dt.date
    sum_daily_el_consumption = (
        sum_daily_el_consumption[["sum_cons_date", "sum_el_daily_value"]]
        .sort_values("sum_cons_date")
        .reset_index(drop=True)
    )

# Quick preview
print("\n=== Hourly preview ===")
print(sum_hourly_el_consumption.head())
print("\n=== Daily preview ===")
print(sum_daily_el_consumption.head())

# --- Optional save (commented out) ---
# daily_out = "elering_consumption_daily_last24months.csv"
# sum_daily_el_consumption.to_csv(daily_out, index=False)
# print(f"Saved daily dataset: {daily_out}")

# --- Show dtypes for both DataFrames ---
print("\n=== Column dtypes: sum_hourly_el_consumption ===")
print(sum_hourly_el_consumption.dtypes)

print("\n=== Column dtypes: sum_daily_el_consumption ===")
print(sum_daily_el_consumption.dtypes)

# (Optional) more detailed overview:
print("\n=== .info() (hourly) ===")
sum_hourly_el_consumption.info()

print("\n=== .info() (daily) ===")
sum_daily_el_consumption.info()
