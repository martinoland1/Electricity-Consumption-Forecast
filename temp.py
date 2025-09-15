# pip install meteostat pandas

from datetime import datetime, timedelta, timezone
import pandas as pd
from meteostat import Hourly, Point

# -----------------------------
# Config
# -----------------------------
# Use local time for daily aggregation so days match Estonia's calendar
USE_LOCAL_TIME_FOR_DAILY = True
LOCAL_TZ = 'Europe/Tallinn'

# -----------------------------
# Time window: last 2 years up to the last full hour (UTC)
# -----------------------------
now_utc_aw = datetime.now(timezone.utc).replace(
    minute=0, second=0, microsecond=0)
# Meteostat expects NAIVE datetimes when you pass timezone='UTC'
end = (now_utc_aw - timedelta(hours=1)).replace(tzinfo=None)
start = end - timedelta(days=365*2)

# -----------------------------
# Points in Estonia (we'll average across these)
# -----------------------------
points = {
    "Tallinn":    Point(59.4370, 24.7536),
    "Tartu":      Point(58.3776, 26.7290),
    "Pärnu":      Point(58.3859, 24.4971),
    "Narva":      Point(59.3793, 28.2000),
    "Kuressaare": Point(58.2528, 22.4869),
}

# -----------------------------
# Fetch hourly temperature per point and combine
# -----------------------------
frames = []
for name, pt in points.items():
    # Use naive start/end + timezone='UTC' to avoid tz conflicts
    df = Hourly(pt, start, end, timezone='UTC').fetch()
    if df.empty or 'temp' not in df.columns:
        continue
    # Keep only temperature and rename column to the point's name
    frames.append(df[['temp']].rename(columns={'temp': name}))

if not frames:
    raise RuntimeError(
        "No hourly temperature data returned by Meteostat for the selected Estonian points.")

# Align on a single hourly UTC timeline and compute the average across points
combined = pd.concat(frames, axis=1).sort_index()
avg_series = combined.mean(axis=1, skipna=True)

# -----------------------------
# Build hourly dataframe
# -----------------------------
avg_hourly_temp = avg_series.reset_index()
avg_hourly_temp.columns = ['hour_temp_time',
                           'hour_temp_value']  # required names

# Optional rounding
# avg_hourly_temp['hour_temp_value'] = avg_hourly_temp['hour_temp_value'].round(2)

# -----------------------------
# (Optional) convert to local time for daily aggregation
# -----------------------------
if USE_LOCAL_TIME_FOR_DAILY:
    # hour_temp_time is tz-aware (UTC). Convert to local time for correct day boundaries.
    avg_hourly_temp['hour_temp_time'] = avg_hourly_temp['hour_temp_time'].dt.tz_convert(
        LOCAL_TZ)

# -----------------------------
# Quality check (hourly)
# -----------------------------
rows_imported_hourly = len(avg_hourly_temp)
missing_counts_hourly = avg_hourly_temp.isna().sum()
all_values_present_hourly = (missing_counts_hourly.sum() == 0)

# -----------------------------
# Print hourly outputs
# -----------------------------
print("\n=== HOURLY: DataFrame preview (first 10 rows) ===")
print(avg_hourly_temp.head(10).to_string(index=False))

print("\n=== HOURLY: Column names ===")
print(list(avg_hourly_temp.columns))

print("\n=== HOURLY: Shape (rows, columns) ===")
print(avg_hourly_temp.shape)

print("\n=== HOURLY: Quality check ===")
print(f"- Rows imported: {rows_imported_hourly}")
for col, miss in missing_counts_hourly.items():
    print(f"- Missing values in '{col}': {miss}")
print(f"- All columns have values: {all_values_present_hourly}")

# -----------------------------
# Save hourly CSV (commented out)
# -----------------------------
# csv_path_hourly = "avg_hourly_temp.csv"
# avg_hourly_temp.to_csv(csv_path_hourly, index=False)
# print(f"\nSaved HOURLY CSV to: {csv_path_hourly}")

# -----------------------------
# Build daily averages from hourly dataframe
# -----------------------------
# Ensure datetime dtype (keep timezone info if present)
avg_hourly_temp['hour_temp_time'] = pd.to_datetime(
    avg_hourly_temp['hour_temp_time'], errors='coerce')

# Extract date for grouping (local date if converted above)
avg_hourly_temp['avg_day_temp_date'] = avg_hourly_temp['hour_temp_time'].dt.date

# Group by date and compute mean of the hourly values
avg_day_temp = (
    avg_hourly_temp
    .groupby('avg_day_temp_date', as_index=False)['hour_temp_value']
    .mean()
    .rename(columns={'hour_temp_value': 'hour_day_value'})
)

# Optional rounding
# avg_day_temp['hour_day_value'] = avg_day_temp['hour_day_value'].round(2)

# -----------------------------
# Quality check (daily)
# -----------------------------
rows_imported_daily = len(avg_day_temp)
missing_counts_daily = avg_day_temp.isna().sum()
all_values_present_daily = (missing_counts_daily.sum() == 0)

# -----------------------------
# Print daily outputs
# -----------------------------
print("\n=== DAILY: DataFrame preview (first 10 rows) ===")
print(avg_day_temp.head(10).to_string(index=False))

print("\n=== DAILY: Column names ===")
print(list(avg_day_temp.columns))

print("\n=== DAILY: Shape (rows, columns) ===")
print(avg_day_temp.shape)

print("\n=== DAILY: Quality check ===")
print(f"- Rows imported: {rows_imported_daily}")
for col, miss in missing_counts_daily.items():
    print(f"- Missing values in '{col}': {miss}")
print(f"- All columns have values: {all_values_present_daily}")

# -----------------------------
# Save daily CSV (commented out)
# -----------------------------
# csv_path_daily = "avg_day_temp.csv"
# avg_day_temp.to_csv(csv_path_daily, index=False)
# print(f"\nSaved DAILY CSV to: {csv_path_daily}")
