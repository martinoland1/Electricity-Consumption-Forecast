
# monthly_temperature_moy_lines.py
# Purpose: Plot monthly average temperatures as LINES by month-of-year (Jan..Dec),
#          with one line per year (different colors assigned automatically by matplotlib).
# Data source: temp.py (expects avg_day_temp with columns:
#   - avg_day_temp_date (date)
#   - hour_day_value    (daily average temperature)
#
# Usage:
#   python monthly_temperature_moy_lines.py
#   (Optionally edit SELECT_YEARS or SELECT_MONTHS below)
#
# Notes:
#   - X-axis is fixed to Jan..Dec (1..12).
#   - PNG saving is commented out; figure is shown interactively.

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calendar
from importlib import import_module

# ----------------------
# User options
# ----------------------
# Choose specific years to compare, e.g., [2023, 2024, 2025].
# If None, include ALL distinct years present in data.
SELECT_YEARS = None

# Optionally restrict to specific months-of-year (1..12). Leave None for all months Jan..Dec.
SELECT_MONTHS = None  # e.g., [6,7,8] for summer

# ----------------------
# Import daily temperatures from temp.py
# ----------------------
try:
    tp = import_module("temp")
except ModuleNotFoundError:
    print("ERROR: Cannot import 'temp.py'. Place this script in the same folder.")
    sys.exit(1)

if not hasattr(tp, "avg_day_temp"):
    print("ERROR: 'temp.py' does not expose 'avg_day_temp'.")
    sys.exit(1)

dft = tp.avg_day_temp.copy()
if dft.empty:
    print("No daily temperature data available. Exiting.")
    sys.exit(0)

# Ensure datetime and build year/month
# UTC-based date in source (or local if configured)
dft["avg_day_temp_date"] = pd.to_datetime(dft["avg_day_temp_date"])
dft["year"] = dft["avg_day_temp_date"].dt.year
dft["month_num"] = dft["avg_day_temp_date"].dt.month

# Monthly averages (UTC months) from daily averages
monthly = (dft
           .groupby(["year", "month_num"], as_index=False)["hour_day_value"]
           .mean()
           .rename(columns={"hour_day_value": "avg_month_temp"}))

# Select years: include ALL years by default (avoid hiding earlier years)
years_sorted = sorted(monthly["year"].unique())
if SELECT_YEARS is None:
    SELECT_YEARS = years_sorted

monthly = monthly[monthly["year"].isin(SELECT_YEARS)].copy()

# Optionally filter months-of-year
if SELECT_MONTHS is not None:
    monthly = monthly[monthly["month_num"].isin(SELECT_MONTHS)].copy()

# Fixed month-of-year domain: Jan..Dec
months_order = list(range(1, 13))
month_labels = [calendar.month_abbr[m] for m in months_order]

# Values per year aligned to months_order
years_order = sorted(SELECT_YEARS)
values_by_year = {}
for y in years_order:
    by_month = monthly[monthly["year"] == y].set_index("month_num")[
        "avg_month_temp"]
    values_by_year[y] = [float(by_month.get(m, np.nan)) for m in months_order]

# Plot LINES
x = np.arange(len(months_order))
fig, ax = plt.subplots(figsize=(12, 6))

for y in years_order:
    yvals = np.array(values_by_year[y], dtype=float)
    ax.plot(x, yvals, marker="o", linewidth=2, label=str(y))

ax.set_xticks(x)
ax.set_xticklabels(month_labels)
ax.set_xlabel("Month (Jan..Dec)")
ax.set_ylabel("Average monthly temperature (°C)")
ax.set_title(
    f"Monthly Average Temperature — Month-of-Year Lines (Years: {', '.join(map(str, years_order))})")
ax.legend(title="Year")
ax.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig("monthly_temperature_moy_lines.png", dpi=120)  # saving optional/commented
if __name__ == "__main__":
    try:
        plt.show()
    except Exception:
        pass
