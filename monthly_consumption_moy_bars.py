
# monthly_consumption_moy_bars.py
# Purpose: Plot electricity consumption as side-by-side BARS by month-of-year (Jan..Dec),
#          with a separate color for each year (matplotlib assigns colors automatically).
# Data source: el_consumption_weekday.py (expects sum_daily_el_consumption with UTC dates)
#
# Usage:
#   python monthly_consumption_moy_bars.py
#   (Optionally edit SELECT_YEARS or SELECT_MONTHS below)
#
# Notes:
#   - X-axis is fixed to Jan..Dec (1..12). If some months are missing in data, bars for those months won't be drawn.
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
# Choose specific years to compare (e.g., [2024, 2025]).
# If None, the script will auto-pick the last two distinct years present in data.
SELECT_YEARS = None  # e.g., [2024, 2025]

# Optionally restrict to specific months-of-year: integers 1..12. Leave None for all months Jan..Dec.
# (Since X-axis is Jan..Dec, this is usually not needed. Example: only summer months -> [6,7,8])
SELECT_MONTHS = None

# ----------------------
# Import daily consumption (UTC) from el_consumption_weekday
# ----------------------
try:
    ec = import_module("el_consumption_weekday")
except ModuleNotFoundError:
    print("ERROR: Cannot import 'el_consumption_weekday.py'. Place this script in the same folder.")
    sys.exit(1)

if not hasattr(ec, "sum_daily_el_consumption"):
    print("ERROR: 'el_consumption_weekday.py' does not expose 'sum_daily_el_consumption'.")
    sys.exit(1)

dfd = ec.sum_daily_el_consumption.copy()
if dfd.empty:
    print("No daily data available. Exiting.")
    sys.exit(0)

# Ensure datetime and build year/month
dfd["sum_cons_date"] = pd.to_datetime(
    dfd["sum_cons_date"])  # UTC-based date in source
dfd["year"] = dfd["sum_cons_date"].dt.year
dfd["month_num"] = dfd["sum_cons_date"].dt.month

# Monthly totals (UTC months)
monthly = (dfd
           .groupby(["year", "month_num"], as_index=False)["sum_el_daily_value"]
           .sum()
           .rename(columns={"sum_el_daily_value": "sum_el_monthly_value"}))

# Select years: default to the last two distinct years present
years_sorted = sorted(monthly["year"].unique())
if SELECT_YEARS is None:
    SELECT_YEARS = years_sorted  # kõik olemasolevad aastad


monthly = monthly[monthly["year"].isin(SELECT_YEARS)].copy()

# Optionally filter months
if SELECT_MONTHS is not None:
    monthly = monthly[monthly["month_num"].isin(SELECT_MONTHS)].copy()

# Fixed month-of-year domain: 1..12
months_order = list(range(1, 13))
month_labels = [calendar.month_abbr[m] for m in months_order]

# Values per year aligned to months_order
years_order = sorted(SELECT_YEARS)
values_by_year = {}
for y in years_order:
    by_month = monthly[monthly["year"] == y].set_index("month_num")[
        "sum_el_monthly_value"]
    values_by_year[y] = [float(by_month.get(m, np.nan)) for m in months_order]

# Plot side-by-side BARS
x = np.arange(len(months_order))
n_years = max(1, len(years_order))
bar_width = 0.8 / n_years
offsets = [(i - (n_years - 1) / 2) * bar_width for i in range(n_years)]

fig, ax = plt.subplots(figsize=(12, 6))

for i, y in enumerate(years_order):
    yvals = np.array(values_by_year[y], dtype=float)
    ax.bar(x + offsets[i], yvals, width=bar_width, label=str(y))

ax.set_xticks(x)
ax.set_xticklabels(month_labels)
ax.set_xlabel("Month (Jan..Dec, UTC)")
ax.set_ylabel("Monthly electricity consumption (sum)")
ax.set_title(
    f"Monthly Consumption — Month-of-Year Bars (Years: {', '.join(map(str, years_order))})")
ax.legend(title="Year")
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
# plt.savefig("monthly_consumption_moy_bars.png", dpi=120)  # saving left optional/commented
if __name__ == "__main__":
    try:
        plt.show()
    except Exception:
        pass
