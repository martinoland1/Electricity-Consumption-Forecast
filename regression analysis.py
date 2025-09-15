# regression_analysis.py
# Purpose: Import el_consumption.py and temp.py, join daily frames by date,
#          run a linear regression (consumption ~ avg day temperature),
#          print metrics, and show a plot (scatter + regression line + equation).
#
# Requirements:
#   pip install pandas numpy matplotlib
#   (optional) pip install scipy   # for p-value via linregress

import os
import sys
import io
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
SUPPRESS_MODULE_OUTPUT = True   # silence prints from the imported scripts
FIG_DPI = 130                   # change if you want sharper plots
SAVE_FIG = False                # set True to also save PNG alongside showing it
FIG_PATH = "temp_vs_consumption_regression.png"

# -----------------------------
# Import helper (silence prints from modules on import)
# -----------------------------


def import_module_silent(name: str):
    if SUPPRESS_MODULE_OUTPUT:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            module = __import__(name)
        return module
    else:
        return __import__(name)


# -----------------------------
# Ensure current directory is on sys.path
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# -----------------------------
# Import project modules
# -----------------------------
# must define sum_daily_el_consumption
ec = import_module_silent("el_consumption")
# must define avg_day_temp OR avg_day_tempo
tp = import_module_silent("temp")

# -----------------------------
# Access dataframes from modules
# -----------------------------
df_cons = getattr(ec, "sum_daily_el_consumption", None)
if df_cons is None:
    raise RuntimeError(
        "sum_daily_el_consumption not found in el_consumption.py")

df_temp = getattr(tp, "avg_day_temp", None)
if df_temp is None:
    df_temp = getattr(tp, "avg_day_tempo", None)
if df_temp is None:
    raise RuntimeError("avg_day_temp (or avg_day_tempo) not found in temp.py")

# -----------------------------
# Validate expected columns
# -----------------------------
expected_cons_cols = {"sum_cons_date", "sum_el_daily_value"}
expected_temp_cols = {"avg_day_temp_date", "hour_day_value"}

missing_cons = expected_cons_cols - set(df_cons.columns)
missing_temp = expected_temp_cols - set(df_temp.columns)
if missing_cons:
    raise RuntimeError(
        f"sum_daily_el_consumption missing columns: {missing_cons}")
if missing_temp:
    raise RuntimeError(f"avg_day_temp/_tempo missing columns: {missing_temp}")

# -----------------------------
# Prepare data (dates & numeric)
# -----------------------------
dfc = df_cons.copy()
dft = df_temp.copy()

# Coerce dates robustly (keep date part only)
dfc["sum_cons_date"] = pd.to_datetime(
    dfc["sum_cons_date"], errors="coerce").dt.date
dft["avg_day_temp_date"] = pd.to_datetime(
    dft["avg_day_temp_date"], errors="coerce").dt.date

# Aggregate defensively in case of duplicates:
dfc = (dfc.groupby("sum_cons_date", as_index=False,
       dropna=False)["sum_el_daily_value"].sum())
dft = (dft.groupby("avg_day_temp_date", as_index=False,
       dropna=False)["hour_day_value"].mean())

# Inner join by date (keep only overlapping days)
merged = pd.merge(
    dfc, dft,
    left_on="sum_cons_date",
    right_on="avg_day_temp_date",
    how="inner"
)

# Keep essentials & enforce numeric
merged = merged[["sum_cons_date",
                 "sum_el_daily_value", "hour_day_value"]].copy()
merged["sum_el_daily_value"] = pd.to_numeric(
    merged["sum_el_daily_value"], errors="coerce")
merged["hour_day_value"] = pd.to_numeric(
    merged["hour_day_value"], errors="coerce")
merged = merged.dropna()

if merged.empty:
    raise RuntimeError("Merged dataset is empty after alignment/cleaning. "
                       "Check upstream scripts, date alignment, and timezones.")

# -----------------------------
# Correlation & regression
# -----------------------------
x = merged["hour_day_value"].to_numpy(dtype=float)          # avg day temp (°C)
y = merged["sum_el_daily_value"].to_numpy(dtype=float)      # daily consumption

# Try SciPy for p-value; fall back to NumPy
try:
    from scipy.stats import linregress
    lr = linregress(x, y)
    slope = float(lr.slope)
    intercept = float(lr.intercept)
    r_value = float(lr.rvalue)
    r2 = r_value ** 2
    p_value = float(lr.pvalue)
except Exception:
    slope, intercept = np.polyfit(x, y, 1)  # no p-value with pure NumPy
    r_value = float(np.corrcoef(x, y)[0, 1])
    r2 = r_value ** 2
    p_value = np.nan

# Predictions & residuals for error metrics
y_hat = intercept + slope * x
res = y - y_hat
rmse = float(np.sqrt(np.mean(res ** 2)))
mae = float(np.mean(np.abs(res)))

# -----------------------------
# Print summary
# -----------------------------
print("\n=== MERGED daily data (first 10 rows) ===")
print(merged.head(10).to_string(index=False))
print("\nMerged shape (rows, cols):", merged.shape)

print("\n=== Linear Regression Summary: consumption = intercept + slope * avg_day_temp ===")
print(f"- Slope (consumption per °C): {slope:.6f}")
print(f"- Intercept (at 0°C):        {intercept:.6f}")
print(f"- R (correlation):           {r_value:.4f}")
print(f"- R²:                        {r2:.4f}")
print(
    f"- p-value:                   {p_value if not np.isnan(p_value) else 'N/A (SciPy not installed)'}")
print(f"- RMSE:                      {rmse:.4f}")
print(f"- MAE:                       {mae:.4f}")
print(f"- Days used:                 {len(merged)}")
print(
    f"- Date range:                {merged['sum_cons_date'].min()} … {merged['sum_cons_date'].max()}")

# -----------------------------
# Plot (scatter + regression line + equation)
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=FIG_DPI)

# Scatter plot
ax.scatter(x, y, alpha=0.6, label="Daily observations")

# Regression line
order = np.argsort(x)
x_line = x[order]
y_line = intercept + slope * x_line
ax.plot(x_line, y_line, linewidth=2, label="Regression line")

# Labels & title
ax.set_title("Daily Consumption vs Average Day Temperature", pad=12)
ax.set_xlabel("Average day temperature (°C)")
ax.set_ylabel("Daily electricity consumption")
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend()

# --- Add equation (mathtext) and R² onto the chart ---
sign = '+' if slope >= 0 else '−'  # visual sign for slope
eq_text = rf"$\hat{{y}} = {intercept:.3f} {sign} {abs(slope):.3f} \times x$"
ann_text = f"{eq_text}\n$R^2$ = {r2:.3f}"

ax.text(
    0.02, 0.98,
    ann_text,
    transform=ax.transAxes,
    va="top", ha="left",
    bbox=dict(boxstyle="round", alpha=0.15, linewidth=0.5)
)

plt.tight_layout()

# Save (optional) and show
if SAVE_FIG:
    plt.savefig(FIG_PATH, dpi=FIG_DPI)
plt.show()
