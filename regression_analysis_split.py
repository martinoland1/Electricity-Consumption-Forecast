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
ec = import_module_silent("el_consumption_weekday")
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
if all(c in dfc.columns for c in ["is_weekend", "is_holiday"]):
    dfc = dfc.groupby("sum_cons_date", as_index=False, dropna=False).agg(
        sum_el_daily_value=("sum_el_daily_value", "sum"),
        is_weekend=("is_weekend", "max"),
        is_holiday=("is_holiday", "max"),
    )
else:
    dfc = (dfc.groupby("sum_cons_date", as_index=False, dropna=False)
           ["sum_el_daily_value"].sum())
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
extra_cols = [c for c in ["is_weekend", "is_holiday"] if c in merged.columns]
merged = merged[["sum_cons_date", "sum_el_daily_value",
                 "hour_day_value"] + extra_cols].copy()
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
# Segment-specific regressions (minimal additions)
# -----------------------------
if set(["is_weekend", "is_holiday"]).issubset(merged.columns):
    def _run_simple_linreg(df_part, label):
        if df_part.empty:
            print(f"\n[{label}] No rows in this segment; skipping regression.")
            return None
        _x = df_part["hour_day_value"].to_numpy(dtype=float)
        _y = df_part["sum_el_daily_value"].to_numpy(dtype=float)
        try:
            from scipy.stats import linregress
            _lr = linregress(_x, _y)
            _slope = float(_lr.slope)
            _intercept = float(_lr.intercept)
            _r = float(_lr.rvalue)
            _r2 = _r ** 2
            _p = float(_lr.pvalue)
        except Exception:
            _slope, _intercept = np.polyfit(_x, _y, 1)
            _r = float(np.corrcoef(_x, _y)[0, 1])
            _r2 = _r ** 2
            _p = np.nan
        _yhat = _intercept + _slope * _x
        _res = _y - _yhat
        _rmse = float(np.sqrt(np.mean(_res ** 2)))
        _mae = float(np.mean(np.abs(_res)))
        print(f"\n=== {label} — Linear Regression Summary ===")
        print(f"- Slope (per °C): {_slope:.6f}")
        print(f"- Intercept:      {_intercept:.6f}")
        print(f"- R:              {_r:.4f}")
        print(f"- R²:             {_r2:.4f}")
        print(
            f"- p-value:        {_p if not np.isnan(_p) else 'N/A (SciPy not installed)'}")
        print(f"- RMSE:           {_rmse:.4f}")
        print(f"- MAE:            {_mae:.4f}")
        print(f"- Days used:      {len(df_part)}")
        print(
            f"- Date range:     {df_part['sum_cons_date'].min()} … {df_part['sum_cons_date'].max()}")
        return (_slope, _intercept, _r2)

    workdays = merged[(~merged["is_weekend"]) & (~merged["is_holiday"])].copy()
    offdays = merged[(merged["is_weekend"]) | (merged["is_holiday"])].copy()

    _run_simple_linreg(workdays, "WORKDAYS (Mon–Fri, non-holiday)")
    _run_simple_linreg(offdays, "WEEKENDS & HOLIDAYS")
else:
    print("\n[info] is_weekend/is_holiday not present — segment-specific regressions skipped.")

# -----------------------------
# Plots (2 figures): WORKDAYS and WEEKENDS & HOLIDAYS (robust titles)
# -----------------------------


def _set_title_with_eq(ax, label, intercept, slope, r2):
    try:
        eq_text = rf"$\hat{{y}} = {intercept:.1f} + {slope:.1f}\,x$   $R^2={r2:.3f}$"
        ax.set_title(f"{label}\n{eq_text}")
    except Exception as e:
        ax.set_title(f"{label}\nEquation unavailable ({e.__class__.__name__})")


if set(["is_weekend", "is_holiday"]).issubset(merged.columns):
    segments = [
        (merged[(~merged["is_weekend"]) & (~merged["is_holiday"])],
         "WORKDAYS (Mon–Fri, non-holiday)"),
        (merged[(merged["is_weekend"]) | (merged["is_holiday"])],
         "WEEKENDS & HOLIDAYS"),
    ]
else:
    print("[info] Flags missing — falling back to a single plot for ALL DAYS.")
    segments = [(merged, "ALL DAYS")]

for df_part, label in segments:
    if df_part.empty:
        print(f"[plot] {label}: no data, skipping.")
        continue

    x = df_part["hour_day_value"].to_numpy(dtype=float)
    y = df_part["sum_el_daily_value"].to_numpy(dtype=float)

    # Fit line
    try:
        from scipy.stats import linregress
        lr = linregress(x, y)
        slope, intercept, r, p = float(lr.slope), float(
            lr.intercept), float(lr.rvalue), float(lr.pvalue)
        r2 = r * r
    except Exception:
        slope, intercept = np.polyfit(x, y, 1)
        r = float(np.corrcoef(x, y)[0, 1])
        r2 = r * r
        p = float("nan")

    # Line for plotting
    xline = np.linspace(x.min(), x.max(), 100)
    yline = intercept + slope * xline

    # Figure
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, s=12, alpha=0.7)
    ax.plot(xline, yline, linewidth=2)

    _set_title_with_eq(ax, label, intercept, slope, r2)
    ax.set_xlabel("Avg day temperature (°C)")
    ax.set_ylabel("Daily consumption")
    ax.grid(True, alpha=0.3)

plt.show()

# Save (optional) and show
if SAVE_FIG:
    plt.savefig(FIG_PATH, dpi=FIG_DPI)
plt.show()
