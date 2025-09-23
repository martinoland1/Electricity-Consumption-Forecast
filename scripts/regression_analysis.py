# regression_analysis.py
# Purpose:
#   Use elering_consumption.py + meteostat_temperature.py daily frames,
#   run a linear regression (daily consumption ~ avg day temperature),
#   print metrics, and plot (scatter + regression line + equation).
#
# Usage (CLI):
#   python regression_analysis.py --months 24
#   python regression_analysis.py --months 36 --save-fig
#
# Requirements:
#   pip install pandas numpy matplotlib
#   (optional) pip install scipy   # for p-value via linregress

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config (defaults)
# -----------------------------
LOCAL_TZ = "Europe/Tallinn"
FIG_DPI = 130
DEFAULT_MONTHS = 24

# -----------------------------
# Ensure current directory is on sys.path
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# -----------------------------
# Import our project modules (functions-based API)
# -----------------------------
try:
    from elering_consumption import get_daily_consumption
except Exception as e:
    raise RuntimeError(
        "Could not find elering_consumption.py or import failed. "
        "Ensure the file is in the same directory."
    ) from e

try:
    from meteostat_temperature import get_daily_temperature
except Exception as e:
    raise RuntimeError(
        "Could not find meteostat_temperature.py or import failed. "
        "Ensure the file is in the same directory."
    ) from e


# -----------------------------
# Core: fetch & align daily frames
# -----------------------------
def load_daily_frames(months: int = DEFAULT_MONTHS,
                      exclude_today: bool = True,
                      tz: str = LOCAL_TZ) -> pd.DataFrame:
    """
    Returns merged daily frame with columns:
      sum_cons_date (date), sum_el_daily_value (float),
      hour_day_value (avg day temp, °C),
      [is_weekend, is_holiday] if available.
    Only overlapping dates are kept (inner join).
    """
    # Daily consumption (already in local calendar, today excluded by default)
    df_cons = get_daily_consumption(
        months=months,
        tz=tz,
        exclude_today=exclude_today,
        add_weekday=True,
        add_holidays=True,
        impute_missing_hourly=True,
    )

    # Daily temperature (already in local calendar, today excluded by default)
    df_temp = get_daily_temperature(
        months=months,
        tz=tz,
        exclude_today=exclude_today,
    )

    # Sanity checks
    expected_cons = {"sum_cons_date", "sum_el_daily_value"}
    expected_temp = {"avg_day_temp_date", "hour_day_value"}
    if not expected_cons.issubset(df_cons.columns):
        missing = expected_cons - set(df_cons.columns)
        raise RuntimeError(
            f"Consumption frame missing columns: {sorted(missing)}")
    if not expected_temp.issubset(df_temp.columns):
        missing = expected_temp - set(df_temp.columns)
        raise RuntimeError(
            f"Temperature frame missing columns: {sorted(missing)}")

    # Ensure date types
    dfc = df_cons.copy()
    dft = df_temp.copy()
    dfc["sum_cons_date"] = pd.to_datetime(
        dfc["sum_cons_date"], errors="coerce").dt.date
    dft["avg_day_temp_date"] = pd.to_datetime(
        dft["avg_day_temp_date"], errors="coerce").dt.date

    # Defensive aggregation (if duplicates)
    extra_cols = []
    if "is_weekend" in dfc.columns:
        extra_cols.append("is_weekend")
    if "is_holiday" in dfc.columns:
        extra_cols.append("is_holiday")

    if extra_cols:
        dfc = dfc.groupby("sum_cons_date", as_index=False, dropna=False).agg(
            sum_el_daily_value=("sum_el_daily_value", "sum"),
            **{c: (c, "max") for c in extra_cols}
        )
    else:
        dfc = dfc.groupby("sum_cons_date", as_index=False, dropna=False)[
            "sum_el_daily_value"].sum()

    dft = dft.groupby("avg_day_temp_date", as_index=False,
                      dropna=False)["hour_day_value"].mean()

    # Inner join on overlapping days
    merged = pd.merge(
        dfc, dft,
        left_on="sum_cons_date",
        right_on="avg_day_temp_date",
        how="inner"
    )

    # Keep essentials
    keep = ["sum_cons_date", "sum_el_daily_value", "hour_day_value"] + \
        [c for c in extra_cols if c in merged.columns]
    merged = merged[keep].copy()

    # Numeric coercion + drop NA
    merged["sum_el_daily_value"] = pd.to_numeric(
        merged["sum_el_daily_value"], errors="coerce")
    merged["hour_day_value"] = pd.to_numeric(
        merged["hour_day_value"], errors="coerce")
    merged = merged.dropna()

    if merged.empty:
        raise RuntimeError(
            "Merged dataset is empty after alignment/cleaning. "
            "Check upstream frames, date ranges, and time zone settings."
        )

    return merged


# -----------------------------
# Regression utils
# -----------------------------
def run_linreg(x: np.ndarray, y: np.ndarray):
    """
    Returns (slope, intercept, r, r2, p_value or np.nan, rmse, mae).
    """
    # Try SciPy (p-value); fall back to NumPy
    try:
        from scipy.stats import linregress
        lr = linregress(x, y)
        slope = float(lr.slope)
        intercept = float(lr.intercept)
        r = float(lr.rvalue)
        r2 = r * r
        p = float(lr.pvalue)
    except Exception:
        slope, intercept = np.polyfit(x, y, 1)
        r = float(np.corrcoef(x, y)[0, 1])
        r2 = r * r
        p = np.nan

    yhat = intercept + slope * x
    res = y - yhat
    rmse = float(np.sqrt(np.mean(res ** 2)))
    mae = float(np.mean(np.abs(res)))
    return slope, intercept, r, r2, p, rmse, mae


def print_summary(title: str, xname: str, yname: str, metrics, nrows: int, dmin, dmax):
    slope, intercept, r, r2, p, rmse, mae = metrics
    print(f"\n=== {title} ===")
    print(f"- Model: {yname} = intercept + slope * {xname}")
    print(f"- Slope (per °C): {slope:.6f}")
    print(f"- Intercept:      {intercept:.6f}")
    print(f"- R:              {r:.4f}")
    print(f"- R²:             {r2:.4f}")
    print(
        f"- p-value:        {p if not np.isnan(p) else 'N/A (SciPy not installed)'}")
    print(f"- RMSE:           {rmse:.4f}")
    print(f"- MAE:            {mae:.4f}")
    print(f"- Rows used:      {nrows}")
    print(f"- Date range:     {dmin} … {dmax}")


def plot_segment(df: pd.DataFrame, label: str, xcol="hour_day_value", ycol="sum_el_daily_value"):
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    slope, intercept, r, r2, p, rmse, mae = run_linreg(x, y)

    xline = np.linspace(x.min(), x.max(), 100)
    yline = intercept + slope * xline

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, s=12, alpha=0.7)
    ax.plot(xline, yline, linewidth=2)

    try:
        eq_text = rf"$\hat{{y}} = {intercept:.1f} + {slope:.1f}\,x$   $R^2={r2:.3f}$"
        ax.set_title(f"{label}\n{eq_text}")
    except Exception:
        ax.set_title(label)

    ax.set_xlabel("Avg day temperature (°C)")
    ax.set_ylabel("Daily consumption (MWh)")
    ax.grid(True, alpha=0.3)
    return fig


# -----------------------------
# Orchestrator
# -----------------------------
def main(months: int = DEFAULT_MONTHS, exclude_today: bool = True, save_fig: bool = False, outdir: str = "output"):
    merged = load_daily_frames(
        months=months, exclude_today=exclude_today, tz=LOCAL_TZ)

    print("\n=== MERGED daily data (head) ===")
    print(merged.head(10).to_string(index=False))
    print("Merged shape:", merged.shape)

    # ALL DAYS
    all_metrics = run_linreg(
        merged["hour_day_value"].to_numpy(dtype=float),
        merged["sum_el_daily_value"].to_numpy(dtype=float),
    )
    print_summary("ALL DAYS — Linear Regression Summary",
                  "avg day temperature (°C)", "daily consumption (MWh)",
                  all_metrics, len(merged),
                  merged["sum_cons_date"].min(), merged["sum_cons_date"].max())

    figs = []
    # If we have flags, also split into segments
    if set(["is_weekend", "is_holiday"]).issubset(merged.columns):
        workdays = merged[(~merged["is_weekend"]) &
                          (~merged["is_holiday"])].copy()
        offdays = merged[(merged["is_weekend"]) |
                         (merged["is_holiday"])].copy()

        if not workdays.empty:
            m = run_linreg(workdays["hour_day_value"].to_numpy(float),
                           workdays["sum_el_daily_value"].to_numpy(float))
            print_summary("WORKDAYS (Mon–Fri, non-holiday)",
                          "avg day temperature (°C)", "daily consumption (MWh)",
                          m, len(workdays), workdays["sum_cons_date"].min(), workdays["sum_cons_date"].max())
            figs.append(("workdays", plot_segment(
                workdays, "WORKDAYS (Mon–Fri, non-holiday)")))
        if not offdays.empty:
            m = run_linreg(offdays["hour_day_value"].to_numpy(float),
                           offdays["sum_el_daily_value"].to_numpy(float))
            print_summary("WEEKENDS & HOLIDAYS",
                          "avg day temperature (°C)", "daily consumption (MWh)",
                          m, len(offdays), offdays["sum_cons_date"].min(), offdays["sum_cons_date"].max())
            figs.append(("offdays", plot_segment(
                offdays, "WEEKENDS & HOLIDAYS")))
    else:
        figs.append(("all_days", plot_segment(merged, "ALL DAYS")))

    # Show and (optionally) save
    for name, fig in figs:
        if save_fig:
            os.makedirs(outdir, exist_ok=True)
            end_local = pd.Timestamp.now(tz=LOCAL_TZ).normalize(
            ) if exclude_today else pd.Timestamp.now(tz=LOCAL_TZ)
            start_local = end_local - pd.offsets.DateOffset(months=months)
            s = pd.Timestamp(start_local).strftime("%Y%m%d")
            e = pd.Timestamp(
                end_local - pd.Timedelta(seconds=1)).strftime("%Y%m%d")
            path = os.path.join(outdir, f"regression_{name}_{s}_{e}.png")
            fig.savefig(path, dpi=FIG_DPI)
            print(f"[saved] {path}")
    plt.show()


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Daily regression: consumption ~ avg day temperature")
    parser.add_argument("--months", type=int, default=DEFAULT_MONTHS,
                        help="How many months back (default: 24)")
    parser.add_argument("--include-today", action="store_true",
                        help="Include today (default: excluded)")
    parser.add_argument("--save-fig", action="store_true",
                        help="Also save figures into output/")
    parser.add_argument("--outdir", type=str, default="output",
                        help="Where to save figures")
    args = parser.parse_args()

    exclude_today = not args.include_today
    main(months=args.months, exclude_today=exclude_today,
         save_fig=args.save_fig, outdir=args.outdir)
