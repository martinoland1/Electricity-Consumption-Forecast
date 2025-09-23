# electricity_hourly_forecast.py
# Hourly forecast for the next 7 days (Europe/Tallinn)
# - Daily forecast via el_consumption_forecast.forecast_next7(...) (or --daily-csv)
# - Split into hours via weekday_profile.split_daily_forecast_to_hourly(...)
# - DST- and public-holiday-aware; hourly totals reconcile back to daily totals

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/Tallinn"
BASE_DIR = Path(__file__).resolve().parent
OUTDIR = BASE_DIR / "output"
OUTDIR.mkdir(parents=True, exist_ok=True)


# --------------------- utils ---------------------
def _safe_save_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    base, ext = path.with_suffix(""), path.suffix
    cand = Path(f"{base}{ext}")
    i = 2
    while True:
        try:
            df.to_csv(cand, index=False)
            return cand
        except PermissionError:
            cand = Path(f"{base}_v{i}{ext}")
            i += 1


def _period_strings_next7(tz: str = LOCAL_TZ) -> Tuple[str, str]:
    now = pd.Timestamp.now(tz=tz).normalize()
    start = now + pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=6)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


# --------------------- module import helper ---------------------
def _import_module_from_file(fname: str, modname: str):
    """
    Load a .py module from the current folder; register under sys.modules before executing it.
    """
    import importlib.util
    import contextlib
    fpath = BASE_DIR / fname
    if not fpath.exists():
        raise FileNotFoundError(f"{fname} not found in the same folder.")
    spec = importlib.util.spec_from_file_location(modname, str(fpath))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import: {fname}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        spec.loader.exec_module(mod)
    return mod


# --------------------- daily forecast loaders ---------------------
def load_daily_forecast_from_module(
    mode: str = "season",
    segmented_bias: bool = True,
    months_hist: int = 24,
    temp_module: Optional[str] = None,
    temp_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Call el_consumption_forecast.forecast_next7(...) and return a DataFrame.
    Requires el_consumption_forecast.py in the same folder.
    """
    ecf = _import_module_from_file(
        "el_consumption_forecast.py", "el_consumption_forecast")
    if not hasattr(ecf, "forecast_next7"):
        raise RuntimeError(
            "el_consumption_forecast.forecast_next7 is missing.")
    df = ecf.forecast_next7(
        mode=mode,
        segmented_bias=segmented_bias,
        months_hist=months_hist,
        temp_module=temp_module,
        temp_csv=temp_csv,
    )
    # minimum columns: date_local (YYYY-MM-DD), yhat_consumption (float)
    need = {"date_local", "yhat_consumption"}
    if not need.issubset(df.columns):
        raise RuntimeError(
            "Daily forecast lacks required columns: date_local, yhat_consumption")
    return df


def load_daily_forecast_from_csv(path: str) -> pd.DataFrame:
    """
    Load a daily forecast CSV. If date_local/yhat_consumption are missing,
    try to infer from common alternatives.
    """
    df = pd.read_csv(path)
    # date
    if "date_local" not in df.columns:
        for cand in ["date", "Date", "day", "datetime", "dt"]:
            if cand in df.columns:
                s = pd.to_datetime(df[cand], errors="coerce")
                if s.dt.tz is None:
                    s = s.dt.tz_localize(LOCAL_TZ)
                else:
                    s = s.dt.tz_convert(LOCAL_TZ)
                df["date_local"] = s.dt.strftime("%Y-%m-%d")
                break
        if "date_local" not in df.columns:
            raise RuntimeError(
                "CSV is missing 'date_local' and no date column could be inferred.")
    # value
    if "yhat_consumption" not in df.columns:
        for cand in ["yhat", "forecast", "consumption", "daily_yhat", "yhat_day"]:
            if cand in df.columns:
                df["yhat_consumption"] = pd.to_numeric(
                    df[cand], errors="coerce")
                break
    if "yhat_consumption" not in df.columns:
        raise RuntimeError(
            "CSV is missing the daily forecast column 'yhat_consumption'.")
    return df


# --------------------- split into hours ---------------------
def split_daily_to_hourly(
    daily_df: pd.DataFrame,
    last_n: int = 6,
    holiday_profile: str = "weekday",
    hourly_csv: Optional[str] = None,
    csv_tz: str = LOCAL_TZ,
    months_for_profile: int = 24,
) -> pd.DataFrame:
    """
    Call weekday_profile.split_daily_forecast_to_hourly(...) (EE local time, DST-aware).
    """
    wp = _import_module_from_file("weekday_profile.py", "weekday_profile")
    if not hasattr(wp, "split_daily_forecast_to_hourly"):
        raise RuntimeError(
            "weekday_profile.split_daily_forecast_to_hourly is missing.")
    hourly = wp.split_daily_forecast_to_hourly(
        daily_df,
        date_col="date_local",
        value_col="yhat_consumption",
        # let the module compute the matrix from the last 'last_n' occurrences
        share_matrix=None,
        last_n=last_n,
        exclude_today=True,           # exclude today from profile training
        holiday_profile=holiday_profile,
        hourly_csv=hourly_csv,        # if you prefer building profiles from CSV
        csv_tz=csv_tz,
        months=months_for_profile,    # history for API-based profile building
    )
    # convenience duplicate
    hourly["date_local"] = hourly["datetime_local"].dt.strftime("%Y-%m-%d")
    return hourly


# --------------------- reconciliation check ---------------------
def check_daily_hourly_match(hourly_df: pd.DataFrame, daily_df: pd.DataFrame) -> float:
    """
    Check whether hourly sums reconcile to the daily forecast.
    Returns max absolute relative error (>= 0).
    """
    day_sum = hourly_df.groupby("date_local")[
        "consumption_hourly"].sum().rename("sum_hourly")
    merged = pd.merge(
        day_sum.reset_index(),
        daily_df[["date_local", "yhat_consumption"]],
        on="date_local",
        how="inner",
    )
    if merged.empty:
        return float("nan")
    rel = (merged["sum_hourly"] / merged["yhat_consumption"]) - 1.0
    return float(np.nanmax(np.abs(rel.values)))


# --------------------- CLI workflow ---------------------
def main(
    # daily forecast
    use_daily_csv: Optional[str] = None,
    mode: str = "season",
    segmented_bias: bool = True,
    months_hist: int = 24,
    temp_module: Optional[str] = None,
    temp_csv: Optional[str] = None,
    # day profile / split
    last_n: int = 6,
    holiday_profile: str = "weekday",      # 'weekday'|'sunday'|'weekend_avg'
    # build profiles from a specific hourly CSV instead of the API
    hourly_csv: Optional[str] = None,
    # interpret tz-naive timestamps in hourly_csv as this zone (e.g., 'UTC')
    csv_tz: str = LOCAL_TZ,
    months_for_profile: int = 24,
    # output
    save_csv: bool = False,
):
    # 1) get daily forecast
    if use_daily_csv:
        daily = load_daily_forecast_from_csv(use_daily_csv)
    else:
        daily = load_daily_forecast_from_module(
            mode=mode,
            segmented_bias=segmented_bias,
            months_hist=months_hist,
            temp_module=temp_module,
            temp_csv=temp_csv,
        )

    # 2) split into hours (EE time)
    hourly = split_daily_to_hourly(
        daily_df=daily,
        last_n=last_n,
        holiday_profile=holiday_profile,
        hourly_csv=hourly_csv,
        csv_tz=csv_tz,
        months_for_profile=months_for_profile,
    )

    # 3) attach daily meta — avoid duplicate names (weekday already in hourly table)
    meta_keep = [c for c in [
        "segment", "season", "is_weekend", "is_holiday",
        "month_num", "EE_avg_temp_C", "bias_key", "bias_factor",
        "yhat_base", "yhat_consumption"
    ] if c in daily.columns]
    daily_meta = daily[["date_local"] +
                       meta_keep].drop_duplicates("date_local")

    # NB: suffixes=("", "_daily") keeps hourly column names unchanged
    out = hourly.merge(daily_meta, on="date_local",
                       how="left", suffixes=("", "_daily"))

    # 4) reconciliation check — hourly vs daily
    max_rel = check_daily_hourly_match(out, daily)
    if np.isfinite(max_rel):
        print(
            f"[check] Hourly → daily reconciliation — max |rel_diff| ≈ {max_rel*100:.5f}%")

    # 5) output
    out = out.sort_values(["datetime_local"]).reset_index(drop=True)

    print("\n=== Hourly forecast (first 48 rows) ===")
    cols = ["datetime_local", "weekday", "hour_local", "consumption_hourly"]
    cols += [c for c in ["segment", "season", "EE_avg_temp_C",
                         "yhat_consumption", "bias_factor"] if c in out.columns]
    print(out[cols].head(48).to_string(index=False))

    if save_csv:
        s, e = _period_strings_next7(tz=LOCAL_TZ)
        path = OUTDIR / \
            f"forecast_consumption_hourly_next7_tallinn_{s}_{e}.csv"
        saved = _safe_save_csv(out, path)
        print(f"[saved] {saved}")

    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Hourly electricity consumption forecast for the next 7 days (Europe/Tallinn)")
    # Daily forecast source
    p.add_argument("--daily-csv", type=str, default=None,
                   help="Use an existing daily forecast CSV (date_local,yhat_consumption)")
    p.add_argument("--mode", choices=["season", "month"], default="season",
                   help="Bias type for the daily forecast (if not using --daily-csv)")
    p.add_argument("--segmented-bias", dest="segmented_bias", action="store_true",
                   help="Apply bias by segment (workday/offday)")
    p.add_argument("--no-seg-bias", dest="segmented_bias", action="store_false",
                   help="Do not apply segment-specific bias")
    p.add_argument("--months", type=int, default=24,
                   help="History window (months) for daily models/bias")
    p.add_argument("--temp-module", type=str, default=None,
                   help="Temperature module (e.g., temp_forecast.py)")
    p.add_argument("--temp-csv", type=str, default=None,
                   help="Temperature CSV (must contain datetime + EE_avg or city columns)")
    # Day profile
    p.add_argument("--last-n", type=int, default=6,
                   help="How many most-recent occurrences per weekday to average")
    p.add_argument("--holiday-profile", choices=["weekday", "sunday", "weekend_avg"], default="weekday",
                   help="How to handle public holidays when splitting")
    p.add_argument("--hourly-csv", type=str, default=None,
                   help="Build profiles from this hourly CSV instead of the API")
    p.add_argument("--csv-tz", type=str, default=LOCAL_TZ,
                   help="If --hourly-csv timestamps are TZ-naive, interpret them in this zone (e.g., 'UTC')")
    p.add_argument("--months-for-profile", type=int, default=24,
                   help="How much history to scan for profiles (API mode)")
    # Output
    p.add_argument("--save-csv", action="store_true",
                   help="Save CSV into output/")

    p.set_defaults(segmented_bias=True)
    args = p.parse_args()

    main(
        use_daily_csv=args.daily_csv,
        mode=args.mode,
        segmented_bias=args.segmented_bias,
        months_hist=args.months,
        temp_module=args.temp_module,
        temp_csv=args.temp_csv,
        last_n=args.last_n,
        holiday_profile=args.holiday_profile,
        hourly_csv=args.hourly_csv,
        csv_tz=args.csv_tz,
        months_for_profile=args.months_for_profile,
        save_csv=args.save_csv,
    )
