# el_consumption_forecast.py
# Daily electricity consumption forecast for the next 7 days starting tomorrow (Europe/Tallinn).
# - Temperature input: temp_forecast.py (by default) or --temp-module / --temp-csv
# - Segmented regression (workday/offday) + bias by season/month (from bias_analysis.py)
# - Output: DataFrame (print), optional CSV + plot

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Paths / settings
# ---------------------------
LOCAL_TZ = "Europe/Tallinn"
BASE_DIR = Path(__file__).resolve().parent
OUTDIR = BASE_DIR / "output"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Month → season map (use 'autumn' instead of 'fall' to match the rest of the pipeline)
SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring",  4: "spring",  5: "spring",
    6: "summer",  7: "summer",  8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}

# ---------------------------
# Safe CSV save
# ---------------------------


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

# ---------------------------
# Temperature loading (module/CSV) + daily averaging by local calendar
# ---------------------------


def _df_to_local_daily_avg(df: pd.DataFrame, tz: str = LOCAL_TZ) -> pd.Series:
    """Expect a datetime index or a datetime column; return local-calendar daily average of EE_avg (Series)."""
    df = df.copy()

    # 1) Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        dt_col = None
        for cand in ["date_local", "datetime", "time", "date"]:
            if cand in df.columns:
                dt_col = cand
                break
        if dt_col is None:
            raise RuntimeError(
                "Temperature input must have a datetime index or a datetime column "
                "named 'date_local'/'datetime'/'time'/'date'."
            )
        df.index = pd.to_datetime(df[dt_col], errors="coerce")

    # 2) Convert to LOCAL TZ
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    else:
        idx = idx.tz_convert(tz)
    df.index = idx

    # 3) EE_avg column
    if "EE_avg" not in df.columns:
        city_cols = [c for c in ["Tallinn", "Tartu", "Pärnu",
                                 "Narva", "Kuressaare"] if c in df.columns]
        if not city_cols:
            raise RuntimeError(
                "Temperature input is missing 'EE_avg' and city columns (Tallinn, Tartu, Pärnu, Narva, Kuressaare)."
            )
        df["EE_avg"] = df[city_cols].mean(axis=1, skipna=True)

    # 4) Hourly → daily (LOCAL calendar)
    daily = df["EE_avg"].resample("D").mean()
    daily.name = "avg_temp_C"
    return daily


def _load_temp_from_module(temp_module: Optional[str]) -> Optional[pd.Series]:
    """Try to load temperature from a Python module. Return daily-average Series or None."""
    import importlib.util
    import importlib
    import contextlib

    if temp_module is None:
        cand = BASE_DIR / "temp_forecast.py"
        if not cand.exists():
            return None
        temp_module = str(cand)

    # .py file -> spec_from_file_location; otherwise import by module name
    if temp_module.endswith(".py"):
        spec = importlib.util.spec_from_file_location(
            "temp_forecast_mod", temp_module)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        # register before exec (dataclasses etc.)
        sys.modules[spec.name] = mod
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            spec.loader.exec_module(mod)
    else:
        # package/module name
        mod = importlib.import_module(temp_module)

    if hasattr(mod, "get_next7_forecast"):
        df = mod.get_next7_forecast()
        return _df_to_local_daily_avg(df)
    if hasattr(mod, "result"):
        return _df_to_local_daily_avg(mod.result)

    return None


def _load_temp_from_csv(path: str) -> pd.Series:
    df = pd.read_csv(path)
    return _df_to_local_daily_avg(df)


def get_next7_local_avg_temp(
    tz: str = LOCAL_TZ,
    temp_module: Optional[str] = None,
    temp_csv: Optional[str] = None
) -> pd.Series:
    """
    Return a Series of length 7 (next 7 local days), named 'avg_temp_C'.
    Load priority: --temp-module → --temp-csv → temp_forecast.py in the same folder.
    """
    s = None
    if temp_module:
        s = _load_temp_from_module(temp_module)
        if s is None:
            raise FileNotFoundError(
                f"Could not load temperature from module: {temp_module}")
    elif temp_csv:
        s = _load_temp_from_csv(temp_csv)
    else:
        s = _load_temp_from_module(None)
        if s is None:
            raise FileNotFoundError(
                "No temperature input found. Use --temp-module or --temp-csv.")

    # Reindex to exactly tomorrow..+6 (LOCAL)
    today_local = pd.Timestamp.now(tz=tz).normalize()
    days = pd.date_range(today_local + pd.Timedelta(days=1),
                         periods=7, freq="D", tz=tz)
    s = s.reindex(days)
    s.index.name = "date_local"
    s.name = "avg_temp_C"
    return s

# ---------------------------
# Day classification (weekend/holiday/season)
# ---------------------------


def classify_days_local(dates_local: pd.DatetimeIndex) -> pd.DataFrame:
    is_weekend = dates_local.dayofweek >= 5

    # EE public holidays
    try:
        import holidays
        years = list(range(dates_local[0].year, dates_local[-1].year + 1))
        ee = holidays.country_holidays("EE", years=years)
        is_holiday = pd.Index(
            [d.date() in ee for d in dates_local], dtype="bool")
    except Exception:
        print("[info] 'holidays' package not available; setting is_holiday=False")
        is_holiday = pd.Index([False] * len(dates_local), dtype="bool")

    segment = np.where(is_weekend | is_holiday, "offday", "workday")
    season = pd.Index([SEASON_MAP[m]
                      for m in dates_local.month], dtype="object")
    weekday = dates_local.day_name()
    return pd.DataFrame(
        {
            "weekday": weekday,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "segment": segment,
            "season": season,
            "month_num": dates_local.month,
        },
        index=dates_local,
    )

# ---------------------------
# Models + bias
# ---------------------------


def get_models_and_bias(
    months_hist: int = 24,
    mode: str = "season",
    segmented_bias: bool = True
):
    """
    Returns:
      models: {"workday":{"a":..,"b":..}, "offday":{"a":..,"b":..}}
      factors: dict — mode='season'|'month'; if segmented_bias=True, keys like 'workday:winter' / 'offday:7'
    This implementation computes segment models from historical merged daily frames (regression_analysis),
    and bias factors from bias_analysis.get_bias_factors (the modern façade).
    """
    # Local imports to avoid circulars at import time
    try:
        from regression_analysis import load_daily_frames, run_linreg
    except Exception as e:
        raise ImportError(
            "Could not import regression_analysis.py (load_daily_frames/run_linreg).") from e
    try:
        from bias_analysis import get_bias_factors
    except Exception as e:
        raise ImportError(
            "Could not import bias_analysis.py (get_bias_factors).") from e

    merged = load_daily_frames(
        months=months_hist, exclude_today=True, tz=LOCAL_TZ).copy()

    def _fit_or_fallback(df_main: pd.DataFrame, df_fallback: pd.DataFrame) -> Dict[str, float]:
        if df_main is None or df_main.empty:
            df_main = df_fallback
        x = df_main["hour_day_value"].to_numpy(dtype=float)
        y = df_main["sum_el_daily_value"].to_numpy(dtype=float)
        slope, intercept, *_ = run_linreg(x, y)
        return {"a": float(intercept), "b": float(slope)}

    if {"is_weekend", "is_holiday"}.issubset(merged.columns):
        workdays = merged[(~merged["is_weekend"]) &
                          (~merged["is_holiday"])].copy()
        offdays = merged[(merged["is_weekend"]) |
                         (merged["is_holiday"])].copy()
    else:
        workdays = merged.copy()
        offdays = merged.copy()

    # Fit models; if one segment is empty, fall back to the other/all
    model_work = _fit_or_fallback(workdays, merged)
    model_off = _fit_or_fallback(offdays, merged)
    models = {"workday": model_work, "offday": model_off}

    # Bias factors
    factors, _meta, _table = get_bias_factors(
        mode=mode, segmented=segmented_bias, months=months_hist, exclude_today=True, tz=LOCAL_TZ
    )

    return models, factors

# ---------------------------
# Forecast computation
# ---------------------------


def forecast_next7(
    mode: str = "season",
    segmented_bias: bool = True,
    months_hist: int = 24,
    temp_module: Optional[str] = None,
    temp_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute daily consumption forecast for 7 local days (tomorrow..+6).
    mode: 'season' | 'month'  (which bias factors to use)
    segmented_bias: True → apply bias by (segment + season/month); False → by season/month only
    months_hist: history used to fit regressions and bias
    temp_module/temp_csv: temperature source (see above)
    """
    # 1) Temperature (EE_avg) for next 7 days
    temp_s = get_next7_local_avg_temp(
        tz=LOCAL_TZ, temp_module=temp_module, temp_csv=temp_csv)
    if temp_s.isna().any():
        print("[warn] Some temperatures are missing in the input; corresponding forecast rows will be NaN.")

    # 2) Day classification (Tallinn)
    cls = classify_days_local(temp_s.index)
    # Fallback mapping: if any Estonian labels sneak in from upstream, normalize to English
    ET2EN = {"talv": "winter", "kevad": "spring",
             "suvi": "summer", "sügis": "autumn"}
    if "season" in cls.columns:
        cls["season"] = cls["season"].replace(ET2EN)

    # 3) Models + bias
    models, factors = get_models_and_bias(
        months_hist=months_hist, mode=mode, segmented_bias=segmented_bias)
    a_w, b_w = models["workday"]["a"], models["workday"]["b"]
    a_o, b_o = models["offday"]["a"],  models["offday"]["b"]

    seg = cls["segment"].to_numpy()
    T = temp_s.to_numpy(dtype=float)

    # 4) Temperature-only prediction: y = a + b*T
    yhat_base = np.where(seg == "offday", a_o + b_o * T, a_w + b_w * T)

    # 5) Bias key + adjustment
    if mode == "season":
        key_base = cls["season"].astype(str)
    else:  # 'month'
        key_base = cls["month_num"].astype(int)

    if segmented_bias:
        keys = [f"{s}:{k}" for s, k in zip(seg, key_base)]
    else:
        keys = list(key_base)

    bias_vals = np.array([factors.get(k, 1.0) for k in keys], dtype=float)
    yhat_adj = yhat_base * bias_vals

    out = pd.DataFrame({
        "date_local": temp_s.index.strftime("%Y-%m-%d"),
        "weekday": cls["weekday"].to_numpy(),
        "is_weekend": cls["is_weekend"].to_numpy(),
        "is_holiday": cls["is_holiday"].to_numpy(),
        "segment": seg,
        "season": cls["season"].to_numpy(),
        "month_num": cls["month_num"].to_numpy(),
        "EE_avg_temp_C": np.round(T, 2),
        "bias_key": keys,
        "bias_factor": np.round(bias_vals, 6),
        "yhat_base": np.round(yhat_base, 2),
        "yhat_consumption": np.round(yhat_adj, 2),
    })

    return out

# ---------------------------
# Plot (bars = consumption, line = temperature)
# ---------------------------


def plot_dual_axis_bars(out_df: pd.DataFrame) -> None:
    dates = pd.to_datetime(out_df["date_local"])
    x = np.arange(len(dates))

    seg = out_df["segment"].to_numpy()
    bar_colors = ["tab:red" if s == "offday" else "tab:blue" for s in seg]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    cons_vals = out_df["yhat_consumption"].to_numpy(dtype=float)
    ax1.bar(x, cons_vals, color=bar_colors,
            label="Forecast consumption", zorder=2)
    ax1.set_ylabel("Forecast consumption (MWh)")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    temp_vals = out_df["EE_avg_temp_C"].to_numpy(dtype=float)
    ax2.plot(x, temp_vals, marker="o", linestyle="--", linewidth=2,
             label="Forecast temperature (°C)", zorder=4)
    ax2.set_ylabel("Forecast temperature (°C)")
    ax2.set_ylim(bottom=min(0, np.nanmin(temp_vals)))

    ax1.set_xticks(x)
    ax1.set_xticklabels([d.strftime("%Y-%m-%d")
                        for d in dates], rotation=45, ha="right")
    ax1.set_xlabel("Date (Europe/Tallinn)")
    plt.tight_layout()
    plt.show()

# ---------------------------
# CLI
# ---------------------------


def main(
    mode: str = "season",
    segmented_bias: bool = True,
    months_hist: int = 24,
    temp_module: Optional[str] = None,
    temp_csv: Optional[str] = None,
    save_csv: bool = False,
    save_plot: bool = False
):
    out = forecast_next7(
        mode=mode,
        segmented_bias=segmented_bias,
        months_hist=months_hist,
        temp_module=temp_module,
        temp_csv=temp_csv
    )

    # Info
    print("\n=== 7-day consumption forecast (Europe/Tallinn) ===")
    print(
        f"- Bias mode: {mode} | segmented_bias={segmented_bias} | months_hist={months_hist}")
    print(out.to_string(index=False))

    # Save
    if save_csv:
        s, e = _period_strings_next7(tz=LOCAL_TZ)
        path = OUTDIR / f"forecast_consumption_daily_next7_tallinn_{s}_{e}.csv"
        saved = _safe_save_csv(out, path)
        print(f"[saved] {saved}")

    if save_plot:
        plot_dual_axis_bars(out)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Daily electricity consumption forecast for the next 7 days (Europe/Tallinn)")
    p.add_argument("--mode", choices=["season", "month"],
                   default="season", help="Bias factor type to apply")
    p.add_argument("--segmented-bias", dest="segmented_bias", action="store_true",
                   help="Apply bias by segment (workday/offday) as well as season/month")
    p.add_argument("--no-seg-bias", dest="segmented_bias", action="store_false",
                   help="Do not apply segment-specific bias (season/month only)")
    p.add_argument("--months", type=int, default=24,
                   help="History window in months for regression/bias (default: 24)")
    p.add_argument("--temp-module", type=str, default=None,
                   help="Temperature module (e.g., temp_forecast.py or a package name)")
    p.add_argument("--temp-csv", type=str, default=None,
                   help="Temperature CSV (must contain datetime+EE_avg or city columns)")
    p.add_argument("--save-csv", action="store_true",
                   help="Save CSV into output/")
    p.add_argument("--save-plot", action="store_true",
                   help="Show a dual-axis chart")
    p.set_defaults(segmented_bias=True)
    args = p.parse_args()

    main(
        mode=args.mode,
        segmented_bias=args.segmented_bias,
        months_hist=args.months,
        temp_module=args.temp_module,
        temp_csv=args.temp_csv,
        save_csv=args.save_csv,
        save_plot=args.save_plot,
    )
