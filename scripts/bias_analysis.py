# bias_analysis.py
# Goal
# - Use regression_analysis.load_daily_frames() to get daily (consumption + avg day temp) in Europe/Tallinn
# - Fit temp-only linear regression: y_hat = a + b * T
# - Compute monthly bias (actual/predicted), aggregate to month-of-year & season factors
# - Provide a reusable API + CLI for saving CSV/JSON and plots
#
# CLI examples:
#   python bias_analysis.py --months 24 --save-csv --save-plot
#   python bias_analysis.py --months 24 --segmented --save-csv --save-plot
#
# Requires:
#   pandas, numpy, matplotlib
#   regression_analysis.py in the same folder (with load_daily_frames, run_linreg)

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

# --------------------------------
# Settings
# --------------------------------
LOCAL_TZ = "Europe/Tallinn"
BASE_DIR = Path(__file__).resolve().parent
OUTDIR = BASE_DIR / "output"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Ensure this folder on path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import project utilities
try:
    from regression_analysis import load_daily_frames, run_linreg
except Exception as e:
    raise RuntimeError(
        "Could not import regression_analysis.py. Make sure the file is in the same folder and compiles."
    ) from e

# Season mapping (English)
SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring",  4: "spring",  5: "spring",
    6: "summer",  7: "summer",  8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}

# --------------------------------
# IO helpers (Windows-friendly)
# --------------------------------


def _safe_save_csv(df: pd.DataFrame, path: Path) -> Path:
    """Save CSV. If locked/exists, append _v2/_v3... until success."""
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


def _safe_save_json(obj, path: Path) -> Path:
    """Save JSON with UTF-8. If locked, append _v2/_v3..."""
    path.parent.mkdir(parents=True, exist_ok=True)
    base, ext = path.with_suffix(""), path.suffix
    cand = Path(f"{base}{ext}")
    i = 2
    while True:
        try:
            with open(cand, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            return cand
        except PermissionError:
            cand = Path(f"{base}_v{i}{ext}")
            i += 1


def _period_strings(months: int, exclude_today: bool, tz: str = LOCAL_TZ) -> Tuple[str, str]:
    """
    Return (start_YYYYMMDD, end_YYYYMMDD) for filenames.
    If exclude_today=True, end is yesterday 23:59:59 local.
    """
    now_local = pd.Timestamp.now(tz=tz)
    end_local_excl = now_local.normalize() if exclude_today else now_local
    start_local_incl = end_local_excl - relativedelta(months=months)
    s = start_local_incl.strftime("%Y%m%d")
    e = (end_local_excl - pd.Timedelta(seconds=1)).strftime("%Y%m%d")
    return s, e

# --------------------------------
# Core data loading
# --------------------------------


def load_merged_for_bias(months: int = 24, exclude_today: bool = True, tz: str = LOCAL_TZ) -> pd.DataFrame:
    """
    Returns daily merged dataset from regression_analysis.load_daily_frames():
      sum_cons_date (date), sum_el_daily_value (float),
      hour_day_value (avg day temperature °C),
      and optionally is_weekend/is_holiday flags.
    """
    merged = load_daily_frames(
        months=months, exclude_today=exclude_today, tz=tz)

    # Ensure required columns exist
    need = {"sum_cons_date", "sum_el_daily_value", "hour_day_value"}
    missing = need - set(merged.columns)
    if missing:
        raise RuntimeError(f"Merged dataset is missing columns: {missing}")

    # Normalize types
    merged = merged.copy()
    merged["sum_cons_date"] = pd.to_datetime(
        merged["sum_cons_date"], errors="coerce").dt.date
    merged["sum_el_daily_value"] = pd.to_numeric(
        merged["sum_el_daily_value"], errors="coerce")
    merged["hour_day_value"] = pd.to_numeric(
        merged["hour_day_value"], errors="coerce")
    merged = merged.dropna(
        subset=["sum_cons_date", "sum_el_daily_value", "hour_day_value"])

    if merged.empty:
        raise RuntimeError("Merged dataset is empty after cleaning.")
    return merged

# --------------------------------
# Segment annotation
# --------------------------------


def annotate_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'segment' = workday vs offday (offday = weekend OR public holiday)."""
    out = df.copy()
    w = out["is_weekend"] if "is_weekend" in out.columns else pd.Series(
        False, index=out.index)
    h = out["is_holiday"] if "is_holiday" in out.columns else pd.Series(
        False, index=out.index)
    out["segment"] = np.where((w.astype(bool)) | (
        h.astype(bool)), "offday", "workday")
    return out

# --------------------------------
# Modeling & predictions
# --------------------------------


def compute_daily_predictions(df: pd.DataFrame) -> Tuple[pd.DataFrame, float, float, Dict[str, float]]:
    """
    Fit linear model y = a + b*T (using regression_analysis.run_linreg).
    Returns (df_with_yhat_resid, intercept a, slope b, metrics_dict)
    metrics: {r, r2, p_value, rmse, mae}
    """
    x = df["hour_day_value"].to_numpy(dtype=float)
    y = df["sum_el_daily_value"].to_numpy(dtype=float)
    slope, intercept, r, r2, p, rmse, mae = run_linreg(x, y)

    out = df.copy()
    out["y_hat"] = intercept + slope * out["hour_day_value"]
    out["resid"] = out["sum_el_daily_value"] - out["y_hat"]

    metrics = {
        "r": float(r), "r2": float(r2),
        "p_value": (None if (p is None or (isinstance(p, float) and np.isnan(p))) else float(p)),
        "rmse": float(rmse), "mae": float(mae),
    }
    return out, float(intercept), float(slope), metrics


def compute_daily_predictions_segmented(df: pd.DataFrame):
    """
    Segment-specific regressions: separate model for 'workday' and 'offday'.
    Returns (df_pred, models, metrics):
      - df_pred: per-segment y_hat and resid
      - models:  {segment: {"a": intercept, "b": slope}}
      - metrics: {segment: {"r","r2","p_value","rmse","mae"}}
    """
    if "segment" not in df.columns:
        df = annotate_segment(df)
    models, metrics = {}, {}
    parts = []
    for seg, g in df.groupby("segment", dropna=False):
        x = g["hour_day_value"].to_numpy(dtype=float)
        y = g["sum_el_daily_value"].to_numpy(dtype=float)
        slope, intercept, r, r2, p, rmse, mae = run_linreg(x, y)
        gg = g.copy()
        gg["y_hat"] = intercept + slope * gg["hour_day_value"]
        gg["resid"] = gg["sum_el_daily_value"] - gg["y_hat"]
        parts.append(gg)
        models[seg] = {"a": float(intercept), "b": float(slope)}
        metrics[seg] = {
            "r": float(r), "r2": float(r2),
            "p_value": (None if (p is None or (isinstance(p, float) and np.isnan(p))) else float(p)),
            "rmse": float(rmse), "mae": float(mae)
        }
    df_pred = pd.concat(parts, ignore_index=True).sort_values("sum_cons_date")
    return df_pred, models, metrics

# --------------------------------
# Aggregations: monthly & season (unsegmented)
# --------------------------------


def aggregate_monthly_bias(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily predictions to months:
      month (Timestamp, month start), actual, predicted, abs_error, pct_error, bias_factor,
      month_num (1..12), season (winter/spring/summer/autumn).
    """
    m = (
        df_pred.assign(_dt=pd.to_datetime(
            df_pred["sum_cons_date"], errors="coerce"))
        .assign(month=lambda d: d["_dt"].dt.to_period("M").dt.to_timestamp())
    )
    monthly = (
        m.groupby("month", as_index=False, dropna=False)
         .agg(actual=("sum_el_daily_value", "sum"),
              predicted=("y_hat", "sum"))
    )
    monthly["abs_error"] = monthly["actual"] - monthly["predicted"]
    monthly["pct_error"] = np.where(monthly["actual"] != 0,
                                    monthly["abs_error"] / monthly["actual"],
                                    np.nan)
    monthly["bias_factor"] = np.where(monthly["predicted"] > 0,
                                      monthly["actual"] / monthly["predicted"],
                                      np.nan)
    monthly["month_num"] = monthly["month"].dt.month
    monthly["season"] = monthly["month_num"].map(SEASON_MAP)
    return monthly


def bias_by_month_of_year(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Month-of-year averages over multiple years.
    Returns: month_num, avg_bias_factor, avg_pct_error, months, season
    """
    df = monthly.copy()
    if "month_num" not in df.columns:
        df["month_num"] = pd.to_datetime(df["month"]).dt.month

    by_moy = (
        df.groupby("month_num", as_index=False, dropna=False)
          .agg(avg_bias_factor=("bias_factor", "mean"),
               avg_pct_error=("pct_error", "mean"),
               months=("bias_factor", "count"))
    )
    by_moy["season"] = by_moy["month_num"].map(SEASON_MAP)
    return by_moy


def bias_by_season(by_moy: pd.DataFrame) -> pd.DataFrame:
    """
    Average bias per season (weighted by months count).
    Returns: season, avg_bias_factor, months
    """
    season_df = (
        by_moy.groupby("season", as_index=False, dropna=False)
        .agg(avg_bias_factor=("avg_bias_factor", "mean"),
             months=("months", "sum"))
    )
    return season_df

# --------------------------------
# Aggregations: segmented
# --------------------------------


def aggregate_monthly_bias_segmented(df_pred: pd.DataFrame) -> pd.DataFrame:
    if "segment" not in df_pred.columns:
        df_pred = annotate_segment(df_pred)
    m = (df_pred.assign(_dt=pd.to_datetime(df_pred["sum_cons_date"], errors="coerce"))
         .assign(month=lambda d: d["_dt"].dt.to_period("M").dt.to_timestamp()))
    monthly = (m.groupby(["month", "segment"], as_index=False, dropna=False)
               .agg(actual=("sum_el_daily_value", "sum"),
                    predicted=("y_hat", "sum")))
    monthly["abs_error"] = monthly["actual"] - monthly["predicted"]
    monthly["pct_error"] = np.where(monthly["actual"] != 0,
                                    monthly["abs_error"] / monthly["actual"], np.nan)
    monthly["bias_factor"] = np.where(monthly["predicted"] > 0,
                                      monthly["actual"] / monthly["predicted"], np.nan)
    monthly["month_num"] = monthly["month"].dt.month
    monthly["season"] = monthly["month_num"].map(SEASON_MAP)
    return monthly


def bias_by_month_of_year_segmented(monthly_seg: pd.DataFrame) -> pd.DataFrame:
    df = monthly_seg.copy()
    if "month_num" not in df.columns:
        df["month_num"] = pd.to_datetime(df["month"]).dt.month
    by_moy_seg = (df.groupby(["segment", "month_num"], as_index=False, dropna=False)
                    .agg(avg_bias_factor=("bias_factor", "mean"),
                         avg_pct_error=("pct_error", "mean"),
                         months=("bias_factor", "count")))
    by_moy_seg["season"] = by_moy_seg["month_num"].map(SEASON_MAP)
    return by_moy_seg


def bias_by_season_segmented(by_moy_seg: pd.DataFrame) -> pd.DataFrame:
    return (by_moy_seg.groupby(["segment", "season"], as_index=False, dropna=False)
                      .agg(avg_bias_factor=("avg_bias_factor", "mean"),
                           months=("months", "sum")))

# --------------------------------
# Apply corrections & accuracy
# --------------------------------


def apply_corrections(monthly: pd.DataFrame, by_moy: pd.DataFrame, season_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add corrected predictions to monthly table:
      pred_month_corr = predicted * month_bias(month_num)
      pred_season_corr = predicted * season_bias(season)
    """
    m = monthly.copy()
    month_bias_map: Dict[int, float] = by_moy.set_index(
        "month_num")["avg_bias_factor"].to_dict()
    season_bias_map: Dict[str, float] = season_df.set_index(
        "season")["avg_bias_factor"].to_dict()
    m["pred_month_corr"] = m.apply(
        lambda r: r["predicted"] * month_bias_map.get(int(r["month_num"]), 1.0), axis=1)
    m["pred_season_corr"] = m.apply(
        lambda r: r["predicted"] * season_bias_map.get(str(r["season"]), 1.0), axis=1)
    return m


def apply_corrections_segmented(monthly_seg: pd.DataFrame, by_moy_seg: pd.DataFrame, season_seg: pd.DataFrame) -> pd.DataFrame:
    m = monthly_seg.copy()
    month_bias_map = by_moy_seg.set_index(["segment", "month_num"])[
        "avg_bias_factor"].to_dict()
    season_bias_map = season_seg.set_index(["segment", "season"])[
        "avg_bias_factor"].to_dict()

    def _month_corr(r): return r["predicted"] * \
        month_bias_map.get((r["segment"], int(r["month_num"])), 1.0)
    def _season_corr(r): return r["predicted"] * \
        season_bias_map.get((r["segment"], str(r["season"])), 1.0)

    m["pred_month_corr"] = m.apply(_month_corr, axis=1)
    m["pred_season_corr"] = m.apply(_season_corr, axis=1)
    return m


def _mape(actual: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> float:
    a = np.array(actual, dtype=float)
    p = np.array(pred, dtype=float)
    mask = a != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])))


def summarize_accuracy(monthly_corr: pd.DataFrame) -> Dict[str, float]:
    """
    Compute accuracy for base vs corrected variants (MAPE), plus overall bias (actual/predicted).
    """
    base = _mape(monthly_corr["actual"], monthly_corr["predicted"])
    month = _mape(monthly_corr["actual"], monthly_corr["pred_month_corr"])
    season = _mape(monthly_corr["actual"], monthly_corr["pred_season_corr"])
    total_actual = float(monthly_corr["actual"].sum())
    total_pred = float(monthly_corr["predicted"].sum())
    overall_bias_factor = (
        total_actual / total_pred) if total_pred > 0 else np.nan
    return {"mape_base": base, "mape_month": month, "mape_season": season, "overall_bias_factor": overall_bias_factor}


def summarize_accuracy_segmented(monthly_corr_seg: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for seg, g in monthly_corr_seg.groupby("segment", dropna=False):
        out[seg] = {
            "mape_base":   _mape(g["actual"], g["predicted"]),
            "mape_month":  _mape(g["actual"], g["pred_month_corr"]),
            "mape_season": _mape(g["actual"], g["pred_season_corr"]),
            "overall_bias_factor": (float(g["actual"].sum()) / float(g["predicted"].sum())) if g["predicted"].sum() > 0 else np.nan,
        }
    return out

# --------------------------------
# Public API (simple)
# --------------------------------


def get_season_bias(months: int = 24, exclude_today: bool = True) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Return (season_bias_df, season_bias_map)."""
    merged = load_merged_for_bias(
        months=months, exclude_today=exclude_today, tz=LOCAL_TZ)
    df_pred, *_ = compute_daily_predictions(merged)
    monthly = aggregate_monthly_bias(df_pred)
    by_moy = bias_by_month_of_year(monthly)
    season_df = bias_by_season(by_moy)
    season_map = season_df.set_index("season")["avg_bias_factor"].to_dict()
    return season_df.copy(), dict(season_map)


def get_month_bias(months: int = 24, exclude_today: bool = True) -> Tuple[pd.DataFrame, Dict[int, float]]:
    """Return (month_of_year_bias_df, month_bias_map) with entries for 1..12."""
    merged = load_merged_for_bias(
        months=months, exclude_today=exclude_today, tz=LOCAL_TZ)
    df_pred, *_ = compute_daily_predictions(merged)
    monthly = aggregate_monthly_bias(df_pred)
    by_moy = bias_by_month_of_year(monthly)
    month_map = by_moy.set_index("month_num")["avg_bias_factor"].to_dict()
    return by_moy.copy(), {int(k): float(v) for k, v in month_map.items()}


def get_season_bias_segmented(months: int = 24, exclude_today: bool = True):
    """Return (season_bias_df_segmented, season_bias_map_segmented, models)."""
    merged = annotate_segment(load_merged_for_bias(
        months=months, exclude_today=exclude_today, tz=LOCAL_TZ))
    df_pred, models, _metrics = compute_daily_predictions_segmented(merged)
    monthly_seg = aggregate_monthly_bias_segmented(df_pred)
    by_moy_seg = bias_by_month_of_year_segmented(monthly_seg)
    season_seg = bias_by_season_segmented(by_moy_seg)
    season_map = season_seg.set_index(["segment", "season"])[
        "avg_bias_factor"].to_dict()
    return season_seg.copy(), dict(season_map), models

# --------------------------------
# Facade API for other scripts (unified)
# --------------------------------


@dataclass
class BiasMeta:
    mode: str                 # "season" | "month"
    segmented: bool           # True -> separate workday/offday
    months: int
    exclude_today: bool
    period_start: str         # YYYYMMDD
    period_end: str           # YYYYMMDD
    schema_version: str = "1.0"


def get_bias_factors(mode: str = "season", segmented: bool = False, months: int = 24, exclude_today: bool = True):
    """
    Unified entry point. Returns (factors, meta, table):

      - factors: dict
          mode="season", segmented=False: {"winter":1.03, "spring":0.95, ...}
          mode="month",  segmented=False: {1:1.01, 2:0.99, ... 12:1.02}
          mode="season", segmented=True:  {"workday:winter":1.02, "offday:winter":1.07, ...}
          mode="month",  segmented=True:  {"workday:1":1.01, "offday:1":1.04, ...}

      - meta: BiasMeta
      - table: underlying DataFrame (corresponding aggregation)
    """
    s, e = _period_strings(months, exclude_today, tz=LOCAL_TZ)

    if not segmented:
        if mode == "season":
            season_df, season_map = get_season_bias(
                months=months, exclude_today=exclude_today)
            factors = {k: float(v) for k, v in season_map.items()}
            meta = BiasMeta(mode="season", segmented=False, months=months, exclude_today=exclude_today,
                            period_start=s, period_end=e)
            return factors, meta, season_df
        elif mode == "month":
            by_moy_df, month_map = get_month_bias(
                months=months, exclude_today=exclude_today)
            factors = {int(k): float(v) for k, v in month_map.items()}
            meta = BiasMeta(mode="month", segmented=False, months=months, exclude_today=exclude_today,
                            period_start=s, period_end=e)
            return factors, meta, by_moy_df
        else:
            raise ValueError("mode must be 'season' or 'month'")

    else:
        # segmented
        season_seg_df, season_map, _models = get_season_bias_segmented(
            months=months, exclude_today=exclude_today)
        if mode == "season":
            factors = {f"{seg}:{sea}": float(val) for (
                seg, sea), val in season_map.items()}
            meta = BiasMeta(mode="season", segmented=True, months=months, exclude_today=exclude_today,
                            period_start=s, period_end=e)
            return factors, meta, season_seg_df
        elif mode == "month":
            by_moy_seg = bias_by_month_of_year_segmented(
                aggregate_monthly_bias_segmented(
                    compute_daily_predictions_segmented(
                        annotate_segment(load_merged_for_bias(
                            months=months, exclude_today=exclude_today, tz=LOCAL_TZ))
                    )[0]
                )
            )
            factors = {f"{row['segment']}:{int(row['month_num'])}": float(row["avg_bias_factor"])
                       for _, row in by_moy_seg.iterrows()}
            meta = BiasMeta(mode="month", segmented=True, months=months, exclude_today=exclude_today,
                            period_start=s, period_end=e)
            return factors, meta, by_moy_seg
        else:
            raise ValueError("mode must be 'season' or 'month'")


def apply_bias_to_forecast(df: pd.DataFrame,
                           predicted_col: str,
                           date_col: str,
                           factors: dict,
                           mode: str = "season",
                           segmented: bool = False,
                           segment_col: str = "segment",
                           season_col: str = "season",
                           month_col: str = "month_num",
                           out_col: str = "pred_bias_adj") -> pd.DataFrame:
    """
    Apply bias factors to a temp-only forecast.

    - mode='season': requires column 'season' (winter/spring/summer/autumn) or derives it from date
    - mode='month' : requires column 'month_num' (1..12) or derives it from date
    - segmented=True: requires 'segment' (workday/offday); key format e.g. 'workday:winter'
    """
    out = df.copy()

    if mode == "month":
        if month_col not in out.columns:
            out[month_col] = pd.to_datetime(out[date_col]).dt.month
    elif mode == "season":
        if season_col not in out.columns:
            month_num = pd.to_datetime(out[date_col]).dt.month
            out[season_col] = month_num.map(SEASON_MAP)

    def _key(row):
        if mode == "season" and segmented:
            return f"{row.get(segment_col, 'workday')}:{row[season_col]}"
        if mode == "month" and segmented:
            return f"{row.get(segment_col, 'workday')}:{int(row[month_col])}"
        return row[season_col] if mode == "season" else int(row[month_col])

    out[out_col] = out[predicted_col] * \
        out.apply(lambda r: factors.get(_key(r), 1.0), axis=1)
    return out

# --------------------------------
# Plotting (optional)
# --------------------------------


def _plot_monthly_comparison(df: pd.DataFrame, title: str, outdir: Path, fname: str) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5), dpi=130)
    ax.plot(df["month"], df["actual"], marker="o", label="Actual")
    ax.plot(df["month"], df["predicted"],
            marker="s", label="Model (uncorrected)")
    ax.plot(df["month"], df["pred_month_corr"],
            marker="^", label="Month-bias adj.")
    ax.plot(df["month"], df["pred_season_corr"],
            marker="D", label="Season-bias adj.")
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Consumption (MWh)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_path = outdir / fname
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path

# --------------------------------
# Orchestration for CLI
# --------------------------------


def main(months: int = 24, exclude_today: bool = True, save_csv: bool = False, save_plot: bool = False,
         segmented: bool = False, outdir: Path = OUTDIR):
    s, e = _period_strings(months, exclude_today, tz=LOCAL_TZ)

    if not segmented:
        merged = load_merged_for_bias(
            months=months, exclude_today=exclude_today, tz=LOCAL_TZ)
        df_pred, a, b, metrics = compute_daily_predictions(merged)
        monthly = aggregate_monthly_bias(df_pred)
        by_moy = bias_by_month_of_year(monthly)
        season_df = bias_by_season(by_moy)
        monthly_corr = apply_corrections(monthly, by_moy, season_df)
        acc = summarize_accuracy(monthly_corr)

        print(
            f"\nSINGLE model: y = {a:.3f} + {b:.3f} * T  | R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}")
        print(
            f"MAPE base: {acc['mape_base']*100:.2f}% | month-adj: {acc['mape_month']*100:.2f}% | season-adj: {acc['mape_season']*100:.2f}%")
        print("\nSeason factors:")
        print(season_df.to_string(index=False))

        if save_csv:
            outdir.mkdir(parents=True, exist_ok=True)
            _safe_save_csv(monthly_corr, outdir /
                           f"bias_monthly_with_corrections_{s}_{e}.csv")
            _safe_save_csv(by_moy,      outdir /
                           f"bias_month_of_year_{s}_{e}.csv")
            _safe_save_csv(season_df,   outdir / f"bias_season_{s}_{e}.csv")
            _safe_save_json(by_moy.set_index("month_num")["avg_bias_factor"].to_dict(),
                            outdir / f"bias_month_map_{s}_{e}.json")
            _safe_save_json(season_df.set_index("season")["avg_bias_factor"].to_dict(),
                            outdir / f"bias_season_map_{s}_{e}.json")
        if save_plot:
            _plot_monthly_comparison(monthly_corr, "Actual vs predictions (single model)", outdir,
                                     f"bias_actual_vs_predictions_{s}_{e}.png")

    else:
        merged = annotate_segment(load_merged_for_bias(
            months=months, exclude_today=exclude_today, tz=LOCAL_TZ))
        df_pred, models, seg_metrics = compute_daily_predictions_segmented(
            merged)
        monthly_seg = aggregate_monthly_bias_segmented(df_pred)
        by_moy_seg = bias_by_month_of_year_segmented(monthly_seg)
        season_seg = bias_by_season_segmented(by_moy_seg)
        monthly_corr_seg = apply_corrections_segmented(
            monthly_seg, by_moy_seg, season_seg)
        acc_seg = summarize_accuracy_segmented(monthly_corr_seg)

        print("\nSEGMENTED models:")
        for seg, m in models.items():
            met = seg_metrics[seg]
            print(
                f"  [{seg}] y = {m['a']:.3f} + {m['b']:.3f} * T  | R²={met['r2']:.4f}, RMSE={met['rmse']:.3f}, MAE={met['mae']:.3f}")
        print("\nSegment accuracies (MAPE, overall bias):")
        for seg, a in acc_seg.items():
            print(f"  [{seg}] MAPE base: {a['mape_base']*100:.2f}% | month-adj: {a['mape_month']*100:.2f}% | season-adj: {a['mape_season']*100:.2f}% | overall bias: {a['overall_bias_factor']:.4f}")

        if save_csv:
            outdir.mkdir(parents=True, exist_ok=True)
            _safe_save_csv(monthly_corr_seg, outdir /
                           f"bias_monthly_with_corrections_segmented_{s}_{e}.csv")
            _safe_save_csv(by_moy_seg,      outdir /
                           f"bias_month_of_year_segmented_{s}_{e}.csv")
            _safe_save_csv(season_seg,      outdir /
                           f"bias_season_segmented_{s}_{e}.csv")
            # JSON maps: {segment: {month_num: factor}}, {(segment,season): factor}
            month_map = {}
            for seg, g in by_moy_seg.groupby("segment"):
                month_map[seg] = g.set_index("month_num")[
                    "avg_bias_factor"].to_dict()
            _safe_save_json(month_map, outdir /
                            f"bias_month_map_segmented_{s}_{e}.json")
            season_map = season_seg.set_index(["segment", "season"])[
                "avg_bias_factor"].to_dict()
            _safe_save_json({f"{k[0]}:{k[1]}": v for k, v in season_map.items()},
                            outdir / f"bias_season_map_segmented_{s}_{e}.json")

        if save_plot:
            for seg, df in monthly_corr_seg.groupby("segment"):
                _plot_monthly_comparison(df, f"Actual vs predictions ({seg})", outdir,
                                         f"bias_actual_vs_predictions_{seg}_{s}_{e}.png")


# --------------------------------
# CLI
# --------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Bias analysis (single model or segmented workday vs offday)"
    )
    parser.add_argument("--months", type=int, default=24,
                        help="How many months of history (default 24).")
    parser.add_argument("--include-today", action="store_true",
                        help="Include today's data (by default it is excluded).")
    parser.add_argument("--save-csv", action="store_true",
                        help="Save CSV/JSON outputs to output/ directory.")
    parser.add_argument("--save-plot", action="store_true",
                        help="Save comparison plot(s) to output/ directory.")
    parser.add_argument("--segmented", action="store_true",
                        help="Use segmented regression and bias (workday vs offday).")
    parser.add_argument("--outdir", type=str, default=str(OUTDIR),
                        help="Output directory (default: output/).")
    args = parser.parse_args()

    exclude_today = not args.include_today
    main(months=args.months,
         exclude_today=exclude_today,
         save_csv=args.save_csv,
         save_plot=args.save_plot,
         segmented=args.segmented,
         outdir=Path(args.outdir))
