# bias_analysis.py
# Compute bias correction factors (multipliers) by season or month,
# optionally segmented by workday/offday. Also provides an apply helper.

from __future__ import annotations
import os
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/Tallinn"
# keep consistent across the pipeline
SEASON_ORDER_EN = ["winter", "spring", "summer", "autumn"]

# ---------------------------------------------------------------------
# Dependencies within the project
# ---------------------------------------------------------------------
# - load_daily_frames: merged daily consumption + temperature (local calendar, today excluded by default)
# - run_linreg: returns slope, intercept, r, r2, p, rmse, mae
try:
    from regression_analysis import load_daily_frames, run_linreg
except Exception as e:
    raise ImportError(
        "Could not import from regression_analysis.py. Ensure it is in the same directory."
    ) from e


# ---------------------------------------------------------------------
# Meta for factor computation
# ---------------------------------------------------------------------
@dataclass
class BiasMeta:
    mode: str                 # 'season' | 'month'
    segmented: bool
    months: int
    exclude_today: bool
    tz: str
    period_start: str         # YYYYMMDD (inclusive)
    # YYYYMMDD (exclusive if exclude_today=True; otherwise now)
    period_end: str


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _month_to_season(month: int) -> str:
    # DJF / MAM / JJA / SON mapping; label 'autumn' (not 'fall') to match the rest of the pipeline
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _season_categorical(series: pd.Series) -> pd.Series:
    return pd.Categorical(series, categories=SEASON_ORDER_EN, ordered=True)


def _safe_mean(x: pd.Series) -> float:
    return float(pd.to_numeric(x, errors="coerce").dropna().mean())


def _date_str(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d")


# ---------------------------------------------------------------------
# Core: compute factors
# ---------------------------------------------------------------------
def _build_training_frame(months: int, exclude_today: bool, tz: str) -> pd.DataFrame:
    """
    Pull merged daily frame and add grouping columns:
      - month_num (1..12)
      - season (winter, spring, summer, autumn)
      - segment (workday/offday) if flags present
    Also compute a global linear regression prediction yhat over ALL rows.
    """
    merged = load_daily_frames(
        months=months, exclude_today=exclude_today, tz=tz).copy()

    # Ensure required columns
    need = {"sum_cons_date", "sum_el_daily_value", "hour_day_value"}
    missing = need - set(merged.columns)
    if missing:
        raise RuntimeError(
            f"Merged frame missing column(s): {sorted(missing)}")

    # Basic types
    merged["sum_el_daily_value"] = pd.to_numeric(
        merged["sum_el_daily_value"], errors="coerce")
    merged["hour_day_value"] = pd.to_numeric(
        merged["hour_day_value"], errors="coerce")
    merged["sum_cons_date"] = pd.to_datetime(
        merged["sum_cons_date"], errors="coerce").dt.date

    # Month / season
    merged["month_num"] = pd.to_datetime(merged["sum_cons_date"]).dt.month
    merged["season"] = merged["month_num"].map(_month_to_season)
    merged["season"] = _season_categorical(merged["season"])

    # Segment (if flags available)
    if {"is_weekend", "is_holiday"}.issubset(merged.columns):
        merged["segment"] = np.where(
            (~merged["is_weekend"]) & (
                ~merged["is_holiday"]), "workday", "offday"
        )
    else:
        # will be ignored unless segmented=True
        merged["segment"] = "workday_offday_unknown"

    # Global regression: y ~ x  (y=consumption, x=avg temp)
    x = merged["hour_day_value"].to_numpy(dtype=float)
    y = merged["sum_el_daily_value"].to_numpy(dtype=float)
    slope, intercept, *_ = run_linreg(x, y)

    merged["yhat_reg"] = intercept + slope * \
        merged["hour_day_value"].astype(float)
    # guard: avoid zero/negatives when forming ratios
    merged = merged[(merged["yhat_reg"].abs() > 1e-9) &
                    merged["sum_el_daily_value"].notna()].copy()
    merged["ratio"] = merged["sum_el_daily_value"] / merged["yhat_reg"]

    return merged


def _aggregate_factors(df: pd.DataFrame, mode: str, segmented: bool) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Build factor map and a support table. Factor = mean(actual / yhat_reg) within the group.
    Keys:
      - season, e.g., 'winter'
      - month, e.g., '7'
    If segmented=True, keys are 'workday:winter' or 'offday:7', etc.
    """
    if mode not in {"season", "month"}:
        raise ValueError("mode must be 'season' or 'month'.")

    group_cols: List[str] = []
    if segmented:
        # ignore synthetic placeholder 'workday_offday_unknown'
        df = df[df["segment"].isin(["workday", "offday"])].copy()
        group_cols.append("segment")

    if mode == "season":
        # enforce categorical order for clean output
        df["season"] = _season_categorical(df["season"])
        group_cols.append("season")
        order_by = ["segment", "season"] if segmented else ["season"]
    else:
        group_cols.append("month_num")
        order_by = ["segment", "month_num"] if segmented else ["month_num"]

    # aggregate
    g = (
        df.groupby(group_cols, dropna=False)
          .agg(
              avg_bias_factor=("ratio", "mean"),
              std_ratio=("ratio", "std"),
              p25=("ratio", lambda s: float(np.nanpercentile(
                  s.dropna().to_numpy(), 25)) if s.notna().any() else np.nan),
              p50=("ratio", lambda s: float(np.nanpercentile(
                  s.dropna().to_numpy(), 50)) if s.notna().any() else np.nan),
              p75=("ratio", lambda s: float(np.nanpercentile(
                  s.dropna().to_numpy(), 75)) if s.notna().any() else np.nan),
              count=("ratio", "count"),
        )
        .reset_index()
    )

    # stable ordering
    if mode == "season":
        g["season"] = _season_categorical(g["season"])
        g = g.sort_values(order_by).reset_index(drop=True)
    else:
        g["month_num"] = pd.to_numeric(g["month_num"], errors="coerce")
        g = g.sort_values(order_by).reset_index(drop=True)

    # build key->factor map
    factors: Dict[str, float] = {}
    for _, row in g.iterrows():
        if mode == "season":
            k = f"{row['season']}"
        else:
            k = f"{int(row['month_num'])}" if pd.notna(
                row["month_num"]) else ""

        if segmented:
            k = f"{row['segment']}:{k}"
        factors[k] = float(row["avg_bias_factor"])

    return factors, g


def get_bias_factors(
    mode: str = "season",
    segmented: bool = True,
    months: int = 24,
    exclude_today: bool = True,
    tz: str = LOCAL_TZ,
):
    """
    Compute bias correction factors on the last `months` of data (today excluded by default).
    Returns:
      factors: Dict[str, float]  e.g., {'workday:winter': 1.03, 'offday:7': 0.98, ...}
      meta:    BiasMeta dataclass
      table:   pd.DataFrame with columns:
               - if season mode: ['season', ('segment'), 'avg_bias_factor', 'std_ratio', 'p25','p50','p75','count']
               - if month  mode: ['month_num', ('segment'), 'avg_bias_factor', 'std_ratio', 'p25','p50','p75','count']
    """
    if mode not in {"season", "month"}:
        raise ValueError("mode must be 'season' or 'month'.")

    df = _build_training_frame(
        months=months, exclude_today=exclude_today, tz=tz)

    # meta period (local)
    end_local = pd.Timestamp.now(tz=tz).normalize(
    ) if exclude_today else pd.Timestamp.now(tz=tz)
    start_local = end_local - pd.offsets.DateOffset(months=months)
    meta = BiasMeta(
        mode=mode,
        segmented=segmented,
        months=months,
        exclude_today=exclude_today,
        tz=tz,
        period_start=_date_str(start_local),
        period_end=_date_str(end_local),
    )

    factors, table = _aggregate_factors(df, mode=mode, segmented=segmented)
    return factors, meta, table


# ---------------------------------------------------------------------
# Apply factors to a forecast
# ---------------------------------------------------------------------
def _build_bias_key(row: pd.Series, mode: str, segmented: bool) -> str:
    if mode == "season":
        season = _month_to_season(int(row["month_num"]))
        return f"{row['segment']}:{season}" if segmented else season
    else:
        m = int(row["month_num"])
        return f"{row['segment']}:{m}" if segmented else f"{m}"


def apply_bias_to_forecast(
    df: pd.DataFrame,
    predicted_col: str = "yhat_consumption",
    date_col: str = "date_local",
    factors: Optional[Dict[str, float]] = None,
    mode: str = "season",
    segmented: bool = True,
    out_col: str = "yhat_consumption_bias_adj",
) -> pd.DataFrame:
    """
    Multiplies `predicted_col` by a bias factor based on date's month/season and (optionally) segment.
    - Expects `date_col` in local calendar date (string or datetime-like).
    - If segmented=True, expects a 'segment' column with values {'workday','offday'}.
    - Adds columns: ['month_num', 'bias_key', out_col].
    """
    if mode not in {"season", "month"}:
        raise ValueError("mode must be 'season' or 'month'.")
    if factors is None:
        factors = {}

    out = df.copy()

    # Coerce date and predicted numeric
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if out[date_col].isna().any():
        n = int(out[date_col].isna().sum())
        raise ValueError(f"{date_col} contains {n} unparsable value(s).")

    out[predicted_col] = pd.to_numeric(out[predicted_col], errors="coerce")

    # Month (for both season/month modes)
    out["month_num"] = out[date_col].dt.month

    # Segment handling
    if segmented:
        if "segment" not in out.columns:
            raise KeyError(
                "Segmented bias requested but 'segment' column is missing.")
        # sanitize segment values
        out["segment"] = out["segment"].astype(str).str.lower().where(
            out["segment"].isin(["workday", "offday"]), other="offday"
        )
    else:
        out["segment"] = "n/a"

    # Compose bias keys
    out["bias_key"] = out.apply(lambda r: _build_bias_key(
        r, mode=mode, segmented=segmented), axis=1)

    # Map factors with sensible fallbacks
    def _lookup_factor(r) -> float:
        k = r["bias_key"]
        if k in factors:
            return float(factors[k])

        # Fallback to unsegmented key if segmented requested
        if segmented:
            if mode == "season":
                base = k.split(":", 1)[-1]  # 'winter'
            else:
                base = k.split(":", 1)[-1]  # '7'
            if base in factors:
                return float(factors[base])

        return 1.0  # default neutral factor

    out["_bias_factor"] = out.apply(_lookup_factor, axis=1)

    # Apply
    out[out_col] = out[predicted_col] * out["_bias_factor"]

    return out


# ---------------------------------------------------------------------
# Optional CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute bias correction factors by season or month.")
    parser.add_argument("--mode", choices=["season", "month"], default="season",
                        help="Group by 'season' or 'month'")
    parser.add_argument("--segmented", action="store_true",
                        help="Compute separate factors for workday/offday")
    parser.add_argument("--months", type=int, default=24,
                        help="History window in months (default: 24)")
    parser.add_argument("--include-today", action="store_true",
                        help="Include today in training (default: excluded)")
    parser.add_argument("--outdir", type=str, default="output",
                        help="Directory to save artifacts (CSV/JSON)")
    args = parser.parse_args()

    exclude_today = not args.include_today
    factors, meta, table = get_bias_factors(
        mode=args.mode,
        segmented=args.segmented,
        months=args.months,
        exclude_today=exclude_today,
        tz=LOCAL_TZ,
    )

    # Save artifacts
    os.makedirs(args.outdir, exist_ok=True)
    start, end = meta.period_start, meta.period_end
    seg_tag = "_seg" if args.segmented else ""

    # CSV: factor table (stable ordering: seasons or months)
    csv_path = os.path.join(
        args.outdir, f"bias_{args.mode}{seg_tag}_TABLE_{start}_{end}.csv")
    table.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    # JSON: factor map
    try:
        import json
        json_path = os.path.join(
            args.outdir, f"bias_{args.mode}{seg_tag}_FACTORS_{start}_{end}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(factors, f, ensure_ascii=False, indent=2)
        print(f"[saved] {json_path}")
    except Exception as e:
        print(f"[warn] Failed to save JSON factors: {e}")

    # META print
    print("=== META ===")
    try:
        from pprint import pprint
        pprint(asdict(meta))
    except Exception:
        print(meta)
