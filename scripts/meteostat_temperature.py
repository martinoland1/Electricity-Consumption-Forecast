# meteostat_temperature.py
# Reusable temperature client (Meteostat) aligned with Europe/Tallinn calendar days
# Fetch (UTC) -> average across EE points -> convert to local -> hard-cut [start,end) -> return hourly/daily

from __future__ import annotations
import os
import math
import time
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
from meteostat import Hourly, Point
from dateutil.relativedelta import relativedelta

LOCAL_TZ = "Europe/Tallinn"

# ---- Default Estonian points (can be overridden) ----
DEFAULT_POINTS: Dict[str, Point] = {
    "Tallinn":    Point(59.4370, 24.7536),
    "Tartu":      Point(58.3776, 26.7290),
    "Pärnu":      Point(58.3859, 24.4971),
    "Narva":      Point(59.3793, 28.2000),
    "Kuressaare": Point(58.2528, 22.4869),
}

# ---------------------------
# Helpers
# ---------------------------


def _chunk_bounds(start_local: pd.Timestamp, end_local: pd.Timestamp, months_per_chunk: int = 12
                  ) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp]]:
    a = start_local
    while a < end_local:
        b = min(a + relativedelta(months=months_per_chunk), end_local)
        yield a, b
        a = b


def _safe_to_csv(df: pd.DataFrame, path: str) -> str:
    """Windows-friendly save: if locked, append _v2/_v3..."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base, ext = os.path.splitext(path)
    i, target = 1, path
    while True:
        try:
            df.to_csv(target, index=False)
            return target
        except PermissionError:
            i += 1
            target = f"{base}_v{i}{ext}"

# ---------------------------
# Core fetch
# ---------------------------


def _fetch_hourly_utc_for_point(pt: Point, start_utc_naive: pd.Timestamp, end_utc_naive: pd.Timestamp,
                                retries: int = 3, backoff: float = 2.0) -> pd.DataFrame:
    """
    Meteostat Hourly expects NAIVE datetimes + timezone='UTC'.
    Returns index tz-aware UTC, with 'temp' column (°C) if available.
    """
    for attempt in range(retries):
        try:
            df = Hourly(pt, start_utc_naive, end_utc_naive,
                        timezone="UTC").fetch()
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (2 ** attempt))


def _average_points(points: Dict[str, Point], start_local: pd.Timestamp, end_local: pd.Timestamp) -> pd.Series:
    """
    Build a single hourly UTC series averaged across points.
    Input bounds are LOCAL; convert to UTC naive for Meteostat.
    """
    # Use UTC naive bounds (Meteostat requirement)
    start_utc_naive = start_local.tz_convert("UTC").tz_localize(None)
    end_utc_naive = end_local.tz_convert("UTC").tz_localize(None)

    frames = []
    for name, pt in points.items():
        df = _fetch_hourly_utc_for_point(pt, start_utc_naive, end_utc_naive)
        if df.empty or "temp" not in df.columns:
            continue
        frames.append(df[["temp"]].rename(columns={"temp": name}))

    if not frames:
        raise RuntimeError(
            "Meteostat ei tagastanud ühelegi punktile temperatuuri (temp) antud perioodil.")

    # Align to one hourly UTC index and average
    combined = pd.concat(frames, axis=1).sort_index()  # index tz-aware UTC
    return combined.mean(axis=1, skipna=True)


def _hard_cut_local(df: pd.DataFrame, ts_col: str, start_local: pd.Timestamp, end_local: pd.Timestamp) -> pd.DataFrame:
    """Keep start <= ts < end in LOCAL time."""
    if df.empty:
        return df
    m = (df[ts_col] >= start_local) & (df[ts_col] < end_local)
    return df.loc[m].reset_index(drop=True)

# ---------------------------
# Public API
# ---------------------------


def get_hourly_temperature(
    months: int = 24,
    end_exclusive_local: Optional[pd.Timestamp] = None,
    tz: str = LOCAL_TZ,
    exclude_today: bool = True,
    points: Optional[Dict[str, Point]] = None,
) -> pd.DataFrame:
    """
    Return hourly average temperature across selected EE points in local time.
    Columns: hour_temp_time (tz-aware local), hour_temp_value (°C)
    """
    points = points or DEFAULT_POINTS
    now_local = pd.Timestamp.now(tz=tz)
    today_local = now_local.normalize()
    if end_exclusive_local is None or exclude_today:
        end_exclusive_local = today_local  # exclude today entirely
    start_inclusive_local = end_exclusive_local - relativedelta(months=months)

    # Fetch in chunks (to be future-proof for longer periods)
    parts = []
    for a_local, b_local in _chunk_bounds(start_inclusive_local, end_exclusive_local, months_per_chunk=12):
        # hourly index in UTC tz-aware
        avg_series = _average_points(points, a_local, b_local)
        parts.append(avg_series)

    hourly_utc = pd.concat(parts).sort_index()
    # Convert to LOCAL timezone and package dataframe
    hourly_local = hourly_utc.tz_convert(tz)
    df = hourly_local.reset_index()
    df.columns = ["hour_temp_time", "hour_temp_value"]  # required names

    # Strict local cut [start, end) to avoid partial today
    df["hour_temp_time"] = pd.to_datetime(
        df["hour_temp_time"], utc=True).dt.tz_convert(tz)
    df = _hard_cut_local(df, "hour_temp_time",
                         start_inclusive_local, end_exclusive_local)

    return df


def get_daily_temperature(
    months: int = 24,
    end_exclusive_local: Optional[pd.Timestamp] = None,
    tz: str = LOCAL_TZ,
    exclude_today: bool = True,
    points: Optional[Dict[str, Point]] = None,
) -> pd.DataFrame:
    """
    Aggregate hourly temperature to local-calendar daily averages.
    Columns: avg_day_temp_date (date), hour_day_value (°C)
    """
    hourly = get_hourly_temperature(
        months=months,
        end_exclusive_local=end_exclusive_local,
        tz=tz,
        exclude_today=exclude_today,
        points=points,
    )

    if hourly.empty:
        return pd.DataFrame(columns=["avg_day_temp_date", "hour_day_value"])

    idx = hourly.set_index("hour_temp_time")  # tz-aware LOCAL index
    daily = (
        idx["hour_temp_value"]
        .resample("D")  # local calendar days
        .mean()
        .rename("hour_day_value")
        .reset_index()
    )

    # Drop any bucket >= end_exclusive_local (safety)
    cut_off = (pd.Timestamp.now(tz=tz).normalize() if exclude_today else None)
    if exclude_today:
        daily = daily[daily["hour_temp_time"] < cut_off].reset_index(drop=True)

    daily["avg_day_temp_date"] = daily["hour_temp_time"].dt.date
    daily = daily[["avg_day_temp_date", "hour_day_value"]].sort_values(
        "avg_day_temp_date").reset_index(drop=True)
    return daily


# ---------------------------
# Optional CLI dump (CSV)
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Meteostat temperature (Europe/Tallinn) — hourly/daily exports.")
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--include-today", action="store_true",
                        help="Include today (default excluded).")
    parser.add_argument("--daily", action="store_true",
                        help="Export daily (default hourly).")
    parser.add_argument("--outdir", type=str, default="output")
    args = parser.parse_args()

    exclude_today = not args.include_today
    if args.daily:
        df = get_daily_temperature(
            months=args.months, exclude_today=exclude_today)
    else:
        df = get_hourly_temperature(
            months=args.months, exclude_today=exclude_today)

    end_local = (pd.Timestamp.now(tz=LOCAL_TZ).normalize()
                 if exclude_today else pd.Timestamp.now(tz=LOCAL_TZ))
    start_local = end_local - relativedelta(months=args.months)
    s = start_local.strftime("%Y%m%d")
    e = (end_local - pd.Timedelta(seconds=1)).strftime("%Y%m%d")
    kind = "daily" if args.daily else "hourly"
    fname = f"meteostat_temp_{kind}_last{args.months}months_{s}_{e}.csv"
    path = os.path.join(args.outdir, fname)
    saved = _safe_to_csv(df, path)
    print(f"Saved: {saved}")
