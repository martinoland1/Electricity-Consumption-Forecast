# temp_forecast.py
# Next 7 days: daily-average temperatures by Europe/Tallinn calendar days.
# Meteostat Hourly requires naive UTC datetimes (tzinfo=None) and model=True to obtain forecast values.

from __future__ import annotations
import os
from typing import Dict, Iterable, Tuple, Optional

import pandas as pd
from meteostat import Point, Hourly
from dateutil.relativedelta import relativedelta

LOCAL_TZ = "Europe/Tallinn"

# Default Estonian points (you can override via get_next7_forecast(points=...))
DEFAULT_POINTS: Dict[str, Point] = {
    "Tallinn":    Point(59.4370, 24.7536),
    "Tartu":      Point(58.3776, 26.7290),
    "Pärnu":      Point(58.3859, 24.4971),
    "Narva":      Point(59.3793, 28.2000),
    "Kuressaare": Point(58.2528, 22.4869),
}

_RESULT_CACHE: Optional[pd.DataFrame] = None


def _local_next7_bounds(tz: str = LOCAL_TZ) -> Tuple[pd.Timestamp, pd.Timestamp, pd.DatetimeIndex]:
    """Return the [start, end) local window for the next 7 complete days (starting tomorrow 00:00)."""
    today = pd.Timestamp.now(tz=tz).normalize()
    start_local = today + pd.Timedelta(days=1)         # tomorrow 00:00 local
    end_local_excl = start_local + pd.Timedelta(days=7)  # [start, end)
    days_local = pd.date_range(start_local, periods=7, freq="D", tz=tz)
    return start_local, end_local_excl, days_local


def _to_utc_naive(t_local: pd.Timestamp) -> pd.Timestamp:
    """tz-aware local -> tz-aware UTC -> naive UTC (required by Meteostat)."""
    return t_local.tz_convert("UTC").tz_localize(None)


def _fetch_hourly_utc(pt: Point, start_utc_naive: pd.Timestamp, end_utc_naive: pd.Timestamp) -> pd.DataFrame:
    """Fetch hourly series from Meteostat (UTC, model=True). Return DataFrame with a UTC-aware index."""
    df = Hourly(pt, start_utc_naive, end_utc_naive, model=True).fetch()
    if df is None or df.empty:
        return pd.DataFrame()
    if "time" in df.columns:
        df = df.set_index("time")
    # Ensure UTC-aware index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def get_next7_forecast(points: Dict[str, Point] = DEFAULT_POINTS, tz: str = LOCAL_TZ) -> pd.DataFrame:
    """
    Return a DataFrame indexed by local calendar dates (Europe/Tallinn), length = 7 rows,
    columns: individual city averages + 'EE_avg' (°C, float).
    """
    global _RESULT_CACHE
    if _RESULT_CACHE is not None and points is DEFAULT_POINTS and tz == LOCAL_TZ:
        return _RESULT_CACHE.copy()

    start_local, end_local_excl, days_local = _local_next7_bounds(tz=tz)
    start_utc_naive = _to_utc_naive(start_local)
    end_utc_naive = _to_utc_naive(end_local_excl)

    frames = []
    for name, pt in points.items():
        dfh = _fetch_hourly_utc(pt, start_utc_naive, end_utc_naive)
        if dfh.empty or "temp" not in dfh.columns:
            continue
        # UTC -> LOCAL, resample by local calendar day, reindex to ensure exactly 7 rows
        s_local = (
            dfh["temp"]
            .tz_convert(tz)
            .resample("D")            # LOCAL day
            .mean()
            .reindex(days_local)
            .rename(name)
        )
        frames.append(s_local)

    city_cols = list(points.keys())
    if not frames:
        # Build an empty frame with expected columns to avoid KeyError when taking the mean
        df = pd.DataFrame(index=days_local, columns=city_cols, dtype=float)
    else:
        df = pd.concat(frames, axis=1)

    # Country average across available cities
    df["EE_avg"] = df[city_cols].mean(axis=1, skipna=True)
    df.index.name = "date_local"

    _RESULT_CACHE = df.copy()
    return df


# Compatibility: also export 'result' so other scripts can `from temp_forecast import result`
try:
    result = get_next7_forecast()
except Exception as e:
    result = pd.DataFrame()
    print(f"[warn] get_next7_forecast() failed: {e}")


# -------------- CLI --------------
def _safe_to_csv(df: pd.DataFrame, path: str) -> str:
    """Windows-friendly save: if the file is locked, append _v2/_v3..."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base, ext = os.path.splitext(path)
    target = path
    i = 2
    while True:
        try:
            df.to_csv(target, index=True)
            return target
        except PermissionError:
            target = f"{base}_v{i}{ext}"
            i += 1


def _period_strings_next7(tz: str = LOCAL_TZ) -> Tuple[str, str]:
    """Return YYYYMMDD strings for the next-7-day window (start=tomorrow, end=+6)."""
    today = pd.Timestamp.now(tz=tz).normalize()
    start = today + pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=6)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Meteostat forecast: daily average temperature for the next 7 days (Europe/Tallinn)")
    parser.add_argument("--save-csv", action="store_true",
                        help="Save CSV into the output/ folder")
    args = parser.parse_args()

    df = get_next7_forecast()
    out = df.round(1).reset_index()
    out["date_local"] = out["date_local"].dt.strftime("%Y-%m-%d")
    print("\nNext 7 days — daily average temperatures (°C) — Europe/Tallinn:")
    print(out[["date_local"] + list(DEFAULT_POINTS.keys()) +
          ["EE_avg"]].to_string(index=False))

    if args.save_csv:
        s, e = _period_strings_next7()
        path = _safe_to_csv(out, os.path.join(
            "output", f"temp_forecast_daily_next7_tallinn_{s}_{e}.csv"))
        print(f"[saved] {path}")
