# elering_consumption.py
# Reusable client for Elering consumption data in Europe/Tallinn.
# API (UTC) -> parse -> convert to local -> hard-cut [start, end) -> (optional) 15min->hour aggregation -> optional impute -> enrich -> (optional) save

from __future__ import annotations
import os
import time
import math
import requests
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple, Iterable, Literal
from dateutil.relativedelta import relativedelta

ELERING_URL = "https://dashboard.elering.ee/api/system/with-plan"
LOCAL_TZ = "Europe/Tallinn"

# ---------------------------
# Internal helpers
# ---------------------------


def _smart_to_datetime(ts):
    """ISO 8601 string or UNIX epoch (s/ms) -> pandas UTC datetime (tz-aware)."""
    if isinstance(ts, str):
        return pd.to_datetime(ts, utc=True, errors="coerce")
    if isinstance(ts, (int, float)) and not math.isnan(ts):
        unit = "ms" if ts > 10_000_000_000 else "s"
        return pd.to_datetime(ts, unit=unit, utc=True, errors="coerce")
    return pd.NaT


def _extract_timestamp_and_consumption(obj) -> list[tuple]:
    """Recursively collect (timestamp, consumption) tuples from JSON."""
    rows = []

    def visit(node):
        if isinstance(node, dict):
            if "timestamp" in node and "consumption" in node:
                rows.append((node["timestamp"], node["consumption"]))
            for v in node.values():
                visit(v)
        elif isinstance(node, list):
            for item in node:
                visit(item)
    visit(obj)
    return rows


def _fetch_chunk(start_utc: datetime, end_utc: datetime, retries: int = 3, backoff: float = 2.0) -> pd.DataFrame:
    """Fetch [start, end) from Elering API."""
    params = {
        "start": start_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "end":   end_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    }
    for attempt in range(retries):
        try:
            r = requests.get(ELERING_URL, params=params, timeout=60)
            r.raise_for_status()
            pairs = _extract_timestamp_and_consumption(r.json())
            df = pd.DataFrame(
                pairs, columns=["sum_cons_time", "sum_el_hourly_value"])
            if not df.empty:
                df["sum_cons_time"] = df["sum_cons_time"].apply(
                    _smart_to_datetime)
                df["sum_el_hourly_value"] = pd.to_numeric(
                    df["sum_el_hourly_value"], errors="coerce")
                df = df.dropna(subset=["sum_cons_time"]).reset_index(drop=True)
            return df
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (2 ** attempt))


def _chunk_bounds(start_local: pd.Timestamp, end_local: pd.Timestamp, months_per_chunk: int = 12) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield local-time chunks [a,b) covering [start_local, end_local)."""
    a = start_local
    while a < end_local:
        b = min(a + relativedelta(months=months_per_chunk), end_local)
        yield a, b
        a = b


def _impute_neighbors_mean(df: pd.DataFrame, value_col: str = "sum_el_hourly_value") -> pd.DataFrame:
    """Fill NaN by mean(prev_valid, next_valid)."""
    if df.empty:
        df["imputed"] = False
        return df
    df = df.sort_values("sum_cons_time").reset_index(drop=True)
    prev_valid = df[value_col].ffill()
    next_valid = df[value_col].bfill()
    to_impute = df[value_col].isna() & prev_valid.notna() & next_valid.notna()
    df.loc[to_impute, value_col] = (
        prev_valid[to_impute] + next_valid[to_impute]) / 2.0
    df["imputed"] = False
    df.loc[to_impute, "imputed"] = True
    return df


def _add_weekday_weekend(df: pd.DataFrame) -> None:
    if df.empty:
        df["weekday"] = pd.Series(dtype="object")
        df["is_weekend"] = pd.Series(dtype="boolean")
        return
    ser = df["sum_cons_time"]
    df["weekday"] = ser.dt.day_name()
    df["is_weekend"] = df["weekday"].isin(["Saturday", "Sunday"])


def _add_holidays_localdate(df: pd.DataFrame, date_col: str) -> None:
    try:
        import holidays  # pip install holidays
        if df.empty:
            df["is_holiday"] = pd.Series(dtype="boolean")
            return
        dmin = df[date_col].min()
        dmax = df[date_col].max()
        if pd.isna(dmin) or pd.isna(dmax):
            df["is_holiday"] = False
            return
        years = range(dmin.year, dmax.year + 1)
        ee = holidays.country_holidays("EE", years=years)
        df["is_holiday"] = df[date_col].map(lambda d: d in ee)
    except Exception:
        df["is_holiday"] = False


def _hard_cut_local(df: pd.DataFrame, start_local: pd.Timestamp, end_local: pd.Timestamp) -> pd.DataFrame:
    """Keep rows where start_local <= ts < end_local (LOCAL tz-aware)."""
    if df.empty:
        return df
    m = (df["sum_cons_time"] >= start_local) & (
        df["sum_cons_time"] < end_local)
    return df.loc[m].reset_index(drop=True)


def _safe_save_csv(df: pd.DataFrame, path: str) -> str:
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


def _aggregate_to_hour(
    df_local: pd.DataFrame,
    strategy: Literal["sum_to_hour", "top_of_hour_only"] = "sum_to_hour",
    value_col: str = "sum_el_hourly_value",
) -> pd.DataFrame:
    """
    Accepts ANY granulaarsus (15min või 60min) ja tagastab tunniseeriana.
    - 'sum_to_hour': summeerib 15min -> 60min (NaN loetakse 0-ks)
      * Kui andmed on juba tunnised, jääb väärtus samaks.
    - 'top_of_hour_only': võtab ainult minute == 0 kirjed (ignoreerib veerandtunde).
    """
    if df_local.empty:
        return df_local

    s = (
        df_local.drop_duplicates(subset=["sum_cons_time"])
        .set_index("sum_cons_time")
        .sort_index()[value_col]
    )

    if strategy == "top_of_hour_only":
        hourly = s[s.index.minute == 0]
        hourly = hourly.to_frame(name=value_col).reset_index()
        return hourly

    # default: sum_to_hour
    hourly = s.fillna(0).resample("H").sum()  # timezone-aware resample
    hourly = hourly.rename(value_col).to_frame().reset_index()
    return hourly

# ---------------------------
# Public API
# ---------------------------


def get_hourly_consumption(
    months: int = 24,
    end_exclusive_local: Optional[pd.Timestamp] = None,
    tz: str = LOCAL_TZ,
    exclude_today: bool = True,
    add_weekday: bool = True,
    add_holidays: bool = True,
    impute_missing: bool = True,
    aggregate_strategy: Literal["sum_to_hour",
                                "top_of_hour_only"] = "sum_to_hour",
) -> pd.DataFrame:
    """
    Return hourly consumption in local time for the last `months` months.
    Alates 01.10.2025 toetab ka 15min sisendit – teisendame tunniks vastavalt `aggregate_strategy`.
    - end_exclusive_local: default = today 00:00 local (excludes today completely)
    - exclude_today: if True, end_exclusive_local is forced to today 00:00
    Columns: sum_cons_time(tz-aware), sum_el_hourly_value, [imputed], [weekday,is_weekend,is_holiday]
    """
    now_local = pd.Timestamp.now(tz=tz)
    today_local = now_local.normalize()
    if end_exclusive_local is None or exclude_today:
        end_exclusive_local = today_local
    start_inclusive_local = end_exclusive_local - relativedelta(months=months)

    # Fetch in chunks via UTC
    parts = []
    for a_local, b_local in _chunk_bounds(start_inclusive_local, end_exclusive_local, months_per_chunk=12):
        a_utc = a_local.tz_convert("UTC").to_pydatetime()
        b_utc = b_local.tz_convert("UTC").to_pydatetime()
        parts.append(_fetch_chunk(a_utc, b_utc))

    raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["sum_cons_time", "sum_el_hourly_value"]
    )
    if raw.empty:
        raw["sum_cons_time"] = pd.Series(dtype="datetime64[ns, UTC]")
        raw["sum_el_hourly_value"] = pd.Series(dtype="float")
        return raw

    # Deduplicate, sort, convert to local time
    raw = (
        raw.drop_duplicates(subset=["sum_cons_time"])
           .sort_values("sum_cons_time")
           .reset_index(drop=True)
    )
    raw["sum_cons_time"] = raw["sum_cons_time"].dt.tz_convert(tz)

    # Strict local cut [start, end)
    raw = _hard_cut_local(raw, start_inclusive_local, end_exclusive_local)

    # NEW: normalize to hourly (supports 15min input)
    df = _aggregate_to_hour(raw, strategy=aggregate_strategy,
                            value_col="sum_el_hourly_value")

    # Optional imputation (applies to hourly series post-aggregation)
    if impute_missing:
        df = _impute_neighbors_mean(df)

    # Optional enrichments
    if add_weekday:
        _add_weekday_weekend(df)
    else:
        df["weekday"] = pd.Series(dtype="object")
        df["is_weekend"] = pd.Series(dtype="boolean")

    if add_holidays:
        local_dates = df["sum_cons_time"].dt.date if not df.empty else pd.Series(
            [], dtype="object")
        tmp = pd.DataFrame({"_date": local_dates})
        _add_holidays_localdate(tmp, "_date")
        df["is_holiday"] = tmp["is_holiday"].values
    else:
        df["is_holiday"] = pd.Series(dtype="boolean")

    return df


def get_daily_consumption(
    months: int = 24,
    end_exclusive_local: Optional[pd.Timestamp] = None,
    tz: str = LOCAL_TZ,
    exclude_today: bool = True,
    add_weekday: bool = True,
    add_holidays: bool = True,
    impute_missing_hourly: bool = True,
    aggregate_strategy: Literal["sum_to_hour",
                                "top_of_hour_only"] = "sum_to_hour",
) -> pd.DataFrame:
    """
    Aggregate hourly -> daily in local time (calendar days).
    Columns: sum_cons_date(date), sum_el_daily_value, [weekday,is_weekend,is_holiday]
    """
    hourly = get_hourly_consumption(
        months=months,
        end_exclusive_local=end_exclusive_local,
        tz=tz,
        exclude_today=exclude_today,
        add_weekday=False,
        add_holidays=False,
        impute_missing=impute_missing_hourly,
        aggregate_strategy=aggregate_strategy,
    )

    if hourly.empty:
        return pd.DataFrame(columns=["sum_cons_date", "sum_el_daily_value", "weekday", "is_weekend", "is_holiday"])

    idx = hourly.set_index("sum_cons_time")
    daily = (
        idx["sum_el_hourly_value"]
        .resample("D")   # local calendar days
        .sum(min_count=1)
        .rename("sum_el_daily_value")
        .reset_index()
    )

    if exclude_today:
        now_local = pd.Timestamp.now(tz=tz).normalize()
        daily = daily[daily["sum_cons_time"] <
                      now_local].reset_index(drop=True)

    daily["sum_cons_date"] = daily["sum_cons_time"].dt.date
    daily = daily[["sum_cons_date", "sum_el_daily_value"]
                  ].sort_values("sum_cons_date").reset_index(drop=True)

    if add_weekday:
        daily["weekday"] = pd.to_datetime(daily["sum_cons_date"]).dt.day_name()
        daily["is_weekend"] = daily["weekday"].isin(["Saturday", "Sunday"])
    else:
        daily["weekday"] = pd.Series(dtype="object")
        daily["is_weekend"] = pd.Series(dtype="boolean")

    if add_holidays:
        _add_holidays_localdate(daily, "sum_cons_date")
    else:
        daily["is_holiday"] = pd.Series(dtype="boolean")

    return daily


# ---------------------------
# Optional: CLI for one-off exports
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Elering consumption downloader (Europe/Tallinn).")
    parser.add_argument("--months", type=int, default=24,
                        help="How many months back (default: 24).")
    parser.add_argument("--tz", type=str, default=LOCAL_TZ,
                        help="Timezone (default: Europe/Tallinn).")
    parser.add_argument("--include-today", action="store_true",
                        help="Include today (by default excluded).")
    parser.add_argument("--no-impute", action="store_true",
                        help="Disable hourly imputation.")
    parser.add_argument("--no-weekday", action="store_true",
                        help="Do not add weekday/is_weekend.")
    parser.add_argument("--no-holidays", action="store_true",
                        help="Do not add is_holiday.")
    parser.add_argument("--daily", action="store_true",
                        help="Export daily instead of hourly.")
    parser.add_argument("--outdir", type=str, default="output",
                        help="Output folder (default: output).")
    parser.add_argument("--hour-mode", type=str, default="sum", choices=["sum", "top"],
                        help="How to build hours when API returns 15min: 'sum' (sum_to_hour, NaN=0) or 'top' (top_of_hour_only).")
    args = parser.parse_args()

    exclude_today = not args.include_today
    agg = "sum_to_hour" if args.hour_mode == "sum" else "top_of_hour_only"

    if args.daily:
        df = get_daily_consumption(
            months=args.months,
            tz=args.tz,
            exclude_today=exclude_today,
            add_weekday=not args.no_weekday,
            add_holidays=not args.no_holidays,
            impute_missing_hourly=not args.no_impute,
            aggregate_strategy=agg,
        )
        now_local = pd.Timestamp.now(tz=args.tz)
        end_local = now_local.normalize() if exclude_today else now_local
        start_local = end_local - relativedelta(months=args.months)
        fname = f"elering_consumption_daily_last{args.months}months_{start_local.strftime('%Y%m%d')}_{(end_local - pd.Timedelta(seconds=1)).strftime('%Y%m%d')}.csv"
    else:
        df = get_hourly_consumption(
            months=args.months,
            tz=args.tz,
            exclude_today=exclude_today,
            add_weekday=not args.no_weekday,
            add_holidays=not args.no_holidays,
            impute_missing=not args.no_impute,
            aggregate_strategy=agg,
        )
        now_local = pd.Timestamp.now(tz=args.tz)
        end_local = now_local.normalize() if exclude_today else now_local
        start_local = end_local - relativedelta(months=args.months)
        fname = f"elering_consumption_hourly_last{args.months}months_{start_local.strftime('%Y%m%d')}_{(end_local - pd.Timedelta(seconds=1)).strftime('%Y%m%d')}.csv"

    outpath = os.path.join(args.outdir, fname)
    saved = _safe_save_csv(df, outpath)
    print(f"Saved: {saved}")
