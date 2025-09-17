# weekday_profile.py
# Päeva kõverad (tunnijaotus) Europe/Tallinn ajas + päevaprognoosi jaotus tunniks
from __future__ import annotations

from typing import Tuple, List, Optional
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/Tallinn"
BASE_DIR = Path(__file__).resolve().parent
OUTDIR = BASE_DIR / "output"
OUTDIR.mkdir(parents=True, exist_ok=True)

WEEKDAY_ORDER: List[str] = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
]

# --------------------------------------------------------------------
# Tunnitarbimise allikas: eelistus elering_consumption.get_hourly_consumption()
# Fallback: --hourly-csv (valikuline); CSV-s peab olema aeg + tunniväärtus
# --------------------------------------------------------------------


def _import_elering_client():
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "elering_consumption", str(BASE_DIR / "elering_consumption.py"))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def _read_hourly_csv(path: str, csv_tz: str = LOCAL_TZ) -> pd.DataFrame:
    df = pd.read_csv(path)
    # leia aeg
    ts_col = None
    for cand in ["sum_cons_time", "datetime", "time", "timestamp", "dt"]:
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        # proovi 1. veerg
        cand = df.columns[0]
        try:
            pd.to_datetime(df[cand])
            ts_col = cand
        except Exception:
            raise RuntimeError(f"CSV-st ei leia ajaveergu: {path}")

    # leia väärtus
    val_col = None
    for cand in ["sum_el_hourly_value", "consumption", "value", "el_hourly", "load"]:
        if cand in df.columns:
            val_col = cand
            break
    if val_col is None:
        # võta esimene numbriline veerg peale ajaveeru
        for c in df.columns:
            if c == ts_col:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                val_col = c
                break
    if val_col is None:
        raise RuntimeError(f"CSV-st ei leia tunnitarbimise veergu: {path}")

    out = df[[ts_col, val_col]].rename(
        columns={ts_col: "sum_cons_time", val_col: "sum_el_hourly_value"}).copy()
    out["sum_cons_time"] = pd.to_datetime(
        out["sum_cons_time"], errors="coerce")

    # kui TZ puudub -> eelda csv_tz; kui olemas -> konverteeri EE-sse
    if out["sum_cons_time"].dt.tz is None:
        out["sum_cons_time"] = out["sum_cons_time"].dt.tz_localize(
            csv_tz).dt.tz_convert(LOCAL_TZ)
    else:
        out["sum_cons_time"] = out["sum_cons_time"].dt.tz_convert(LOCAL_TZ)

    out["sum_el_hourly_value"] = pd.to_numeric(
        out["sum_el_hourly_value"], errors="coerce")
    return out.dropna(subset=["sum_cons_time", "sum_el_hourly_value"])


def load_hourly_source(
    hourly_csv: Optional[str] = None,
    csv_tz: str = LOCAL_TZ,
    months: int = 24,
    exclude_today: bool = True
) -> pd.DataFrame:
    """
    Tagasta tunnipõhine DF veergudega:
      - sum_cons_time (tz-aware, Europe/Tallinn)
      - sum_el_hourly_value (float)
    Eelistus: elering_consumption.get_hourly_consumption(...) -> CSV.
    """
    if hourly_csv:
        return _read_hourly_csv(hourly_csv, csv_tz=csv_tz)

    client = _import_elering_client()
    if client and hasattr(client, "get_hourly_consumption"):
        df = client.get_hourly_consumption(
            months=months,
            tz=LOCAL_TZ,
            exclude_today=exclude_today,
            add_weekday=False,    # teeme ise
            add_holidays=False,   # teeme ise
            impute_missing=True,
        )
        need = {"sum_cons_time", "sum_el_hourly_value"}
        if not need.issubset(df.columns):
            raise RuntimeError(
                "elering_consumption.get_hourly_consumption() ei tagasta nõutud veerge.")
        # norm TZ
        df = df.copy()
        df["sum_cons_time"] = pd.to_datetime(
            df["sum_cons_time"], errors="coerce")
        if df["sum_cons_time"].dt.tz is None:
            df["sum_cons_time"] = df["sum_cons_time"].dt.tz_localize(LOCAL_TZ)
        else:
            df["sum_cons_time"] = df["sum_cons_time"].dt.tz_convert(LOCAL_TZ)
        df["sum_el_hourly_value"] = pd.to_numeric(
            df["sum_el_hourly_value"], errors="coerce")
        return df.dropna(subset=["sum_cons_time", "sum_el_hourly_value"])

    raise FileNotFoundError(
        "Ei leia tunniseid andmeid. Lahendused:\n"
        " - hoia elering_consumption.py samas kaustas (funktsioon get_hourly_consumption)\n"
        " - või anna CSV tee: --hourly-csv <path> (vajadusel --csv-tz UTC, kui CSV ajatemplid on UTC-naive)"
    )

# --------------------------------------------------------------------
# Pühad (EE) – määrame kohalikust kuupäevast
# --------------------------------------------------------------------


def _attach_ee_holidays_local(ts: pd.Series) -> pd.Series:
    """Tagasta bool-seeria 'is_holiday' Eesti riigipühade järgi (kohalik kuupäev)."""
    try:
        import holidays
    except Exception:
        return pd.Series(False, index=ts.index)

    s = pd.to_datetime(ts, errors="coerce")
    if s.dt.tz is None:
        s = s.dt.tz_localize(LOCAL_TZ)
    else:
        s = s.dt.tz_convert(LOCAL_TZ)

    local_dates = s.dt.date
    if len(local_dates) == 0:
        return pd.Series(False, index=ts.index)

    years = range(local_dates.min().year, local_dates.max().year + 1)
    try:
        ee = holidays.country_holidays("EE", years=years)
        return pd.Series([d in ee for d in local_dates], index=ts.index, dtype="bool")
    except Exception:
        return pd.Series(False, index=ts.index)

# --------------------------------------------------------------------
# Sisendi ettevalmistus (EE aeg, pühad välja, tänane välja)
# --------------------------------------------------------------------


def _prepare_hourly_df(
    hourly_csv: Optional[str] = None,
    csv_tz: str = LOCAL_TZ,
    exclude_today: bool = True,
    months: int = 24
) -> pd.DataFrame:
    """
    Vorminda sisend **Eesti aja järgi** ja lisa abiveerud; pühad + tänane välja.
    """
    df = load_hourly_source(hourly_csv=hourly_csv, csv_tz=csv_tz,
                            months=months, exclude_today=exclude_today)
    if df.empty:
        raise RuntimeError("weekday_profile: sisendandmestik on tühi.")

    # Tänane (poolik) välja
    if exclude_today:
        today_local = pd.Timestamp.now(tz=LOCAL_TZ).normalize().date()
        df = df[df["sum_cons_time"].dt.date < today_local]

    # Pühad (EE kohalik kalendrikuupäev)
    is_holiday = _attach_ee_holidays_local(df["sum_cons_time"])
    df = df[~is_holiday].copy()
    if df.empty:
        raise RuntimeError(
            "weekday_profile: pärast pühade/tänase välistamist puuduvad andmed.")

    # Kohalikud abiveerud
    df["date_local"] = df["sum_cons_time"].dt.date
    df["hour_local"] = df["sum_cons_time"].dt.hour
    df["weekday"] = df["sum_cons_time"].dt.day_name()  # Monday..Sunday
    return df

# --------------------------------------------------------------------
# Profiili ehitus (DST-safe): iga päeva 24-elemendiline share-vektor
# --------------------------------------------------------------------


def _day_share_vector_24(day_df: pd.DataFrame) -> np.ndarray:
    """
    Ehita ühe kalendripäeva (EE) 24-elemendiline osakaaluvektor:
      - summeeri topelttunnid (25h päeval),
      - puuduv tund (23h päeval) = 0,
      - renormeeritakse nii, et summa = 1.0 (kui päevane kogus >0).
    """
    # tunnisummad antud päeval
    sums = day_df.groupby("hour_local")["sum_el_hourly_value"].sum()
    vec = pd.Series(0.0, index=range(24), dtype=float)
    for h, v in sums.items():
        if 0 <= int(h) <= 23:
            vec[int(h)] += float(v)

    total = vec.sum()
    if total > 0:
        vec = vec / total
    # kui total==0 -> kõik nullid (nii on õige)
    return vec.to_numpy(dtype=float)


def _build_profiles_and_days_used(df: pd.DataFrame, last_n: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tagastab:
      profile_df: (weekday, hour_local, avg_hourly_share_of_day, n_days)
      days_used:  (weekday, date_local) – millised kuupäevad kaasati
    """
    profiles = []
    days_rows = []

    for wd in WEEKDAY_ORDER:
        d_wd = df[df["weekday"] == wd]
        if d_wd.empty:
            continue

        # viimased 'last_n' kuupäeva selle nädalapäeva kohta
        last_days = (d_wd[["date_local"]].drop_duplicates()
                     .sort_values("date_local")
                     .tail(last_n)["date_local"].tolist())
        if not last_days:
            continue

        # kogu päevade share'id, siis elementwise keskmine
        share_stack = []
        for d in last_days:
            day_df = d_wd[d_wd["date_local"] == d]
            vec = _day_share_vector_24(day_df)
            share_stack.append(vec)
            days_rows.append({"weekday": wd, "date_local": d})

        if not share_stack:
            continue

        mean_vec = np.mean(np.vstack(share_stack), axis=0)  # 24 elem
        prof = pd.DataFrame({
            "hour_local": np.arange(24, dtype=int),
            "avg_hourly_share_of_day": mean_vec,
            "weekday": wd,
            "n_days": len(share_stack),
        })
        profiles.append(prof)

    if not profiles:
        raise RuntimeError(
            "weekday_profile: profiile ei õnnestunud koostada (andmeid napib).")

    profile_df = pd.concat(profiles, ignore_index=True).sort_values(
        ["weekday", "hour_local"])
    days_used = pd.DataFrame(days_rows, columns=[
                             "weekday", "date_local"]).sort_values(["weekday", "date_local"])
    return profile_df.reset_index(drop=True), days_used.reset_index(drop=True)


def _profile_df_to_share_matrix(profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    profile_df -> 24x7 maatriks; iga veeru summa = 1.0.
    Kui mõni nädalapäev puudub, täida ühtlase jaotusega (1/24).
    """
    mat = (profile_df.pivot(index="hour_local", columns="weekday", values="avg_hourly_share_of_day")
           .reindex(range(24)))
    for wd in WEEKDAY_ORDER:
        if wd not in mat.columns:
            mat[wd] = np.nan
    mat = mat[WEEKDAY_ORDER].astype(float)

    col_sums = mat.sum(axis=0, skipna=True)
    for wd in mat.columns:
        s = col_sums.get(wd, 0.0)
        if pd.notna(s) and s > 0:
            mat[wd] = mat[wd] / s
        else:
            mat[wd] = 1.0 / 24.0

    mat.index.name = "hour_local"
    mat.columns.name = "weekday"
    return mat

# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------


def get_weekday_hour_share_matrix(
    last_n: int = 6,
    exclude_today: bool = True,
    hourly_csv: Optional[str] = None,
    csv_tz: str = LOCAL_TZ,
    months: int = 24
) -> pd.DataFrame:
    """Tagastab 24×7 tunnijaotuse maatriksi Eesti aja järgi (veeru-summa==1.0)."""
    df_hourly = _prepare_hourly_df(hourly_csv=hourly_csv, csv_tz=csv_tz,
                                   exclude_today=exclude_today, months=months)
    profile_df, _ = _build_profiles_and_days_used(df_hourly, last_n=last_n)
    return _profile_df_to_share_matrix(profile_df)


def get_weekday_days_used(
    last_n: int = 6,
    exclude_today: bool = True,
    hourly_csv: Optional[str] = None,
    csv_tz: str = LOCAL_TZ,
    months: int = 24
) -> pd.DataFrame:
    """Tagastab (weekday, date_local) – millised 'last_n' kuupäeva iga nädalapäeva kohta arvesse läksid."""
    df_hourly = _prepare_hourly_df(hourly_csv=hourly_csv, csv_tz=csv_tz,
                                   exclude_today=exclude_today, months=months)
    _, days_used = _build_profiles_and_days_used(df_hourly, last_n=last_n)
    return days_used

# --------------------------------------------------------------------
# Päevaprognoosi jaotus tunniks (EE ajas, DST-aware)
# --------------------------------------------------------------------


def _make_local_hour_range(day: pd.Timestamp) -> pd.DatetimeIndex:
    start = day.tz_localize(LOCAL_TZ)
    end = (day + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ)
    return pd.date_range(start, end, freq="H", inclusive="left", tz=LOCAL_TZ)


def _stretch_shares_for_dst(base_shares: np.ndarray, hours_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Kohanda 24-elemendiline jaotus lutile:
      - 23h päeval: puuduv tund eemaldatakse ja renormeeritakse,
      - 25h päeval: duplikaattunni osakaal poolitatakse kaheks (kokku sama osakaal), renormeeritakse.
    """
    hours = [ts.hour for ts in hours_index]
    if len(hours) == 24:
        return base_shares.copy()

    shares_map = {h: base_shares[h] for h in range(24)}

    if len(hours) == 23:
        missing = set(range(24)) - set(hours)
        miss_h = next(iter(missing))
        present = [h for h in range(24) if h != miss_h]
        v = np.array([shares_map[h] for h in present], dtype=float)
        v = v / v.sum()
        return v

    if len(hours) == 25:
        seen = set()
        dup_h = None
        for h in hours:
            if h in seen:
                dup_h = h
                break
            seen.add(h)
        v = []
        split_once = False
        for h in hours:
            if h == dup_h and not split_once:
                v.append(shares_map[h] / 2.0)
                split_once = True
            elif h == dup_h and split_once:
                v.append(shares_map[h] / 2.0)
            else:
                v.append(shares_map[h])
        v = np.array(v, dtype=float)
        v = v / v.sum()
        return v

    # ootamatu (mitte 23/24/25)
    v = np.ones(len(hours), dtype=float)
    return v / v.size


def split_daily_forecast_to_hourly(
    daily_df: pd.DataFrame,
    date_col: str = "date_local",
    value_col: str = "yhat_consumption",
    share_matrix: Optional[pd.DataFrame] = None,
    last_n: int = 6,
    exclude_today: bool = True,
    holiday_profile: str = "weekday",   # 'weekday' | 'sunday' | 'weekend_avg'
    hourly_csv: Optional[str] = None,
    csv_tz: str = LOCAL_TZ,
    months: int = 24
) -> pd.DataFrame:
    """
    Võtab päevaprognoosi (nt el_consumption_forecast.py väljund) ja jaotab tunniks.
    Tagastab veerud: ['datetime_local','weekday','hour_local','consumption_hourly'].
    """
    if share_matrix is None:
        share_matrix = get_weekday_hour_share_matrix(last_n=last_n, exclude_today=exclude_today,
                                                     hourly_csv=hourly_csv, csv_tz=csv_tz, months=months)

    # precompute weekend_avg, kui vaja
    weekend_avg = None
    if holiday_profile == "weekend_avg":
        weekend_avg = (share_matrix["Saturday"].to_numpy(
        ) + share_matrix["Sunday"].to_numpy()) / 2.0

    # pühad
    try:
        import holidays
        ee_holidays = holidays.country_holidays("EE", years=range(2000, 2100))
    except Exception:
        ee_holidays = set()

    out_rows = []
    for _, row in daily_df.iterrows():
        # kuupäev -> Timestamp (00:00 EE)
        day = pd.Timestamp(str(row[date_col]))
        hours_idx = _make_local_hour_range(day)

        weekday_name = hours_idx[0].day_name()  # Monday..Sunday
        is_holiday = (day.date() in ee_holidays) if hasattr(
            ee_holidays, "__contains__") else False

        # vali jaotus
        if is_holiday and holiday_profile == "sunday":
            base = share_matrix["Sunday"].to_numpy()
        elif is_holiday and holiday_profile == "weekend_avg":
            base = weekend_avg
        else:
            base = share_matrix[weekday_name].to_numpy()

        shares = _stretch_shares_for_dst(base, hours_idx)
        hourly_vals = float(row[value_col]) * shares

        for ts, hval in zip(hours_idx, hourly_vals):
            out_rows.append({
                "datetime_local": ts,                # tz-aware EE
                "weekday": ts.day_name(),
                "hour_local": ts.hour,
                "consumption_hourly": float(hval),
            })

    out = pd.DataFrame(out_rows).sort_values(
        "datetime_local").reset_index(drop=True)
    return out


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Päeva kõverad (Eesti aeg) ja päevaprognoosi jaotus tunniks")
    parser.add_argument("--last-n", type=int, default=6,
                        help="Mitu viimast esinemist iga nädalapäeva jaoks")
    parser.add_argument("--include-today", action="store_true",
                        help="Kaasa tänane päev profiili arvutusse (vaikimisi mitte)")
    parser.add_argument("--save-matrix", action="store_true",
                        help="Salvesta profiilimaatriks CSV-na output/ kausta")
    parser.add_argument("--apply-daily-csv", type=str, default=None,
                        help="Jaga CSV-s olev päevaprognoos (date_local,yhat_consumption) tunniks")
    parser.add_argument("--holiday-profile", choices=["weekday", "sunday", "weekend_avg"],
                        default="weekday", help="Kuidas käsitleda riigipühi jaotusel")
    parser.add_argument("--hourly-csv", type=str, default=None,
                        help="(Valikuline) tunnise tarbimise CSV, kui API-moodulit pole")
    parser.add_argument("--csv-tz", type=str, default=LOCAL_TZ,
                        help="Kui CSV ajatemplid on TZ-naive, loe neid selles vööndis (nt 'UTC')")
    parser.add_argument("--months", type=int, default=24,
                        help="Kui pikalt minevikku (kuud) profiili arvutuseks")
    parser.add_argument("--outdir", type=str,
                        default=str(OUTDIR), help="Väljundkaust")
    args = parser.parse_args()

    exclude_today = not args.include_today
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Profiilimaatriks
    M = get_weekday_hour_share_matrix(last_n=args.last_n, exclude_today=exclude_today,
                                      hourly_csv=args.hourly_csv, csv_tz=args.csv_tz, months=args.months)
    DU = get_weekday_days_used(last_n=args.last_n, exclude_today=exclude_today,
                               hourly_csv=args.hourly_csv, csv_tz=args.csv_tz, months=args.months)

    print("\n=== 24x7 profiilimaatriks (veerud sum==1.0) – Eesti aeg ===")
    print(M.round(4).to_string())

    print("\n=== Arvesse läinud kuupäevad (pühad välistatud) ===")
    print(DU.to_string(index=False))

    if args.save_matrix:
        (outdir / "weekday_share_matrix.csv").write_text(M.to_csv(index=True), encoding="utf-8")
        (outdir / "weekday_days_used.csv").write_text(DU.to_csv(index=False), encoding="utf-8")
        print(f"[saved] {outdir / 'weekday_share_matrix.csv'}")
        print(f"[saved] {outdir / 'weekday_days_used.csv'}")

    # 2) Päevaprognoosi jaotus tunniks (soovi korral)
    if args.apply_daily_csv:
        daily_df = pd.read_csv(args.apply_daily_csv)
        hourly = split_daily_forecast_to_hourly(
            daily_df,
            date_col="date_local",
            value_col="yhat_consumption",
            share_matrix=M,
            last_n=args.last_n,
            exclude_today=exclude_today,
            holiday_profile=args.holiday_profile,
            hourly_csv=args.hourly_csv,
            csv_tz=args.csv_tz,
            months=args.months,
        )
        hourly_path = outdir / "forecast_hourly_split.csv"
        hourly.to_csv(hourly_path, index=False)
        print(f"[saved] {hourly_path}")
