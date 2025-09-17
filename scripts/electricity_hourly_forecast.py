# hourly_forecast.py
# Tunnipõhine prognoos järgmiseks 7 päevaks (Europe/Tallinn)
# - Päevaprognoos el_consumption_forecast.forecast_next7(...) kaudu (või --daily-csv)
# - Tunniks jaotus weekday_profile.split_daily_forecast_to_hourly(...)
# - DST- ja pühadeteadlik; väljund tasakaalustub tagasi päevatasemele

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


# --------------------- utilid ---------------------
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


# --------------------- abi: failist import ---------------------
def _import_module_from_file(fname: str, modname: str):
    """Lae .py moodul kindlast failist (registreeri sys.modules alla enne exec’i)."""
    import importlib.util
    import contextlib
    fpath = BASE_DIR / fname
    if not fpath.exists():
        raise FileNotFoundError(f"{fname} ei leitud samas kaustas.")
    spec = importlib.util.spec_from_file_location(modname, str(fpath))
    if spec is None or spec.loader is None:
        raise ImportError(f"Ei saanud importida: {fname}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        spec.loader.exec_module(mod)
    return mod


# --------------------- päevaprognoosi laadimine ---------------------
def load_daily_forecast_from_module(
    mode: str = "season",
    segmented_bias: bool = True,
    months_hist: int = 24,
    temp_module: Optional[str] = None,
    temp_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Kutsub el_consumption_forecast.forecast_next7(...) ja tagastab DataFrame'i.
    Eeldab, et el_consumption_forecast.py on samas kaustas.
    """
    ecf = _import_module_from_file(
        "el_consumption_forecast.py", "el_consumption_forecast")
    if not hasattr(ecf, "forecast_next7"):
        raise RuntimeError("el_consumption_forecast.forecast_next7 puudub.")
    df = ecf.forecast_next7(
        mode=mode,
        segmented_bias=segmented_bias,
        months_hist=months_hist,
        temp_module=temp_module,
        temp_csv=temp_csv,
    )
    # miinimum: date_local (YYYY-MM-DD), yhat_consumption (float)
    need = {"date_local", "yhat_consumption"}
    if not need.issubset(df.columns):
        raise RuntimeError(
            "Päevaprognoos ei sisalda vajalikke veerge: date_local, yhat_consumption")
    return df


def load_daily_forecast_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # kuupäev
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
                "CSV ei sisalda veergu 'date_local' ega tuletatavat kuupäeva.")
    # väärtus
    if "yhat_consumption" not in df.columns:
        for cand in ["yhat", "forecast", "consumption", "daily_yhat", "yhat_day"]:
            if cand in df.columns:
                df["yhat_consumption"] = pd.to_numeric(
                    df[cand], errors="coerce")
                break
    if "yhat_consumption" not in df.columns:
        raise RuntimeError(
            "CSV ei sisalda päevaprognoosi veergu (yhat_consumption).")
    return df


# --------------------- tunniks jaotamine ---------------------
def split_daily_to_hourly(
    daily_df: pd.DataFrame,
    last_n: int = 6,
    holiday_profile: str = "weekday",
    hourly_csv: Optional[str] = None,
    csv_tz: str = LOCAL_TZ,
    months_for_profile: int = 24,
) -> pd.DataFrame:
    """
    Kutsub weekday_profile.split_daily_forecast_to_hourly(...) (Eesti aeg, DST-aware).
    """
    wp = _import_module_from_file("weekday_profile.py", "weekday_profile")
    if not hasattr(wp, "split_daily_forecast_to_hourly"):
        raise RuntimeError(
            "weekday_profile.split_daily_forecast_to_hourly puudub.")
    hourly = wp.split_daily_forecast_to_hourly(
        daily_df,
        date_col="date_local",
        value_col="yhat_consumption",
        # laseb ise arvutada (viimased 'last_n' esinemist)
        share_matrix=None,
        last_n=last_n,
        exclude_today=True,           # profiilidest tänane välja
        holiday_profile=holiday_profile,
        hourly_csv=hourly_csv,        # kui soovid profiili ehitada CSV põhjal
        csv_tz=csv_tz,
        months=months_for_profile,    # kui profiil ehitatakse API-st, kui palju ajalugu võtta
    )
    # mugav lisaveerg
    hourly["date_local"] = hourly["datetime_local"].dt.strftime("%Y-%m-%d")
    return hourly


# --------------------- kontroll ja salvestus ---------------------
def check_daily_hourly_match(hourly_df: pd.DataFrame, daily_df: pd.DataFrame) -> float:
    """
    Kontroll: kas tunnid summeeruvad päevaprognoosiks. Tagastab max abs suhteline viga (0..).
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


# --------------------- CLI töövoog ---------------------
def main(
    # päevaprognoos
    use_daily_csv: Optional[str] = None,
    mode: str = "season",
    segmented_bias: bool = True,
    months_hist: int = 24,
    temp_module: Optional[str] = None,
    temp_csv: Optional[str] = None,
    # päeva-kõver / jaotus
    last_n: int = 6,
    holiday_profile: str = "weekday",      # 'weekday'|'sunday'|'weekend_avg'
    # kui soovid profiili ehitada konkreetsest tunnise CSV-st
    hourly_csv: Optional[str] = None,
    # kui hourly_csv ajatemplid on tz-naive, millise vööndina lugeda (nt 'UTC')
    csv_tz: str = LOCAL_TZ,
    months_for_profile: int = 24,
    # väljund
    save_csv: bool = False,
):
    # 1) too päevaprognoos
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

    # 2) jaota tunniks (EE aeg)
    hourly = split_daily_to_hourly(
        daily_df=daily,
        last_n=last_n,
        holiday_profile=holiday_profile,
        hourly_csv=hourly_csv,
        csv_tz=csv_tz,
        months_for_profile=months_for_profile,
    )

    # 3) lisa päevameta — VÄLDIME duplikaatnimesid (weekday jääb tunnitabelist)
    meta_keep = [c for c in [
        "segment", "season", "is_weekend", "is_holiday",
        "month_num", "EE_avg_temp_C", "bias_key", "bias_factor",
        "yhat_base", "yhat_consumption"
    ] if c in daily.columns]
    daily_meta = daily[["date_local"] +
                       meta_keep].drop_duplicates("date_local")

    # NB: suffixes=("", "_daily") hoiab tunnitabeli veerud nimedeta muutmata
    out = hourly.merge(daily_meta, on="date_local",
                       how="left", suffixes=("", "_daily"))

    # 4) kontroll – kas tunnid summeeruvad päevadeks
    max_rel = check_daily_hourly_match(out, daily)
    if np.isfinite(max_rel):
        print(
            f"[kontroll] Päevasummad vs yhat_consumption — max |rel_diff| ≈ {max_rel*100:.5f}%")

    # 5) väljund
    out = out.sort_values(["datetime_local"]).reset_index(drop=True)

    print("\n=== Hourly forecast (esimesed 48 rida) ===")
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
        description="Järgmise 7 päeva TUNNI-põhine prognoos (Europe/Tallinn)")
    # Päevaprognoosi allikas
    p.add_argument("--daily-csv", type=str, default=None,
                   help="Kasuta valmis päevaprognoosi CSV-d (date_local,yhat_consumption)")
    p.add_argument("--mode", choices=["season", "month"], default="season",
                   help="Bias-tüüp päevaprognoosile (kui ei kasuta --daily-csv)")
    p.add_argument("--segmented-bias", dest="segmented_bias",
                   action="store_true", help="Bias segmendi kaupa (workday/offday)")
    p.add_argument("--no-seg-bias", dest="segmented_bias",
                   action="store_false", help="Bias mitte segmendi kaupa")
    p.add_argument("--months", type=int, default=24,
                   help="Ajalugu (kuudes) päevaprognoosi mudelite/biasi jaoks")
    p.add_argument("--temp-module", type=str, default=None,
                   help="Ilma moodul (nt temp_forecast.py)")
    p.add_argument("--temp-csv", type=str, default=None,
                   help="Ilma CSV (sisaldab datetime + EE_avg või linnaveerud)")
    # Päeva kõver
    p.add_argument("--last-n", type=int, default=6,
                   help="Mitu viimast esinemist iga nädalapäeva jaoks profiilis")
    p.add_argument("--holiday-profile", choices=["weekday", "sunday", "weekend_avg"], default="weekday",
                   help="Kuidas käsitleda riigipüha jaotusel")
    p.add_argument("--hourly-csv", type=str, default=None,
                   help="Kui soovid profiili ehitada konkreetsest tunnise CSV-st (möödudes API-st)")
    p.add_argument("--csv-tz", type=str, default=LOCAL_TZ,
                   help="Kui --hourly-csv ajalised väärtused on TZ-naive, millise vööndina lugeda (nt 'UTC')")
    p.add_argument("--months-for-profile", type=int, default=24,
                   help="Kui palju ajalugu võtta profiili jaoks (API režiimis)")
    # Väljund
    p.add_argument("--save-csv", action="store_true",
                   help="Salvesta CSV kausta output/")

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
