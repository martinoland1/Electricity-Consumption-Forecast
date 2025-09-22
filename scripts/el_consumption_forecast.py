# el_consumption_forecast.py
# Päevatarbimise prognoos järgmiseks 7 päevaks alates homsest (Europe/Tallinn)
# - Ilma sisend: temp_forecast.py (vaikimisi) või --temp-module / --temp-csv
# - Segmenteeritud regressioon (workday/offday) + bias (hooaja/kuu) bias_analysis.py-st
# - Väljund: DataFrame (print), valikuline CSV + graafik

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
# Seaded/kaustad
# ---------------------------
LOCAL_TZ = "Europe/Tallinn"
BASE_DIR = Path(__file__).resolve().parent
OUTDIR = BASE_DIR / "output"
OUTDIR.mkdir(parents=True, exist_ok=True)

SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring",  4: "spring",  5: "spring",
    6: "summer",  7: "summer",  8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}

# ---------------------------
# Ohutu salvestus
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
# Ilma laadimine (moodul/CSV) + päevaksmine lokaalse kalendri järgi
# ---------------------------


def _df_to_local_daily_avg(df: pd.DataFrame, tz: str = LOCAL_TZ) -> pd.Series:
    """Eeldab datetime indexit või veergu; tagastab päevakeskmise EE_avg (Series, tz-aware D)."""
    df = df.copy()

    # 1) DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        dt_col = None
        for cand in ["date_local", "datetime", "time", "date"]:
            if cand in df.columns:
                dt_col = cand
                break
        if dt_col is None:
            raise RuntimeError(
                "Ilma andmetes puudub datetime index või veerg ('date_local'/'datetime'/'time'/'date').")
        df.index = pd.to_datetime(df[dt_col], errors="coerce")

    # 2) TZ -> LOCAL
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    else:
        idx = idx.tz_convert(tz)
    df.index = idx

    # 3) EE_avg
    if "EE_avg" not in df.columns:
        city_cols = [c for c in ["Tallinn", "Tartu", "Pärnu",
                                 "Narva", "Kuressaare"] if c in df.columns]
        if not city_cols:
            raise RuntimeError(
                "Ilma andmetes pole 'EE_avg' ega linnaveerge (Tallinn, Tartu, Pärnu, Narva, Kuressaare).")
        df["EE_avg"] = df[city_cols].mean(axis=1, skipna=True)

    # 4) tunnid -> päevad (LOCAL kalendri järgi)
    daily = df["EE_avg"].resample("D").mean()
    daily.name = "avg_temp_C"
    return daily


def _load_temp_from_module(temp_module: Optional[str]) -> Optional[pd.Series]:
    """Proovi tuua ilma andmed moodulist. Tagastab päevakeskmise (Series) või None."""
    import importlib.util
    import importlib
    import contextlib

    if temp_module is None:
        cand = BASE_DIR / "temp_forecast.py"
        if not cand.exists():
            return None
        temp_module = str(cand)

    # .py failina → spec_from_file_location; muidu → import_module
    if temp_module.endswith(".py"):
        spec = importlib.util.spec_from_file_location(
            "temp_forecast_mod", temp_module)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        # --- oluline: registreeri moodul enne exec_module (dataclass jt jaoks) ---
        sys.modules[spec.name] = mod
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            spec.loader.exec_module(mod)
    else:
        # paketinimi
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


def get_next7_local_avg_temp(tz: str = LOCAL_TZ,
                             temp_module: Optional[str] = None,
                             temp_csv: Optional[str] = None) -> pd.Series:
    """
    Tagasta Series pikkusega 7 (järgmised 7 päeva, LOCAL), nimega 'avg_temp_C'.
    Laadimise prioriteet: --temp-module -> --temp-csv -> temp_forecast.py samas kaustas.
    """
    s = None
    if temp_module:
        s = _load_temp_from_module(temp_module)
        if s is None:
            raise FileNotFoundError(f"Ei saanud ilma moodulist: {temp_module}")
    elif temp_csv:
        s = _load_temp_from_csv(temp_csv)
    else:
        s = _load_temp_from_module(None)
        if s is None:
            raise FileNotFoundError(
                "Ilma sisendit ei leitud. Kasuta --temp-module või --temp-csv.")

    # Homsest 7 päeva indeks
    today_local = pd.Timestamp.now(tz=tz).normalize()
    days = pd.date_range(today_local + pd.Timedelta(days=1),
                         periods=7, freq="D", tz=tz)
    s = s.reindex(days)
    s.index.name = "date_local"
    s.name = "avg_temp_C"
    return s

# ---------------------------
# Päevade klassifikatsioon (weekend/holiday/season)
# ---------------------------


def classify_days_local(dates_local: pd.DatetimeIndex) -> pd.DataFrame:
    is_weekend = dates_local.dayofweek >= 5
    # EE riigipühad
    try:
        import holidays
        years = list(range(dates_local[0].year, dates_local[-1].year + 1))
        ee = holidays.country_holidays("EE", years=years)
        is_holiday = pd.Index(
            [d.date() in ee for d in dates_local], dtype="bool")
    except Exception:
        print("[info] 'holidays' teek puudub; is_holiday=False")
        is_holiday = pd.Index([False] * len(dates_local), dtype="bool")

    segment = np.where(is_weekend | is_holiday, "offday", "workday")
    season = pd.Index([SEASON_MAP[m]
                      for m in dates_local.month], dtype="object")
    weekday = dates_local.day_name()
    return pd.DataFrame({
        "weekday": weekday,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "segment": segment,
        "season": season,
        "month_num": dates_local.month,
    }, index=dates_local)

# ---------------------------
# Regressioonid + bias (bias_analysis.py)
# ---------------------------


def get_models_and_bias(months_hist: int = 24, mode: str = "season", segmented_bias: bool = True):
    """
    Tagasta:
      models: {"workday":{"a":..,"b":..}, "offday":{"a":..,"b":..}}
      factors: dict — mode='season'/'month'; segmented_bias=True -> võtmed 'workday:talv' vms
    """
    import importlib.util
    import contextlib

    ba_path = BASE_DIR / "bias_analysis.py"
    if not ba_path.exists():
        raise FileNotFoundError("bias_analysis.py ei leitud samas kaustas.")

    spec = importlib.util.spec_from_file_location(
        "bias_analysis", str(ba_path))
    if spec is None or spec.loader is None:
        raise ImportError(
            "Ei saanud koostada import spec'i bias_analysis.py jaoks.")
    ba = importlib.util.module_from_spec(spec)
    # --- oluline: registreeri moodul enne exec_module (dataclass jt jaoks) ---
    sys.modules[spec.name] = ba
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        spec.loader.exec_module(ba)

    if not hasattr(ba, "get_season_bias_segmented"):
        raise RuntimeError("bias_analysis.get_season_bias_segmented puudub.")
    season_seg_df, season_map, models = ba.get_season_bias_segmented(
        months=months_hist)

    if hasattr(ba, "get_bias_factors"):
        factors, meta, _table = ba.get_bias_factors(
            mode=mode, segmented=segmented_bias, months=months_hist)
    else:
        # Fallback, kui façade puudub
        if mode == "season":
            if segmented_bias and hasattr(ba, "get_season_bias_segmented"):
                season_seg_df, season_map, _ = ba.get_season_bias_segmented(
                    months=months_hist)
                factors = {f"{k[0]}:{k[1]}": float(
                    v) for k, v in season_map.items()}
            elif hasattr(ba, "get_season_bias"):
                _, season_map = ba.get_season_bias(months=months_hist)
                factors = {k: float(v) for k, v in season_map.items()}
            else:
                factors = {}
        else:
            if hasattr(ba, "get_month_bias"):
                _, month_map = ba.get_month_bias(months=months_hist)
                factors = {int(k): float(v) for k, v in month_map.items()}
            else:
                factors = {}

    return models, factors

# ---------------------------
# Prognoosi arvutus
# ---------------------------


def forecast_next7(mode: str = "season",
                   segmented_bias: bool = True,
                   months_hist: int = 24,
                   temp_module: Optional[str] = None,
                   temp_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Arvuta päevatarbimise prognoos (7 päeva alates homsest).
    mode: 'season' | 'month'  (millist bias koefit kasutada)
    segmented_bias: True -> bias segment+season/kuunumber järgi; False -> ainult season/kuunumber
    months_hist: mitu kuud minevikku regressioonide/biasi hindamiseks
    temp_module/temp_csv: ilma sisendi allikas (vt ülal)
    """
    # 1) Ilm (EE_avg) järgmised 7 päeva
    temp_s = get_next7_local_avg_temp(
        tz=LOCAL_TZ, temp_module=temp_module, temp_csv=temp_csv)
    if temp_s.isna().any():
        print("[warn] Osad temperatuurid puuduvad ilma sisendis; vastavad päevad jäävad NaN-ks prognoosis.")

    # 2) Päevaklassid Tallinnas
    cls = classify_days_local(temp_s.index)
    # Normalize season labels to English just in case (fallback safety)
    ET2EN = {"talv": "winter", "kevad": "spring",
             "suvi": "summer", "sügis": "autumn"}
    if "season" in cls.columns:
        cls["season"] = cls["season"].replace(ET2EN)

    # 3) Regressioonid + bias
    models, factors = get_models_and_bias(
        months_hist=months_hist, mode=mode, segmented_bias=segmented_bias)
    a_w, b_w = models["workday"]["a"], models["workday"]["b"]
    a_o, b_o = models["offday"]["a"],  models["offday"]["b"]

    seg = cls["segment"].to_numpy()
    T = temp_s.to_numpy(dtype=float)

    # 4) Temp-only ennustus: y = a + b*T
    yhat_base = np.where(seg == "offday",
                         a_o + b_o * T,
                         a_w + b_w * T)

    # 5) Bias key + korrigeerimine
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
# Plot (tulbad tarbimine, joon temperatuur)
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
    ax1.set_ylabel("Forecast consumption")
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


def main(mode: str = "season",
         segmented_bias: bool = True,
         months_hist: int = 24,
         temp_module: Optional[str] = None,
         temp_csv: Optional[str] = None,
         save_csv: bool = False,
         save_plot: bool = False):
    out = forecast_next7(mode=mode, segmented_bias=segmented_bias, months_hist=months_hist,
                         temp_module=temp_module, temp_csv=temp_csv)

    # Info
    print("\n=== 7 päeva tarbimise prognoos (Europe/Tallinn) ===")
    print(
        f"- Bias mode: {mode} | segmented_bias={segmented_bias} | months_hist={months_hist}")
    print(out.to_string(index=False))

    # Salvestus
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
        description="Järgmise 7 päeva päevatarbimise prognoos (Europe/Tallinn)")
    p.add_argument("--mode", choices=["season", "month"],
                   default="season", help="Bias koefitsiendi tüüp")
    p.add_argument("--segmented-bias", dest="segmented_bias",
                   action="store_true", help="Bias segmendi kaupa (workday/offday)")
    p.add_argument("--no-seg-bias", dest="segmented_bias",
                   action="store_false", help="Bias mitte segmendi kaupa")
    p.add_argument("--months", type=int, default=24,
                   help="Kui pikalt minevikku bias/regressioonidele (kuud)")
    p.add_argument("--temp-module", type=str, default=None,
                   help="Ilma moodul (nt temp_forecast.py või paketinimi)")
    p.add_argument("--temp-csv", type=str, default=None,
                   help="Ilma CSV (sisaldab datetime+EE_avg või linnaveerud)")
    p.add_argument("--save-csv", action="store_true",
                   help="Salvesta CSV kausta output/")
    p.add_argument("--save-plot", action="store_true", help="Näita graafikut")
    p.set_defaults(segmented_bias=True)
    args = p.parse_args()

    main(mode=args.mode,
         segmented_bias=args.segmented_bias,
         months_hist=args.months,
         temp_module=args.temp_module,
         temp_csv=args.temp_csv,
         save_csv=args.save_csv,
         save_plot=args.save_plot)
