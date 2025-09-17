# hourly_forecast.py — clean versioon (duplikaadivaba)
# Prognoosib järgmise 7 päeva TUNNI-põhise tarbimise:
# 1) Võtab EE_avg temp (UTC päevad) temp_forecast.py-st
# 2) Rakendab segmenteeritud regressiooni (WORKDAYS vs OFFDAYS) + aastaaja bias
# 3) Jaotab iga päeva 24 tunniks, kasutades weekday_profile.get_weekday_hour_share_matrix(last_n=6)
# NB: ei salvesta CSV-sid ega joonista graafikuid.

import os, re, sys, io
from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------ Reg-võrrandite võtmine (nagu su eelmises skriptis) ------------------------
WORKDAYS_HDR = r"=== WORKDAYS .*?Linear Regression Summary ==="
OFFDAYS_HDR  = r"=== WEEKENDS\s*&\s*HOLIDAYS .*?Linear Regression Summary ==="
SLOPE_LINE   = r"- Slope .*?:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
INTERCEPT_LINE = r"- Intercept.*?:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"

def _extract_segment(block: str):
    sm = re.search(SLOPE_LINE, block); im = re.search(INTERCEPT_LINE, block)
    if not sm or not im: return None
    return {"slope": float(sm.group(1)), "intercept": float(im.group(1))}

def get_segment_equations(path: Path) -> dict:
    import subprocess
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run([sys.executable, str(path)], cwd=str(path.parent),
                          env=env, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        print(f"[warn] regression script exit code {proc.returncode}; püüan ikkagi parsimist.")
    out = proc.stdout or ""
    res = {}
    m = re.search(WORKDAYS_HDR + r"(.*?)(?:\n===|\Z)", out, flags=re.S)
    if m:
        seg = _extract_segment(m.group(0))
        if seg: res["workdays"] = seg
    m = re.search(OFFDAYS_HDR + r"(.*?)(?:\n===|\Z)", out, flags=re.S)
    if m:
        seg = _extract_segment(m.group(0))
        if seg: res["offdays"] = seg
    return res

# ------------------------ Temp 7 päeva EE_avg temp_forecast.py-st ------------------------
def get_next7_avg_temp_utc_from_temp_forecast() -> pd.Series:
    import importlib.util, contextlib
    tf_path = Path(__file__).with_name("temp_forecast.py")
    if not tf_path.exists():
        raise FileNotFoundError("temp_forecast.py ei leitud samas kaustas.")
    spec = importlib.util.spec_from_file_location("temp_forecast", str(tf_path))
    tf = importlib.util.module_from_spec(spec)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        spec.loader.exec_module(tf)  # loob 'result' & 'days_utc'
    if not hasattr(tf, "result"):
        raise RuntimeError("temp_forecast.py ei ekspordi 'result' DataFrame'i.")
    df = tf.result.copy()
    if "EE_avg" not in df.columns:
        city_cols = [c for c in ["Tallinn","Tartu","Pärnu","Narva","Kuressaare"] if c in df.columns]
        if not city_cols:
            raise RuntimeError("temp_forecast.result ei sisalda 'EE_avg' ega linnaveerge.")
        df["EE_avg"] = df[city_cols].mean(axis=1, skipna=True)
    s = df["EE_avg"]
    if hasattr(tf, "days_utc"):
        s = s.reindex(tf.days_utc)
    s.name = "avg_temp_C"
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    else:
        s.index = s.index.tz_convert("UTC")
    s.index.name = "date_utc"
    return s

# ------------------------ Päevaklassid (UTC) ------------------------
def classify_days_utc(dates_utc: pd.DatetimeIndex) -> pd.DataFrame:
    is_weekend = dates_utc.dayofweek >= 5
    try:
        import holidays
        years = list(range(dates_utc[0].year, dates_utc[-1].year + 1))
        ee = holidays.country_holidays("EE", years=years)
        is_holiday = pd.Index([d.date() in ee for d in dates_utc], dtype="bool")
    except Exception:
        print("[info] 'holidays' teek puudub või tõrkus; eeldan is_holiday=False")
        is_holiday = pd.Index([False]*len(dates_utc), dtype="bool")
    seg = np.where(is_weekend | is_holiday, "OFFDAYS", "WORKDAYS")
    return pd.DataFrame({"is_weekend": is_weekend, "is_holiday": is_holiday, "segment": seg}, index=dates_utc)

# ------------------------ Aastaaja bias bias_analysis.py-st ------------------------
def get_season_bias_map() -> dict:
    import importlib.util, contextlib
    ba_path = Path(__file__).with_name("bias_analysis.py")
    if not ba_path.exists():
        print("[info] bias_analysis.py puudub – jätkan factor=1.0.")
        return {}
    spec = importlib.util.spec_from_file_location("bias_analysis", str(ba_path))
    ba = importlib.util.module_from_spec(spec)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        spec.loader.exec_module(ba)
    if hasattr(ba, "get_season_bias"):
        try:
            _, mp = ba.get_season_bias()
            return dict(mp)
        except Exception:
            pass
    if hasattr(ba, "season_bias_map"):
        try:
            return dict(ba.season_bias_map)
        except Exception:
            pass
    print("[info] Ei saanud bias-kaarti – factor=1.0.")
    return {}

# ------------------------ Weekday tunnijaotuse maatriks ------------------------
def get_share_matrix(last_n: int = 6) -> pd.DataFrame:
    import weekday_profile as wp
    M = wp.get_weekday_hour_share_matrix(last_n=last_n)  # 24x7, sum(col)=1
    # kindlustame, et indeks 0..23 ja veerunimed Monday..Sunday
    M = M.reindex(range(24))
    return M

# ------------------------ Pea-protsess ------------------------
def get_hourly_forecast(last_n: int = 6) -> pd.DataFrame:
    # === kogu praegune main() sisu SIINSE funktsiooni sisse ===
    # 1) regressiooni segmendid
    reg_path = Path(__file__).with_name("regression_analysis_split.py")
    if not reg_path.exists():
        raise FileNotFoundError("regression_analysis_split.py ei leitud.")
    eq = get_segment_equations(reg_path)
    if not {"workdays", "offdays"} <= set(eq.keys()):
        raise RuntimeError("Ei saanud kätte mõlema segmendi võrrandeid.")
    w, o = eq["workdays"], eq["offdays"]

    # 2) EE_avg 7 päeva temp
    temp_s = get_next7_avg_temp_utc_from_temp_forecast()
    if temp_s.empty:
        raise RuntimeError("EE_avg temp tühi.")

    # 3) segmendid + aastaaeg + bias
    cls = classify_days_utc(temp_s.index)
    months = temp_s.index.month
    season = np.where(np.isin(months, [3,4,5]),  "kevad",
              np.where(np.isin(months, [6,7,8]),  "suvi",
              np.where(np.isin(months, [9,10,11]), "sügis", "talv")))
    season_bias_map = get_season_bias_map()

    # 4) päevane prognoos (baas + bias)
    x = temp_s.to_numpy(float)
    seg = cls["segment"].to_numpy()
    yhat_base = np.where(seg=="OFFDAYS",
                         o["intercept"] + o["slope"]*x,
                         w["intercept"] + w["slope"]*x)
    bias_factors = np.array([season_bias_map.get(s, 1.0) for s in season], float)
    daily_yhat = yhat_base * bias_factors

    # 5) “päevatabel”
    daily_df = pd.DataFrame({
        "date_utc": temp_s.index.tz_convert("UTC"),
        "EE_avg_temp_C": np.round(x, 1),
        "segment": seg,
        "season": season,
        "bias_factor": np.round(bias_factors, 6),
        "daily_yhat": daily_yhat
    })
    daily_df["weekday"] = daily_df["date_utc"].dt.day_name()

    # 6) tunnijaotus
    M = get_share_matrix(last_n=last_n)  # index=hour_utc, columns=weekday
    share_long = (M.reset_index()
                    .rename(columns={M.reset_index().columns[0]: "hour_utc"})
                    .melt(id_vars="hour_utc", var_name="weekday", value_name="hourly_share"))
    all_hours = pd.DataFrame({"hour_utc": range(24)})
    weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    base_grid = all_hours.assign(key=1).merge(
        pd.DataFrame({"weekday": weekdays}).assign(key=1), on="key").drop(columns="key")
    share_long = base_grid.merge(share_long, how="left", on=["hour_utc","weekday"])
    share_long["hourly_share"] = share_long["hourly_share"].fillna(1.0/24.0)

    # 7) ristkorrutus → 7*24 rida
    out = (daily_df
           .merge(share_long, on="weekday", how="left")
           .sort_values(["date_utc","hour_utc"])
           .reset_index(drop=True))

    # 8) tunniprognoos + abiveerud
    out["hourly_yhat"] = out["daily_yhat"] * out["hourly_share"]
    out["date_str"] = out["date_utc"].dt.strftime("%Y-%m-%d")
    try:
        import holidays
        years = list({d.year for d in out["date_utc"]})
        ee = holidays.country_holidays("EE", years=years)
        out["is_holiday_local_EE"] = out["date_utc"].dt.tz_convert("Europe/Tallinn").dt.date.map(lambda d: d in ee)
    except Exception:
        out["is_holiday_local_EE"] = False

    return out  # <-- tagasta DataFrame

def main():
    out = get_hourly_forecast(last_n=6)

    # kontroll: sum(hourly) == daily
    chk = (out.groupby("date_str")["hourly_yhat"].sum()
           / out.groupby("date_str")["daily_yhat"].first()) - 1.0
    max_rel = float(np.nanmax(np.abs(chk.values))) if len(chk) else 0.0
    print(f"[kontroll] Päevasummad vs daily_yhat — max |rel_diff| ≈ {max_rel*100:.5f}%")

    # esimesed 48 rida
    print("\n=== Hourly forecast (first 48 rows) ===")
    cols = ["date_str","hour_utc","weekday","segment","season",
            "EE_avg_temp_C","daily_yhat","hourly_share","hourly_yhat","is_holiday_local_EE"]
    print(out[cols].head(48).to_string(index=False))

if __name__ == "__main__":
    main()
