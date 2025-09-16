# el_consumption_forecast.py
# 7 päeva tarbimise prognoos (UTC päevad) kasutades segmenteeritud lineaarset regressiooni:
# - WORKDAYS (Mon–Fri, non-holiday)
# - WEEKENDS & HOLIDAYS
#
# Sõltuvused: pandas, meteostat, holidays
#   pip install pandas meteostat holidays
#
# Eeldus: samas kaustas on regression_analysis_split.py (skript, mis prindib regressiooni kokkuvõtted)

import os
import re
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from meteostat import Point, Hourly

# ------------- Seaded -------------
CITY = "Tallinn"
POINT_TALLINN = Point(59.4370, 24.7536)

# ------------- Abiks: hangime regressioonivõrrandid käivitades regression_analysis_split.py -------------
WORKDAYS_HDR = r"=== WORKDAYS .*?Linear Regression Summary ==="
OFFDAYS_HDR = r"=== WEEKENDS\s*&\s*HOLIDAYS .*?Linear Regression Summary ==="
SLOPE_LINE = r"- Slope .*?:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
INTERCEPT_LINE = r"- Intercept.*?:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"


def _extract_segment(block: str):
    sm = re.search(SLOPE_LINE, block)
    im = re.search(INTERCEPT_LINE, block)
    if not sm or not im:
        return None
    return {"slope": float(sm.group(1)), "intercept": float(im.group(1))}


def get_segment_equations(path: Path) -> dict:
    import subprocess
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(path.parent),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        print(
            f"[warn] regression script exit code {proc.returncode}; püüan ikkagi parsimist.")
    out = proc.stdout or ""

    res = {}
    m = re.search(WORKDAYS_HDR + r"(.*?)(?:\n===|\Z)", out, flags=re.S)
    if m:
        b = m.group(0)
        seg = _extract_segment(b)
        if seg:
            res["workdays"] = seg
    m = re.search(OFFDAYS_HDR + r"(.*?)(?:\n===|\Z)", out, flags=re.S)
    if m:
        b = m.group(0)
        seg = _extract_segment(b)
        if seg:
            res["offdays"] = seg
    return res

# ------------- Meteostat: 7 päeva UTC päevakeskmine temp Tallinna kohta -------------


def get_next7_avg_temp_utc(pt: Point) -> pd.Series:
    now_utc = pd.Timestamp.now(tz="UTC")
    # alusta järgmisest UTC keskööst
    start_utc = (now_utc.floor("D") + pd.Timedelta(days=1))
    end_utc = start_utc + pd.Timedelta(days=7)

    # Meteostat vajab naive UTC datetimes:
    start_dt = start_utc.to_pydatetime().replace(tzinfo=None)
    end_dt = end_utc.to_pydatetime().replace(tzinfo=None)

    df = Hourly(pt, start_dt, end_dt, model=True).fetch()
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    if "time" in df.columns:
        df = df.set_index("time")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    s = df["temp"].resample("D").mean()  # °C
    s = s.reindex(pd.date_range(start=start_utc,
                  periods=7, freq="D", tz="UTC"))
    s.index.name = "date_utc"
    s.name = "avg_temp_C"
    return s

# ------------- Nädalavahetus/püha (UTC) -------------


def classify_days_utc(dates_utc: pd.DatetimeIndex) -> pd.DataFrame:
    # UTC-nädalavahetus: laup(5) või pühap(6)
    is_weekend = dates_utc.dayofweek >= 5

    # Riigipühad: kasutame 'holidays' (EE). Kasutame UTC kuupäeva (yyyy-mm-dd).
    try:
        import holidays
        years = list(range(dates_utc[0].year, dates_utc[-1].year + 1))
        ee = holidays.country_holidays("EE", years=years)
        # mapime kuupäeva (date) järgi
        is_holiday = pd.Index(
            [d.date() in ee for d in dates_utc], dtype="bool")
    except Exception:
        print("[info] 'holidays' teek puudub või tõrkus; eeldan is_holiday=False")
        is_holiday = pd.Index([False]*len(dates_utc), dtype="bool")

    seg = np.where(is_weekend | is_holiday, "OFFDAYS", "WORKDAYS")
    return pd.DataFrame({"is_weekend": is_weekend, "is_holiday": is_holiday, "segment": seg}, index=dates_utc)

# ------------- Pea-prognoos -------------


def main():
    # 1) Hangi regressioonivõrrandid
    reg_path = Path(__file__).with_name("regression_analysis_split.py")
    if not reg_path.exists():
        print("ERROR: regression_analysis_split.py ei leitud samas kaustas.")
        sys.exit(1)

    eq = get_segment_equations(reg_path)
    if not {"workdays", "offdays"} <= set(eq.keys()):
        print("ERROR: ei saanud kätte mõlema segmendi võrrandeid. Kontrolli regressiooniskripti väljundit.")
        sys.exit(2)

    w = eq["workdays"]  # {'slope': ..., 'intercept': ...}
    o = eq["offdays"]

    # 2) Tallinna 7-p päeva UTC päevakeskmine temperatuur
    temp_s = get_next7_avg_temp_utc(POINT_TALLINN)
    if temp_s.empty:
        print("ERROR: ei saanud Tallinna temperatuuri prognoose.")
        sys.exit(3)

    # 3) Päevaklassid (UTC)
    cls = classify_days_utc(temp_s.index)

    # 4) Valemi rakendamine
    x = temp_s.to_numpy(dtype=float)
    seg = cls["segment"].to_numpy()

    yhat = np.where(
        seg == "OFFDAYS",
        o["intercept"] + o["slope"] * x,
        w["intercept"] + w["slope"] * x
    )

    # 5) Väljundtabel
    out = pd.DataFrame({
        "date_utc": temp_s.index.tz_convert("UTC").strftime("%Y-%m-%d"),
        "avg_temp_C": np.round(x, 1),
        "segment": seg,
        "yhat_consumption": np.round(yhat, 2)
    })

    # Ilus print
    print("\n7 päeva tarbimise prognoos (UTC päevad, linn: Tallinn):")
    print(
        f"- WORKDAYS valem:     y = {w['intercept']:.6f} + ({w['slope']:.6f})*x")
    print(
        f"- OFFDAYS valem:      y = {o['intercept']:.6f} + ({o['slope']:.6f})*x")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
