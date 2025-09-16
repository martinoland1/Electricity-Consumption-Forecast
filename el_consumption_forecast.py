# el_consumption_forecast.py
# 7 päeva tarbimise prognoos (UTC) segmenteeritud lineaarse regressiooniga
# + DIAGRAMM: vasakul teljel prognoositud tarbimine (tulbad), paremal teljel prognoositud temperatuur (EE_avg joon).
# + AASTAAJA BIAS: käivita bias_analysis.py ja korrigeeri prognoos vastavalt (kevad/suvi/sügis/talv).

import os
import re
import sys
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------ Regressioonivõrrandite hankimine ------------------------

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
        seg = _extract_segment(m.group(0))
        if seg:
            res["workdays"] = seg
    m = re.search(OFFDAYS_HDR + r"(.*?)(?:\n===|\Z)", out, flags=re.S)
    if m:
        seg = _extract_segment(m.group(0))
        if seg:
            res["offdays"] = seg
    return res

# ------------------------ EE_avg temperatuur temp_forecast.py-st ------------------------


def get_next7_avg_temp_utc_from_temp_forecast() -> pd.Series:
    """
    Impordib temp_forecast.py (summutab printid), võtab DataFrame'i 'result'
    ja tagastab veeru 'EE_avg' 7-päevase UTC päevaseeria (name='avg_temp_C').
    """
    import importlib.util
    import contextlib
    tf_path = Path(__file__).with_name("temp_forecast.py")
    if not tf_path.exists():
        raise FileNotFoundError("temp_forecast.py ei leitud samas kaustas.")

    spec = importlib.util.spec_from_file_location(
        "temp_forecast", str(tf_path))
    tf = importlib.util.module_from_spec(spec)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # käivitab temp_forecast.py ja loob 'result' & 'days_utc'
        spec.loader.exec_module(tf)

    if not hasattr(tf, "result"):
        raise RuntimeError(
            "temp_forecast.py ei ekspordi 'result' DataFrame'i.")
    df = tf.result.copy()

    if "EE_avg" not in df.columns:
        city_cols = [c for c in ["Tallinn", "Tartu", "Pärnu",
                                 "Narva", "Kuressaare"] if c in df.columns]
        if not city_cols:
            raise RuntimeError(
                "temp_forecast.result ei sisalda 'EE_avg' ega linnaveerge.")
        df["EE_avg"] = df[city_cols].mean(axis=1, skipna=True)

    s = df["EE_avg"]
    if hasattr(tf, "days_utc"):  # hoia täpne 7 päeva UTC indeks
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
    # UTC-nädalavahetus: laup(5) või pühap(6)
    is_weekend = dates_utc.dayofweek >= 5

    # Riigipühad: Eesti, UTC kuupäeva järgi
    try:
        import holidays
        years = list(range(dates_utc[0].year, dates_utc[-1].year + 1))
        ee = holidays.country_holidays("EE", years=years)
        is_holiday = pd.Index(
            [d.date() in ee for d in dates_utc], dtype="bool")
    except Exception:
        print("[info] 'holidays' teek puudub või tõrkus; eeldan is_holiday=False")
        is_holiday = pd.Index([False] * len(dates_utc), dtype="bool")

    seg = np.where(is_weekend | is_holiday, "OFFDAYS", "WORKDAYS")
    return pd.DataFrame({"is_weekend": is_weekend, "is_holiday": is_holiday, "segment": seg}, index=dates_utc)

# ------------------------ AASTAAJA BIAS: käivita bias_analysis.py ja loe faktorid ------------------------


def get_season_bias_map() -> dict:
    """
    Käivitab bias_analysis.py vaikselt ja tagastab dict'i: {'kevad': factor, 'suvi': ..., 'sügis': ..., 'talv': ...}
    Kui import ebaõnnestub, tagastab tühja dict (ehk faktor 1.0 kasutatakse).
    """
    import importlib.util
    import contextlib
    ba_path = Path(__file__).with_name("bias_analysis.py")
    if not ba_path.exists():
        print(
            "[info] bias_analysis.py puudub – jätkan ilma bias-korrigeerimiseta (factor=1.0).")
        return {}

    spec = importlib.util.spec_from_file_location(
        "bias_analysis", str(ba_path))
    ba = importlib.util.module_from_spec(spec)

    # Summuta bias_analysis.py print
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        spec.loader.exec_module(ba)

    # Eelistus: ametlik API
    if hasattr(ba, "get_season_bias"):
        try:
            _, mp = ba.get_season_bias()
            return dict(mp)
        except Exception:
            pass
    # Varuvariant: proovi otse season_bias_map
    if hasattr(ba, "season_bias_map"):
        try:
            return dict(ba.season_bias_map)
        except Exception:
            pass

    print("[info] Ei saanud bias_analysis.py-st bias-kaarti – jätkan factor=1.0.")
    return {}

# ------------------------ DIAGRAMM: tarbimine (tulbad) vs temperatuur (joon) ------------------------


def plot_dual_axis_bars(out_df: pd.DataFrame) -> None:
    """
    Tulbad: prognoositud tarbimine (WORKDAYS vs WEEKENDS/HOLIDAYS eri värviga, vasak Y).
    Joon:   prognoositud temperatuur EE_avg (teist värvi, paremal Y, pealpool).
    Temperatuuri telg näitab ka miinuseid (alampiir = min(0, min(temp))).
    """
    from matplotlib.patches import Patch

    df = out_df.copy()
    dates = pd.to_datetime(df["date_utc"])
    x = np.arange(len(dates))

    # --- Värvid ---
    COLOR_CONS_WORK = "tab:blue"    # tööpäevade tarbimine
    COLOR_CONS_OFF = "tab:red"     # nädalavahetus/riigipüha tarbimine
    COLOR_TEMP_LINE = "tab:orange"  # temperatuuri joon

    seg = df["segment"].to_numpy()
    bar_colors = [COLOR_CONS_OFF if s ==
                  "OFFDAYS" else COLOR_CONS_WORK for s in seg]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Tarbimine tulpadena (vasak Y) — kasutame KORRIGEERITUD prognoosi
    cons_vals = df["yhat_consumption"].to_numpy(dtype=float)
    bars = ax1.bar(x, cons_vals, color=bar_colors,
                   label="Forecast consumption", zorder=2)
    ax1.set_ylabel("Forecast consumption")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, axis="y", alpha=0.3)

    # Temperatuur joonena (parem Y) – eespool ja teist värvi
    ax2 = ax1.twinx()
    temp_vals = df["EE_avg_temp_C"].to_numpy(dtype=float)
    line2, = ax2.plot(
        x, temp_vals, marker="o", linestyle="--", linewidth=2,
        color=COLOR_TEMP_LINE, label="Forecast temperature (°C)", zorder=4
    )
    ax2.set_ylabel("Forecast temperature (°C)")
    ax2.set_ylim(bottom=min(0, np.nanmin(temp_vals)))  # näita ka miinuseid

    # X-telg: kuupäevad
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.strftime("%Y-%m-%d")
                        for d in dates], rotation=45, ha="right")
    ax1.set_xlabel("Date (UTC)")

    # Legend
    legend_handles = [
        Patch(color=COLOR_CONS_WORK, label="Consumption (workday)"),
        Patch(color=COLOR_CONS_OFF,  label="Consumption (weekend/holiday)"),
        line2
    ]
    ax1.legend(handles=legend_handles, loc="upper left")

    plt.tight_layout()
    # plt.savefig("el_consumption_forecast_dualaxis_bars.png", dpi=130)
    plt.show()

# ------------------------ Pea-protsess ------------------------


def main():
    # 1) Regressioonivõrrandid
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

    # 2) EE_avg 7 päeva UTC temp temp_forecast.py-st
    temp_s = get_next7_avg_temp_utc_from_temp_forecast()
    if temp_s.empty:
        print("ERROR: ei saanud EE_avg temperatuuri prognoose temp_forecast.py-st.")
        sys.exit(3)

    # 3) Päevaklassid (UTC)
    cls = classify_days_utc(temp_s.index)

    # 4) Aastaaja bias-kaart bias_analysis.py-st
    season_bias_map = get_season_bias_map()

    # 5) Valemi rakendamine (baas + aastaaja bias)
    x = temp_s.to_numpy(dtype=float)
    seg = cls["segment"].to_numpy()

    # Baasprognoos ilma biasita:
    yhat_base = np.where(
        seg == "OFFDAYS",
        o["intercept"] + o["slope"] * x,
        w["intercept"] + w["slope"] * x
    )

    # Aastaaja määramine: kevad(mar–may), suvi(jun–aug), sügis(sep–nov), talv(dec–feb)
    months = pd.DatetimeIndex(temp_s.index).month
    season = np.where(np.isin(months, [3, 4, 5]),  "kevad",
                      np.where(np.isin(months, [6, 7, 8]),  "suvi",
                               np.where(np.isin(months, [9, 10, 11]), "sügis", "talv")))

    # Bias-faktor igale reale (vaikimisi 1.0, kui kaardis puudub):
    bias_factors = np.array([season_bias_map.get(s, 1.0)
                            for s in season], dtype=float)

    # Korrigeeritud prognoos:
    yhat_adj = yhat_base * bias_factors

    # 6) Väljundtabel (kasutame graafikul korrigeeritud prognoosi)
    out = pd.DataFrame({
        "date_utc": temp_s.index.tz_convert("UTC").strftime("%Y-%m-%d"),
        "EE_avg_temp_C": np.round(x, 1),
        "segment": seg,
        "season": season,
        "bias_factor": np.round(bias_factors, 6),
        "yhat_base": np.round(yhat_base, 2),
        # korrigeeritud väärtus (kasutame diagrammil)
        "yhat_consumption": np.round(yhat_adj, 2)
    })

    print("\n7 päeva tarbimise prognoos (UTC, temp: EE_avg; sh aastaaja bias):")
    print(
        f"- WORKDAYS valem:     y = {w['intercept']:.6f} + ({w['slope']:.6f})*x")
    print(
        f"- OFFDAYS valem:      y = {o['intercept']:.6f} + ({o['slope']:.6f})*x")
    if season_bias_map:
        print(f"- Aastaaja bias-kaart: {season_bias_map}")
        print("  (Lõplik valem:  ŷ = (intercept + slope·x) × bias_factor(season))")
    else:
        print("- Aastaaja bias puudub (kasutan factor=1.0 kõigile).")
    print(out.to_string(index=False))

    # 7) Diagramm (tulbad + joon)
    plot_dual_axis_bars(out)


if __name__ == "__main__":
    main()
