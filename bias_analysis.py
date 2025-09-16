# bias_analysis.py
# Arvutab temp-põhise lihtmudei bias'i (kallaku) kuude kaupa ja kalendrikuu keskmised.
# Nõuded: pandas, numpy, matplotlib

import os, sys, io, contextlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Seaded ---
SUPPRESS_MODULE_OUTPUT = True
OUTDIR = Path("output")
OUTDIR.mkdir(parents=True, exist_ok=True)

# --- Abifunktsioon: impordi moodul vaikselt (kui seal on print'e) ---
def import_module_silent(name: str):
    if SUPPRESS_MODULE_OUTPUT:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            module = __import__(name)
        return module
    return __import__(name)

# --- Lisa käesolev kaust pythoni teele ---
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# --- Impordi teie projektimoodulid ---
ec = import_module_silent("el_consumption")  # ootab: sum_daily_el_consumption
tp = import_module_silent("temp")            # ootab: avg_day_temp (või avg_day_tempo)

# --- Võta datafreimid välja ja kontrolli veerud ---
df_cons = getattr(ec, "sum_daily_el_consumption", None)
if df_cons is None:
    raise RuntimeError("el_consumption.py peab defineerima 'sum_daily_el_consumption' DataFrame'i.")

df_temp = getattr(tp, "avg_day_temp", None)
if df_temp is None:
    df_temp = getattr(tp, "avg_day_tempo", None)
if df_temp is None:
    raise RuntimeError("avg_day_temp (or avg_day_tempo) not found in temp.py")

need_cons = {"sum_cons_date", "sum_el_daily_value"}
need_temp = {"avg_day_temp_date", "hour_day_value"}
if not need_cons.issubset(df_cons.columns):
    raise RuntimeError(f"Tarvis veerge {need_cons}, leidsin {set(df_cons.columns)}")
if not need_temp.issubset(df_temp.columns):
    raise RuntimeError(f"Tarvis veerge {need_temp}, leidsin {set(df_temp.columns)}")

# --- Kuupäevad kuupäevaks, duplikaadid kokku ---
c = df_cons.copy()
t = df_temp.copy()
c["sum_cons_date"] = pd.to_datetime(c["sum_cons_date"], errors="coerce").dt.date
t["avg_day_temp_date"] = pd.to_datetime(t["avg_day_temp_date"], errors="coerce").dt.date
c = c.groupby("sum_cons_date", as_index=False)["sum_el_daily_value"].sum()
t = t.groupby("avg_day_temp_date", as_index=False)["hour_day_value"].mean()

# --- Join: ainult ühisosa päevad ---
df = pd.merge(
    c, t,
    left_on="sum_cons_date",
    right_on="avg_day_temp_date",
    how="inner"
)[["sum_cons_date", "sum_el_daily_value", "hour_day_value"]].dropna()

df["sum_cons_date"] = pd.to_datetime(df["sum_cons_date"])
df = df.sort_values("sum_cons_date").reset_index(drop=True)

# --- Lihtne lineaarne mudel: y = a + b * T (fit kogu vahemiku peal) ---
x = df["hour_day_value"].to_numpy(float)      # temp (°C)
y = df["sum_el_daily_value"].to_numpy(float)  # daily consumption

# --- Lihtne lineaarne mudel: y = a + b * T (kasutame etteantud väärtuseid) ---
a = 24939.181233   # intercept
b = -396.694905    # slope

# Päevaprognoos ja jäägid
df["y_hat"] = a + b * df["hour_day_value"]
df["resid"] = df["sum_el_daily_value"] - df["y_hat"]
df["abs_pct_error"] = (df["sum_el_daily_value"] - df["y_hat"]) / df["sum_el_daily_value"]  # võib olla neg
# kuupäeva abi
df["month"] = df["sum_cons_date"].dt.to_period("M").dt.to_timestamp()  # kuu algus
df["month_num"] = df["sum_cons_date"].dt.month
df["year"] = df["sum_cons_date"].dt.year

# --- Kuu kokkuvõte: actual vs predicted + bias-faktor ---
monthly = (df.groupby("month", as_index=False)
             .agg(actual=("sum_el_daily_value", "sum"),
                  predicted=("y_hat", "sum")))
monthly["abs_error"] = monthly["actual"] - monthly["predicted"]
monthly["pct_error"] = monthly["abs_error"] / monthly["actual"]              # + tähendab üle/alla?
monthly["bias_factor"] = np.where(monthly["predicted"] > 0,
                                  monthly["actual"] / monthly["predicted"],
                                  np.nan)

# --- Kalendrikuu (Jan..Dec) keskmised bias'id (mitme aasta pealt) ---
by_moy = (monthly.assign(month_num=monthly["month"].dt.month)
                  .groupby("month_num", as_index=False)
                  .agg(avg_bias_factor=("bias_factor", "mean"),
                       avg_pct_error=("pct_error", "mean"),
                       months=("bias_factor", "count")))

# --- AASTAAJAD + KORRIGEERIMINE (PÄRAST by_moy ARVUTUST) ---

# 1) Aastaajad (talv/kevad/suvi/sügis) keskmine bias
season_map = {
    12: "talv", 1: "talv", 2: "talv",
     3: "kevad", 4: "kevad", 5: "kevad",
     6: "suvi",  7: "suvi",  8: "suvi",
     9: "sügis", 10: "sügis", 11: "sügis",
}
by_moy["season"] = by_moy["month_num"].map(season_map)
season_bias = (by_moy
               .groupby("season", as_index=False)
               .agg(avg_bias_factor=("avg_bias_factor", "mean"),
                    months=("months", "sum")))

# 2) Rakenda korrigeerimine: kuupõhine ja aastaajapõhine
month_bias = by_moy.set_index("month_num")["avg_bias_factor"].to_dict()
monthly["pred_month_corr"] = monthly.apply(
    lambda r: r["predicted"] * month_bias[r["month"].month],
    axis=1
)

season_bias_map = season_bias.set_index("season")["avg_bias_factor"].to_dict()
monthly["season"] = monthly["month"].dt.month.map(season_map)
monthly["pred_season_corr"] = monthly.apply(
    lambda r: r["predicted"] * season_bias_map[r["season"]],
    axis=1
)

# 3) Täpsusnäitajad
def mape(actual, pred):
    return float(np.mean(np.abs((actual - pred) / actual)))

mape_base   = mape(monthly["actual"], monthly["predicted"])
mape_month  = mape(monthly["actual"], monthly["pred_month_corr"])
mape_season = mape(monthly["actual"], monthly["pred_season_corr"])

print("\n== Täpsuse võrdlus ==")
print(f"Base (temp-only)   MAPE: {mape_base*100:.2f}%")
print(f"Kuupõhine korrigeer MAPE: {mape_month*100:.2f}%")
print(f"Aastaajapõhine     MAPE: {mape_season*100:.2f}%")

# 4) Salvesta täiendavad failid
(OUTDIR / "bias_season.csv").write_text(
    season_bias.to_csv(index=False), encoding="utf-8"
)

monthly_out = monthly[["month","season","actual","predicted",
                       "pred_month_corr","pred_season_corr",
                       "abs_error","pct_error","bias_factor"]]
monthly_out.to_csv(OUTDIR / "bias_monthly_with_corrections.csv", index=False)

# 5) Võrdlusgraafik
fig, ax = plt.subplots(figsize=(11,5), dpi=130)
ax.plot(monthly["month"], monthly["actual"], marker="o", label="Tegelik")
ax.plot(monthly["month"], monthly["predicted"], marker="s", label="Mudel (ilma korrekts.)")
ax.plot(monthly["month"], monthly["pred_month_corr"], marker="^", label="Mudel + kuupõhine bias")
ax.plot(monthly["month"], monthly["pred_season_corr"], marker="D", label="Mudel + aastaajapõhine bias")
ax.set_title("Tegelik vs prognoosid (korrigeerimata ja korrigeeritud)")
ax.set_xlabel("Kuu"); ax.set_ylabel("Tarbimine (ühikud)")
ax.grid(True, linestyle="--", alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "bias_actual_vs_all_predictions.png", dpi=130)
plt.show()


# --- Üldine (kogu periood) bias ---
total_actual = monthly["actual"].sum()
total_pred   = monthly["predicted"].sum()
overall_bias_factor = float(total_actual / total_pred) if total_pred > 0 else np.nan
overall_mape = float(np.mean(np.abs(monthly["pct_error"])))  # keskmine absoluuthälve protsentides

# --- Salvesta CSV-d ---
monthly_path = OUTDIR / "bias_monthly.csv"
moy_path     = OUTDIR / "bias_month_of_year.csv"
summary_path = OUTDIR / "bias_summary.txt"

monthly.to_csv(monthly_path, index=False)
by_moy.to_csv(moy_path, index=False)

with open(summary_path, "w", encoding="utf-8") as f:
    f.write(f"Mudel: y = {a:.3f} + {b:.3f} * T\n")
    f.write(f"Periood: {df['sum_cons_date'].min().date()} … {df['sum_cons_date'].max().date()}\n")
    f.write(f"Üldine bias_factor (actual/pred): {overall_bias_factor:.4f}\n")
    f.write(f"Keskmine |pct_error| (MAPE):      {overall_mape*100:.2f}%\n")
    f.write("\nKalendrikuu keskmised (Jan..Dec):\n")
    f.write(by_moy.to_string(index=False))

print(f"Kirjutasin: {monthly_path}")
print(f"Kirjutasin: {moy_path}")
print(f"Kirjutasin: {summary_path}")
print(f"Üldine bias_factor = {overall_bias_factor:.4f}  |  MAPE ≈ {overall_mape*100:.2f}%")

# --- Lihtne graafik: kuu % viga ajas ---
fig, ax = plt.subplots(figsize=(10,5), dpi=130)
ax.plot(monthly["month"], monthly["pct_error"]*100.0, marker="o")
ax.axhline(0, color="k", lw=1)
ax.set_title("Kuu prognoosi % viga (temp-ainus mudel)")
ax.set_xlabel("Kuu")
ax.set_ylabel("% viga  ( (actual - predicted) / actual ) × 100")
ax.grid(True, linestyle="--", alpha=0.3)
fig.tight_layout()
plt.savefig(OUTDIR / "bias_monthly_plot.png", dpi=130)
plt.show()

# --- Näide: kuidas rakendada bias'i prognoosile ---
# Kui teed tulevikuprogo (ainult temperatuurist) ja tahad kalendrikuu kaupa korrigeerida:
# 1) leia vastava kuu number m (1..12)
# 2) korrigeeritud_prognoos = prognoos_temp_pohine * by_moy.loc[by_moy.month_num==m, 'avg_bias_factor'].iloc[0]
