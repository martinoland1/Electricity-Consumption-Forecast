# 🧩 Cell 1 — Seadistus (teegid ja failiteed)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# >> Kohanda teed vastavalt oma arvutile <<
tarb_path = Path(r"C:\Users\User\Desktop\Electricity-Forecast-Git\Electricity-Consumption-Forecast\electricity_input_data\Tarb_aug24.xlsx")
temp_path = Path(r"C:\Users\User\Desktop\Electricity-Forecast-Git\Electricity-Consumption-Forecast\electricity_input_data\Temp_aug24.xlsx")

# 🧩 Cell 2 — Abifunktsioonid (veerunimed, arvuliseks, datetime)

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim + tühikud -> alakriips + väiketähed (turvaline veerunimede ühtlustus)."""
    df.columns = (df.columns
                  .str.strip()
                  .str.replace(r"\s+", "_", regex=True)
                  .str.lower())
    return df

def ensure_numeric(s: pd.Series) -> pd.Series:
    """Teisenda veerg arvuliseks, arvestab ka tühikuid ja koma->punkt vahetust."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s2 = (s.astype(str)
            .str.replace("\u00A0", "", regex=False)  # non-breaking space
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False))
    return pd.to_numeric(s2, errors="coerce")

def combine_date_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loo ühtne 'datetime' veerg.
    - 'kpv' on juba datetime (Exceli date), 'aeg' on time või 'HH:MM(:SS)' string.
    """
    if "kpv" not in df.columns or "aeg" not in df.columns:
        raise ValueError("Vajalikud veerud 'kpv' ja 'aeg' puuduvad.")
    if not pd.api.types.is_datetime64_any_dtype(df["kpv"]):
        raise TypeError("Veerg 'kpv' peab olema datetime64 dtype (Exceli kuupäev).")

    # teisenda 'aeg' pandas Timedelta-ks
    time_str = df["aeg"].astype(str).str.strip()          # nt '01:00:00'
    time_delta = pd.to_timedelta(time_str, errors="coerce")

    out = df.copy()
    out["datetime"] = out["kpv"].dt.normalize() + time_delta
    out = out.dropna(subset=["datetime"]).copy()          # eemalda vigased read
    return out

# 🧩 Cell 3 — Impordi mõlemad failid ja eeltöötlus

# Loe Excelid
tarb = pd.read_excel(tarb_path, engine="openpyxl")
temp = pd.read_excel(temp_path, engine="openpyxl")

# Ühtlusta veerunimed
tarb = clean_columns(tarb)   # eeldus: kpv, aeg, tarbimine
temp = clean_columns(temp)   # eeldus: kpv, aeg, temperatuur (või 'temp')

# Kui temperatuur on veerus 'temp', nimeta ümber
if "temp" in temp.columns and "temperatuur" not in temp.columns:
    temp = temp.rename(columns={"temp": "temperatuur"})

# Loo ühtne ajatempel
tarb = combine_date_time(tarb)
temp = combine_date_time(temp)

# Teisenda mõõdikud arvuliseks
tarb["tarbimine"]   = ensure_numeric(tarb["sum_el_hourly_value"])
temp["temperatuur"] = ensure_numeric(temp["hour_temp_value"])

# Eemalda read, kus mõõdik puudub
tarb = tarb.dropna(subset=["tarbimine"]).copy()
temp = temp.dropna(subset=["temperatuur"]).copy()

print("Tarb vahemik:", tarb["datetime"].min(), "→", tarb["datetime"].max(), "| read:", len(tarb))
print("Temp vahemik:", temp["datetime"].min(), "→", temp["datetime"].max(), "| read:", len(temp))

# 🧩 Cell 5 — Tunnipõhine koond (tarbimine=sum, temperatuur=mean)

tarb_h = (
    tarb.resample("H", on="datetime")["tarbimine"]
        .sum()
        .reset_index()
)

temp_h = (
    temp.resample("H", on="datetime")["temperatuur"]
        .mean()
        .reset_index()
)

tarb_h.head(), temp_h.head()

# 🧩 Cell 6 — Kuupõhised kokkuvõtted + katvus (coverage)

def month_start(ts: pd.Series) -> pd.Series:
    """Kuu alguse Timestamp (MS)."""
    return ts.dt.to_period("M").dt.to_timestamp()

# Lisa kuu veerg
tarb_h["kuu"] = month_start(tarb_h["datetime"])
temp_h["kuu"] = month_start(temp_h["datetime"])

# Kuu koond: tarbimine (sum) + mitu tunnikirjet oli (coverage jaoks)
tarb_m = (
    tarb_h.groupby("kuu", as_index=False)
          .agg(kuu_sum_tarb=("tarbimine", "sum"),
               hours_tarb_present=("tarbimine", "count"))
)

# Kuu koond: temperatuur (mean) + mitu tunnikirjet oli
temp_m = (
    temp_h.groupby("kuu", as_index=False)
          .agg(kuu_keskmine_temp=("temperatuur", "mean"),
               hours_temp_present=("temperatuur", "count"))
)

# Arvuta kuu kogutundide arv (min..max tunnivahemiku põhjal)
all_min = min(tarb_h["datetime"].min(), temp_h["datetime"].min())
all_max = max(tarb_h["datetime"].max(), temp_h["datetime"].max())
all_hours = pd.DataFrame({"datetime": pd.date_range(all_min, all_max, freq="H")})
all_hours["kuu"] = month_start(all_hours["datetime"])
hours_total = (all_hours.groupby("kuu", as_index=False)["datetime"]
                        .count()
                        .rename(columns={"datetime":"hours_total"}))

# Lõplik kuutabel
monthly = (tarb_m.merge(temp_m, on="kuu", how="outer")
                 .merge(hours_total, on="kuu", how="left")
                 .sort_values("kuu")
                 .reset_index(drop=True))

# Coverage suhtarvud
monthly["tarb_coverage"] = (monthly["hours_tarb_present"] / monthly["hours_total"]).round(3)
monthly["temp_coverage"] = (monthly["hours_temp_present"] / monthly["hours_total"]).round(3)

# Esituslik ümardus
monthly["kuu_keskmine_temp"] = monthly["kuu_keskmine_temp"].round(2)
monthly["kuu_sum_tarb"] = monthly["kuu_sum_tarb"].round(0)
print(monthly)