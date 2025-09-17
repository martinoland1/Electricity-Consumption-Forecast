# weekday_profile.py
# Eesmärk: arvutada iga nädalapäeva (Monday..Sunday) keskmine tunnijaotus **Eesti aja (Europe/Tallinn)** järgi,
# kasutades iga nädalapäeva viimaseid 'last_n' esinemisi ja välistades Eesti riigipühad (kohaliku kalendripäeva järgi).
#
# Eksporditavad funktsioonid:
#   - get_weekday_hour_share_matrix(last_n=6) -> pd.DataFrame
#       index = hour_local (0..23, Eesti aeg), columns = Monday..Sunday,
#       values = avg_hourly_share_of_day, kus iga veeru summa == 1.0
#   - get_weekday_days_used(last_n=6) -> pd.DataFrame (columns=['weekday','date_local'])
#   - plot_weekday_profiles(share_matrix: pd.DataFrame) -> None
#
# Märkused:
#   - Sisendiks eeldame DataFrame'i ecw.sum_hourly_el_consumption veergudega:
#       ['sum_cons_time', 'sum_el_hourly_value']
#     kus 'sum_cons_time' võib olla timezone-naive, UTC või mõni muu tz — skript konverteerib selle Europe/Tallinn ajaks.
#   - Kõik päevade ja tundide arvutused (date_local, hour_local, weekday, päevade summad) tehakse Eesti aja järgi.

from typing import Tuple, List
import sys
import numpy as np
import pandas as pd

WEEKDAY_ORDER: List[str] = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
]

# --- Andmeallikas: tunnipõhine tarbimine (sum_cons_time, sum_el_hourly_value) ---
try:
    # peab olema samas kaustas ja sisaldama sum_hourly_el_consumption
    import el_consumption_weekday as ecw
except Exception as e:
    print(f"ERROR: ei suutnud importida el_consumption_weekday.py: {e}")
    sys.exit(1)


def _attach_ee_holidays_local(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lisab veeru 'is_holiday' Eesti riigipühade järgi.
    Pühade määramine tehakse **Eesti kohaliku kuupäeva** (Europe/Tallinn) järgi.
    """
    try:
        import holidays
    except Exception as e:
        print(
            f"[warn] 'holidays' teek puudub või ei õnnestu importida ({e}); eeldan, et pühi pole.")
        df["is_holiday"] = False
        return df

    if df.empty:
        df["is_holiday"] = False
        return df

    ts = df["sum_cons_time"]
    # Kindlusta timezone ja teisendus Tallinnaks
    if not pd.api.types.is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts, errors="coerce",
                            utc=True).dt.tz_convert("Europe/Tallinn")
    else:
        if ts.dt.tz is None:
            # Kui puudub tz, eelda UTC ja konverdi Tallinnaks
            ts = ts.dt.tz_localize("UTC").dt.tz_convert("Europe/Tallinn")
        else:
            ts = ts.dt.tz_convert("Europe/Tallinn")

    local_dates = ts.dt.date
    years = range(local_dates.min().year, local_dates.max().year + 1)

    try:
        ee_holidays = holidays.country_holidays("EE", years=years)
        df["is_holiday"] = pd.Series(local_dates).map(
            lambda d: d in ee_holidays).values
    except Exception as e:
        print(
            f"[warn] pühade arvutus ebaõnnestus ({e}); eeldan, et pühi pole.")
        df["is_holiday"] = False

    return df


def _prepare_hourly_df() -> pd.DataFrame:
    """
    Laeb ja vormindab sisendi **Eesti aja järgi**:
      - teisendab ajatemplid Europe/Tallinn vööndisse
      - arvutab uuesti 'is_holiday' Eesti kalendri järgi
      - välistab pühad
      - lisab 'date_local', 'hour_local', 'weekday'
      - arvutab 'hourly_share_of_day' (tunni osakaal päeva summast)
    """
    df = ecw.sum_hourly_el_consumption.copy()
    if df.empty:
        raise RuntimeError("weekday_profile: sisendandmestik on tühi.")

    # Kindlusta, et sum_cons_time on datetime ja teisenda Tallinnaks
    if not pd.api.types.is_datetime64_any_dtype(df["sum_cons_time"]):
        df["sum_cons_time"] = pd.to_datetime(
            df["sum_cons_time"], errors="coerce", utc=True)
    else:
        if df["sum_cons_time"].dt.tz is None:
            df["sum_cons_time"] = df["sum_cons_time"].dt.tz_localize("UTC")
    df["sum_cons_time"] = df["sum_cons_time"].dt.tz_convert("Europe/Tallinn")

    # Märgi pühad Eesti kohaliku kalendri järgi
    df = _attach_ee_holidays_local(df)

    # Välista pühad
    df = df[~df["is_holiday"].fillna(False)].copy()
    if df.empty:
        raise RuntimeError(
            "weekday_profile: pärast pühade välistamist puuduvad andmed.")

    # Kohalik kuupäev, tund, nädalapäev
    df["date_local"] = df["sum_cons_time"].dt.date
    df["hour_local"] = df["sum_cons_time"].dt.hour
    if "weekday" not in df.columns or df["weekday"].isna().all():
        df["weekday"] = df["sum_cons_time"].dt.day_name()  # Monday..Sunday

    # Päeva summa ja tunni osakaal (Eesti aja järgi)
    daily_totals = df.groupby("date_local")[
        "sum_el_hourly_value"].transform("sum")
    df["hourly_share_of_day"] = df["sum_el_hourly_value"] / daily_totals

    # Eemalda read, kus osakaal ei ole arvutatav
    df = df.dropna(subset=["hourly_share_of_day"])
    if df.empty:
        raise RuntimeError(
            "weekday_profile: pärast puhastust puuduvad andmed.")
    return df


def _build_profiles_and_days_used(df: pd.DataFrame, last_n: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Arvutab:
      - profile_df: (weekday, hour_local, avg_hourly_share_of_day, n_days)
      - days_used: (weekday, date_local) – millised kuupäevad arvesse läksid iga nädalapäeva korral
    """
    profiles = []
    days_used_rows = []

    for wd in WEEKDAY_ORDER:
        sub = df[df["weekday"] == wd]
        if sub.empty:
            continue

        # viimased 'last_n' selle nädalapäeva **kohaliku** kuupäeva järgi (pühad on juba välistatud)
        last_days = (
            sub[["date_local"]].drop_duplicates().sort_values(
                "date_local").tail(last_n)["date_local"].tolist()
        )
        if not last_days:
            continue

        # salvesta päeva loetelu aruandluseks
        for d in last_days:
            days_used_rows.append({"weekday": wd, "date_local": d})

        sub_n = sub[sub["date_local"].isin(last_days)]
        if sub_n.empty:
            continue

        avg_prof = (
            sub_n.groupby("hour_local")["hourly_share_of_day"]
            .mean()
            .rename("avg_hourly_share_of_day")
            .reset_index()
        )
        avg_prof["weekday"] = wd
        avg_prof["n_days"] = len(set(last_days))
        profiles.append(avg_prof)

    if not profiles:
        raise RuntimeError(
            "weekday_profile: profiile ei õnnestunud koostada (andmeid napib).")

    profile_df = pd.concat(profiles, ignore_index=True)
    profile_df = profile_df.sort_values(
        ["weekday", "hour_local"]).reset_index(drop=True)

    days_used = pd.DataFrame(days_used_rows, columns=["weekday", "date_local"])
    days_used = days_used.sort_values(
        ["weekday", "date_local"]).reset_index(drop=True)
    return profile_df, days_used


def _profile_df_to_share_matrix(profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Muudab profile_df -> 24x7 maatriksiks Eesti aja järgi ja normaliseerib iga veeru summa 1.0-ks.
    Kui mõni nädalapäev puudub, täidetakse ühtlase jaotusega (1/24).
    """
    mat = (
        profile_df.pivot(index="hour_local", columns="weekday",
                         values="avg_hourly_share_of_day")
        .reindex(range(24))
    )

    # veerud samas järjekorras + puuduolevad täita
    for wd in WEEKDAY_ORDER:
        if wd not in mat.columns:
            mat[wd] = np.nan
    mat = mat[WEEKDAY_ORDER].astype(float)

    # normaliseeri veerud (sum=1), kui tühi -> ühtlane jaotus
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


# --- Public API ---------------------------------------------------------------

def get_weekday_hour_share_matrix(last_n: int = 6) -> pd.DataFrame:
    """
    Tagastab DataFrame'i kujul 24x7 tunnijaotuse maatriksi **Eesti aja järgi**:
      index: hour_local (0..23)
      columns: ['Monday', 'Tuesday', ..., 'Sunday']
      values: avg_hourly_share_of_day (iga veeru summa == 1.0)
    Arvestab iga nädalapäeva viimaseid 'last_n' esinemist (vaikimisi 6),
    kusjuures Eesti pühad on välistatud (määritud kohaliku EE kuupäeva järgi).
    """
    df_hourly = _prepare_hourly_df()
    profile_df, _ = _build_profiles_and_days_used(df_hourly, last_n=last_n)
    return _profile_df_to_share_matrix(profile_df)


def get_weekday_days_used(last_n: int = 6) -> pd.DataFrame:
    """
    Tagastab DataFrame'i (weekday, date_local) – millised 'last_n' kuupäeva
    iga nädalapäeva puhul arvesse läksid (pühad välistatud), **Eesti aja järgi**.
    """
    df_hourly = _prepare_hourly_df()
    _, days_used = _build_profiles_and_days_used(df_hourly, last_n=last_n)
    return days_used


def plot_weekday_profiles(share_matrix: pd.DataFrame, title: str = None) -> None:
    """
    Joonistab joondiagrammi: tunnid 0..23 X-teljel, iga nädalapäev eraldi joonena (Eesti aeg).
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib ei ole saadaval: {e}")
        return

    if title is None:
        title = "Keskmine tunnijaotus (hourly_share_of_day) – viimased esinemised, Eesti aeg (pühad välistatud)"

    plt.figure(figsize=(10, 6))
    for wd in WEEKDAY_ORDER:
        if wd in share_matrix.columns:
            plt.plot(share_matrix.index, share_matrix[wd], label=wd)

    plt.title(title)
    plt.xlabel("Tund (Eesti aeg, 0–23)")
    plt.ylabel("Tunni osakaal päeva tarbimisest")
    plt.xticks(range(0, 24, 1))
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(title="Weekday", ncol=2)
    plt.tight_layout()
    # Soovi korral salvesta:
    # plt.savefig("weekday_avg_profiles_EE_no_holidays.png", dpi=160)
    plt.show()


# --- CLI kasutus -------------------------------------------------------------

if __name__ == "__main__":
    try:
        M = get_weekday_hour_share_matrix(
            last_n=6)   # 24x7 maatriks (Eesti aeg)
        DU = get_weekday_days_used(last_n=6)          # (weekday, date_local)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(2)

    print("\n=== Pivot (tunnid ridades, nädalapäevad veergudes) – Eesti aeg ===")
    print(M.round(4).to_string())

    print("\n=== Millised kuupäevad läksid arvesse (iga nädalapäev, viimased esinemised; pühad välistatud) – Eesti aeg ===")
    du_print = (DU.assign(date_local=DU["date_local"].astype(str))
                  .sort_values(["weekday", "date_local"]))
    print(du_print.to_string(index=False))
