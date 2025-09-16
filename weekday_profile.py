# weekday_profile.py
# Eesmärk: arvutada iga nädalapäeva (Monday..Sunday) keskmine tunnijaotus (UTC) viimase
# 'last_n' selle nädalapäeva esinemise põhjal, välistades pühad (Eesti riigipühad).
#
# Eksporditavad funktsioonid:
#   - get_weekday_hour_share_matrix(last_n=6) -> pd.DataFrame (index=hour_utc 0..23; columns=Monday..Sunday)
#   - get_weekday_days_used(last_n=6) -> pd.DataFrame (columns=['weekday','date_utc'])
#   - plot_weekday_profiles(share_matrix: pd.DataFrame) -> None

from typing import Tuple, List
import sys
import numpy as np
import pandas as pd

WEEKDAY_ORDER: List[str] = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
]

# --- Andmeallikas: tunnipõhine tarbimine (sum_cons_time, sum_el_hourly_value) ---
try:
    import el_consumption_weekday as ecw  # peab olema samas kaustas
except Exception as e:
    print(f"ERROR: ei suutnud importida el_consumption_weekday.py: {e}")
    sys.exit(1)


def _attach_ee_holidays_local(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lisab veeru 'is_holiday' Eesti riigipühade järgi.
    Oluline: pühade määramine tehakse kohaliku Eesti kalendripäeva (Europe/Tallinn) järgi.
    Ülejäänud arvutused (päev/tund) tehakse UTC järgi.
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

    # Veendu, et ajatempel on timezone-aware UTC
    ts = df["sum_cons_time"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts, utc=True, errors="coerce")
    else:
        # kui timezone-naive, eelda, et see on UTC ja lisa tz
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
    # lokaalne Eesti kuupäev pühade tuvastuseks
    local_dates = ts.dt.tz_convert("Europe/Tallinn").dt.date

    # vali aasta(h) vahemik
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
    Laeb ja vormindab sisendi:
      - arvutab uuesti 'is_holiday' Eesti kalendri järgi (kui võimalik)
      - välistab pühad
      - lisab UTC 'date_utc', 'hour_utc', 'weekday'
      - arvutab 'hourly_share_of_day'
    """
    df = ecw.sum_hourly_el_consumption.copy()
    if df.empty:
        raise RuntimeError("weekday_profile: sisendandmestik on tühi.")

    # Kindlusta datetime UTC-na
    if not pd.api.types.is_datetime64_any_dtype(df["sum_cons_time"]):
        df["sum_cons_time"] = pd.to_datetime(
            df["sum_cons_time"], utc=True, errors="coerce")
    else:
        if df["sum_cons_time"].dt.tz is None:
            df["sum_cons_time"] = df["sum_cons_time"].dt.tz_localize("UTC")

    # (Re)compute holidays using EE local calendar
    df = _attach_ee_holidays_local(df)

    # Välista pühad
    df = df[~df["is_holiday"].fillna(False)].copy()
    if df.empty:
        raise RuntimeError(
            "weekday_profile: pärast pühade välistamist puuduvad andmed.")

    # UTC kuupäev ja tund
    df["date_utc"] = df["sum_cons_time"].dt.date
    df["hour_utc"] = df["sum_cons_time"].dt.hour

    # Nädalapäev (inglise keeles)
    if "weekday" not in df.columns or df["weekday"].isna().all():
        df["weekday"] = df["sum_cons_time"].dt.day_name()  # Monday..Sunday

    # Päeva summa ja tunni osakaal
    daily_totals = df.groupby("date_utc")[
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
      - profile_df: (weekday, hour_utc, avg_hourly_share_of_day, n_days)
      - days_used: (weekday, date_utc) – millised kuupäevad arvesse läksid iga nädalapäeva korral
    """
    profiles = []
    days_used_rows = []

    for wd in WEEKDAY_ORDER:
        sub = df[df["weekday"] == wd]
        if sub.empty:
            continue

        # viimased 'last_n' selle nädalapäeva kuupäeva UTC järgi (pühad on juba välistatud)
        last_days = (
            sub[["date_utc"]].drop_duplicates().sort_values(
                "date_utc").tail(last_n)["date_utc"].tolist()
        )
        if not last_days:
            continue

        # salvesta päeva loetelu aruandluseks
        for d in last_days:
            days_used_rows.append({"weekday": wd, "date_utc": d})

        sub_n = sub[sub["date_utc"].isin(last_days)]
        if sub_n.empty:
            continue

        avg_prof = (
            sub_n.groupby("hour_utc")["hourly_share_of_day"]
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
        ["weekday", "hour_utc"]).reset_index(drop=True)

    days_used = pd.DataFrame(days_used_rows, columns=["weekday", "date_utc"])
    days_used = days_used.sort_values(
        ["weekday", "date_utc"]).reset_index(drop=True)
    return profile_df, days_used


def _profile_df_to_share_matrix(profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Muudab profile_df -> 24x7 maatriksiks ja normaliseerib iga veeru summa 1.0-ks.
    Kui mõni nädalapäev puudub, täidetakse ühtlase jaotusega (1/24).
    """
    mat = (
        profile_df.pivot(index="hour_utc", columns="weekday",
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

    mat.index.name = "hour_utc"
    mat.columns.name = "weekday"
    return mat


# --- Public API ---------------------------------------------------------------

def get_weekday_hour_share_matrix(last_n: int = 6) -> pd.DataFrame:
    """
    Tagastab DataFrame'i kujul 24x7 tunnijaotuse maatriksi:
      index: hour_utc (0..23)
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
    Tagastab DataFrame'i (weekday, date_utc) – millised 'last_n' kuupäeva
    iga nädalapäeva puhul arvesse läksid (pühad välistatud).
    """
    df_hourly = _prepare_hourly_df()
    _, days_used = _build_profiles_and_days_used(df_hourly, last_n=last_n)
    return days_used


def plot_weekday_profiles(share_matrix: pd.DataFrame, title: str = None) -> None:
    """
    Joonistab joondiagrammi: tunnid 0..23 X-teljel, iga nädalapäev eraldi joonena.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib ei ole saadaval: {e}")
        return

    if title is None:
        title = "Keskmine tunnijaotus (hourly_share_of_day) – viimased esinemised, UTC (pühad välistatud)"

    plt.figure(figsize=(10, 6))
    for wd in WEEKDAY_ORDER:
        if wd in share_matrix.columns:
            plt.plot(share_matrix.index, share_matrix[wd], label=wd)

    plt.title(title)
    plt.xlabel("Tund (UTC, 0–23)")
    plt.ylabel("Tunni osakaal päeva tarbimisest")
    plt.xticks(range(0, 24, 1))
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(title="Weekday", ncol=2)
    plt.tight_layout()
    # Soovi korral salvesta:
    # plt.savefig("weekday_avg_profiles_UTC_no_holidays.png", dpi=160)
    plt.show()


# --- CLI kasutus -------------------------------------------------------------

if __name__ == "__main__":
    try:
        M = get_weekday_hour_share_matrix(last_n=6)   # 24x7 maatriks
        DU = get_weekday_days_used(last_n=6)          # (weekday, date_utc)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(2)

    print("\n=== Pivot (tunnid ridades, nädalapäevad veergudes) ===")
    print(M.round(4).to_string())

    print("\n=== Millised kuupäevad läksid arvesse (iga nädalapäev, viimased esinemised; pühad välistatud) ===")
    du_print = (DU.assign(date_utc=DU["date_utc"].astype(str))
                  .sort_values(["weekday", "date_utc"]))
    print(du_print.to_string(index=False))

    plot_weekday_profiles(M)
