"""
Eksport: nädalapäeva tunniprofiilid CSV-des
- PIVOT: tunnid ridades (0..23), nädalapäevad veergudes (Monday..Sunday), väärtus = avg_hourly_share_of_day
- DAYS_USED: millised kuupäevad (UTC) läksid arvesse iga nädalapäeva jaoks
- SOURCE_ROWS: allikaread koos kuupäeva, nädalapäeva, püha lipu, tunni tarbimise ja tunni osakaaluga
- EE_HOLIDAYS: Eesti riigipühade kuupäevad (kohaliku EE kalendripäeva järgi)

Eeldused:
- Samas kaustas: el_consumption_weekday.py (sisaldab sum_hourly_el_consumption DataFrame’i)
- Samas kaustas: weekday_profile.py (arvutusloogika; pühad, viimased N, pivot) – kasutame seda otse. 
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd

# --- Import kohalikest moodulitest ---
try:
    # Sinu olemasolev loogika (pühad, filtrid, pivot)
    import weekday_profile as wp
except Exception as e:
    print(f"ERROR: ei õnnestu importida weekday_profile.py: {e}")
    sys.exit(1)

try:
    import el_consumption_weekday as ecw  # peab sisaldama sum_hourly_el_consumption
except Exception as e:
    print(f"ERROR: ei õnnestu importida el_consumption_weekday.py: {e}")
    sys.exit(1)


def compute_source_rows_with_holidays() -> pd.DataFrame:
    """
    Tagastab DataFrame'i veergudega:
      - sum_cons_time (tz-aware UTC)
      - date_utc (UTC kuupäev)
      - hour_utc (0..23)
      - weekday (inglise keeles)
      - is_holiday (EE kohaliku kalendripäeva järgi)
      - sum_el_hourly_value (tunni tarbimine)
      - hourly_share_of_day (tunni osakaal päeva tarbimisest)
      - ee_local_date (EE kohaliku kuupäevana stringina, mugav ekspordiks)
    NB! Ei välista pühi – see on kogu alusmaterjal.
    """
    df = ecw.sum_hourly_el_consumption.copy()
    if df.empty:
        raise RuntimeError(
            "Sisendandmestik on tühi (sum_hourly_el_consumption).")

    # Veendu, et ajatempel on tz-aware UTC
    if not pd.api.types.is_datetime64_any_dtype(df["sum_cons_time"]):
        df["sum_cons_time"] = pd.to_datetime(
            df["sum_cons_time"], utc=True, errors="coerce")
    else:
        if df["sum_cons_time"].dt.tz is None:
            df["sum_cons_time"] = df["sum_cons_time"].dt.tz_localize("UTC")

    # Lisa pühad (EE, kohaliku kuupäeva järgi) – kasutame sama funktsiooni, mis weekday_profile’is
    # (privaatne abi, aga siinses projektis on ok kasutada)
    df = wp._attach_ee_holidays_local(df)  # pylint: disable=protected-access

    # UTC kuupäev, tund, nädalapäev
    df["date_utc"] = df["sum_cons_time"].dt.date
    df["hour_utc"] = df["sum_cons_time"].dt.hour
    if "weekday" not in df.columns or df["weekday"].isna().all():
        df["weekday"] = df["sum_cons_time"].dt.day_name()

    # Päeva summa ja tunni osakaal
    daily_totals = df.groupby("date_utc")[
        "sum_el_hourly_value"].transform("sum")
    df["hourly_share_of_day"] = df["sum_el_hourly_value"] / daily_totals

    # Kohaliku EE kuupäeva string (lihtsaks auditeerimiseks)
    df["ee_local_date"] = (
        df["sum_cons_time"].dt.tz_convert("Europe/Tallinn").dt.date.astype(str)
    )

    # Jäta alles huvipakkuvad veerud
    cols = [
        "sum_cons_time",
        "date_utc",
        "hour_utc",
        "weekday",
        "is_holiday",
        "sum_el_hourly_value",
        "hourly_share_of_day",
        "ee_local_date",
    ]
    df = df[cols].dropna(subset=["hourly_share_of_day"])
    return df


def compute_filtered_for_profiles(source_df: pd.DataFrame) -> pd.DataFrame:
    """
    Välistab pühad (is_holiday=True) ning tagastab DataFrame'i,
    mida kasutatakse profiilide arvutuseks (viimased N päeva iga nädalapäeva kohta).
    """
    df = source_df[~source_df["is_holiday"].fillna(False)].copy()
    if df.empty:
        raise RuntimeError("Pärast pühade välistamist puuduvad andmed.")
    return df


def build_outputs(last_n: int):
    """
    Koostab 3 põhiväljundit:
      - pivot_df: 24x7 maatriks (hour_utc x weekday), veerusummad=1.0
      - days_used: (weekday, date_utc) – millised kuupäevad läksid arvesse
      - source_df: allikaread koos koefitsentidega (pühad sees)
      - ee_holidays_df: unikaalsed EE pühade kohaliku kuupäeva read
    """
    # 1) Täielik allikas (pühad lipuga)
    source_df = compute_source_rows_with_holidays()

    # 2) Profiilide sisend (pühad välja)
    filtered_df = compute_filtered_for_profiles(source_df)

    # 3) Profiilid + days_used (kasutame sama tuuma, mis weekday_profile'is)
    prof_df, days_used = wp._build_profiles_and_days_used(
        filtered_df, last_n=last_n)  # pylint: disable=protected-access
    pivot_df = wp._profile_df_to_share_matrix(
        prof_df)  # pylint: disable=protected-access

    # 4) Eesti riigipühad (kohaliku kuupäeva alusel)
    ee_holidays_df = (
        source_df[source_df["is_holiday"]]
        .drop_duplicates(subset=["ee_local_date"])
        .loc[:, ["ee_local_date"]]
        .rename(columns={"ee_local_date": "ee_local_holiday_date"})
        .sort_values("ee_local_holiday_date")
        .reset_index(drop=True)
    )

    return pivot_df, days_used, source_df, ee_holidays_df


def save_csvs(pivot_df: pd.DataFrame, days_used: pd.DataFrame,
              source_df: pd.DataFrame, ee_holidays_df: pd.DataFrame,
              out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # PIVOT: 24x7
    pivot_to_save = pivot_df.reset_index()  # hour_utc to veerg
    pivot_path = out_dir / "weekday_hour_share_pivot.csv"
    pivot_to_save.to_csv(pivot_path, index=False)

    # DAYS_USED: weekday, date_utc
    days_used_path = out_dir / "weekday_days_used.csv"
    days_used.sort_values(["weekday", "date_utc"]).to_csv(
        days_used_path, index=False)

    # SOURCE_ROWS: allikaread + koefitsendid (hea auditiks ja kontrolliks)
    source_cols_print = [
        "sum_cons_time",
        "ee_local_date",
        "date_utc",
        "hour_utc",
        "weekday",
        "is_holiday",
        "sum_el_hourly_value",
        "hourly_share_of_day",
    ]
    source_path = out_dir / "hourly_shares_source_rows.csv"
    source_df[source_cols_print].sort_values(
        ["date_utc", "hour_utc"]).to_csv(source_path, index=False)

    # EE_HOLIDAYS: eraldi nimekiri kohaliku kuupäevaga
    holidays_path = out_dir / "ee_holidays_dates.csv"
    ee_holidays_df.to_csv(holidays_path, index=False)

    return {
        "pivot": str(pivot_path),
        "days_used": str(days_used_path),
        "source_rows": str(source_path),
        "ee_holidays": str(holidays_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ekspordi nädalapäeva tunniprofiilid CSV-des.")
    parser.add_argument("--last-n", type=int, default=6,
                        help="Mitu viimast selle nädalapäeva esinemist arvesse võtta (vaikimisi 6).")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Kaust, kuhu CSV-d salvestada (vaikimisi töökaust).")
    args = parser.parse_args()

    pivot_df, days_used, source_df, ee_holidays_df = build_outputs(
        last_n=args.last_n)
    paths = save_csvs(pivot_df, days_used, source_df,
                      ee_holidays_df, Path(args.out_dir))

    print("\n=== Valmis. Salvestasin CSV-d: ===")
    for k, v in paths.items():
        print(f"{k:>12}: {v}")

    print("\n=== Pivot (tunnid ridades, nädalapäevad veergudes) – eelvaade ===")
    print(pivot_df.round(4).to_string())


if __name__ == "__main__":
    main()
