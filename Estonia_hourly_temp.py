#!/usr/bin/env python3
"""
Eesti tunnikeskmine temperatuur viimase 1 päeva kohta (Meteostat).

- Küsib kõik Eesti jaamadelt (EE).
- Arvutab iga tunni kohta jaamade keskmise.
- Salvestab CSV ja Parquet.

Kasutus:
    pip install meteostat pandas pyarrow pytz
    python meteostat_estonia_lastday.py
"""

from datetime import datetime, timedelta
import pandas as pd
from meteostat import Stations, Hourly
import pytz

TZ = "Europe/Tallinn"

def get_estonian_stations():
    stations = Stations().region('EE')
    df = stations.fetch()
    return df.index.tolist()

def fetch_hourly_mean(start, end, station_ids):
    data = Hourly(station_ids, start, end).fetch()

    # Kui mitu jaama, siis unstackimiseks:
    if isinstance(data.index, pd.MultiIndex) and 'station' in data.index.names:
        temp_wide = data['temp'].unstack('station')
    else:
        temp_wide = data[['temp']].rename(columns={'temp': 'single'})

    print(f"\nIMPOORDITUD ANDMERIDU: {len(temp_wide)} (tunnid)")

    # Lisa või teisenda ajavöönd
    if temp_wide.index.tz is None:
        temp_wide.index = temp_wide.index.tz_localize("UTC").tz_convert(TZ)
    else:
        temp_wide.index = temp_wide.index.tz_convert(TZ)

    # --- Andmekontroll: puuduvad väärtused ---
    missing_per_hour = temp_wide.isna().sum(axis=1)
    missing_per_station = temp_wide.isna().sum(axis=0)
    total_missing = temp_wide.isna().sum().sum()
    if total_missing > 0:
        print("\nANDMEKONTROLL: Puuduvad väärtused!")
        print(f"  Kokku puudub {int(total_missing)} mõõtmist.")
        print("  Puuduvate mõõtmiste arv tunni kaupa (kui >0):")
        print(missing_per_hour[missing_per_hour > 0])
        print("  Puuduvate mõõtmiste arv jaama kaupa (kui >0):")
        print(missing_per_station[missing_per_station > 0])
    else:
        print("\nANDMEKONTROLL: Puuduvad väärtused puuduvad (kõik olemas)")

    # --- Interpoleeri puuduvad väärtused ja märgista originaal/interpoleeritud ---
    temp_interp = temp_wide.copy()
    orig_mask = ~temp_interp.isna()
    temp_interp = temp_interp.interpolate(method='linear', axis=0, limit_direction='both')
    interp_mask = orig_mask.copy()
    interp_mask[temp_interp.notna() & ~orig_mask] = False  # False = interpoleeritud, True = originaal
    interp_mask[orig_mask] = True

    # Koosta DataFrame, kus iga jaama kohta on kaks veergu: temperatuur ja kas originaal (True/False)
    result_cols = {}
    for col in temp_interp.columns:
        result_cols[f"{col}_temp"] = temp_interp[col]
        result_cols[f"{col}_orig"] = interp_mask[col]
    result_df = pd.DataFrame(result_cols, index=temp_interp.index)

    # Arvuta Eesti keskmine tunnitemperatuur interpoleeritud andmete põhjal
    result_df['temp_mean_c'] = temp_interp.mean(axis=1)

    return result_df

def main():

    end = datetime.now()
    # Võta vaikimisi 10 päeva, mitte 30
    start = end - timedelta(days=10)


    # Võta jaamade kogu info DataFrame
    stations_df = Stations().region('EE').fetch()
    station_ids = stations_df.index.tolist()

    result_df = fetch_hourly_mean(start, end, station_ids).sort_index()

    # Muudame indeksi (datetime) stringiks ilma ajavööndita ja lisame eraldi veerud

    df = result_df.copy()
    df = df.reset_index()
    # Leia ajatähise veerg (datetime64 tüüpi)
    datetime_col = None
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_col = col
            break
    if datetime_col is None:
        raise ValueError('Ajatähise veergu ei leitud!')
    df['hour_temp_time'] = df[datetime_col].dt.tz_localize(None).dt.strftime('%Y-%m-%d %H:%M:%S')
    df['hour_temp_value'] = df['temp_mean_c']
    main_cols = ['hour_temp_time', 'hour_temp_value']
    other_cols = [c for c in df.columns if c not in main_cols and c != datetime_col and c != 'temp_mean_c']
    df = df[main_cols + sorted(other_cols)]

    df.to_csv("estonia_hourly_temp_mean_lastday.csv", index=False)
    df.to_parquet("estonia_hourly_temp_mean_lastday.parquet", index=False)

    # Salvesta kasutatud jaamade kogu info (ID, nimi, asukoht jne)
    stations_df = stations_df.reset_index()
    stations_df.to_csv("estonia_stations_used.csv", index=False)

    print(df.tail())

    # --- Tartu Füüsikainstituudi tunniandmete liitmine eemaldatud ---

    # --- Jaamade asukohtade visualiseerimine kaardil (folium) ---
    try:
        import folium
        # Loo kaart Eesti keskpunktiga
        m = folium.Map(location=[58.5953, 25.0136], zoom_start=7)
        # Lisa iga jaam kaardile
        for _, row in stations_df.iterrows():
            if not (pd.isna(row.get('latitude')) or pd.isna(row.get('longitude'))):
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=f"{row.get('name', '')} ({row.get('id', row.get('station_id', ''))})",
                    tooltip=row.get('name', '')
                ).add_to(m)
        m.save("estonia_stations_map.html")
        print("Jaamade kaart salvestatud: estonia_stations_map.html")
    except ImportError:
        print("Folium pole paigaldatud. Kaardi loomiseks paigalda: pip install folium")

if __name__ == "__main__":
    main()
