# temp_forecast.py
# Järgmise 7 päeva päevakeskmised temperatuurid (UTC päevad) Meteostat'iga.
# OLULINE: Meteostat Hourly vajab naive (tzinfo=None) UTC datetimes.

from datetime import timedelta
import pandas as pd
from meteostat import Point, Hourly

# --- Punktid ---
points = {
    "Tallinn":    Point(59.4370, 24.7536),
    "Tartu":      Point(58.3776, 26.7290),
    "Pärnu":      Point(58.3859, 24.4971),
    "Narva":      Point(59.3793, 28.2000),
    "Kuressaare": Point(58.2528, 22.4869),
}

# --- Ajavahemik UTC-s: alusta järgmisest UTC keskööst (täis 24h päevad) ---
now_utc = pd.Timestamp.now(tz="UTC")
start_utc = (now_utc.floor("D") + pd.Timedelta(days=1))
end_utc = start_utc + pd.Timedelta(days=7)

# Meteostat'i jaoks TULEB anda naive UTC datetimes (tzinfo=None)
start_dt = start_utc.to_pydatetime().replace(tzinfo=None)
end_dt = end_utc.to_pydatetime().replace(tzinfo=None)


def daily_avg_temp_utc(pt: Point, start_dt_naive, end_dt_naive) -> pd.Series:
    # model=True -> lülita sisse mudeliandmed (forecast)
    df = Hourly(pt, start_dt_naive, end_dt_naive, model=True).fetch()
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    # Meteostat võib tagastada 'time' veerus; teeme sellest indeksi
    if "time" in df.columns:
        df = df.set_index("time")

    # Kindlusta UTC indeks
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Päevakeskmine (UTC kalendripäev)
    s = df["temp"].resample("D").mean()
    return s


# --- Arvuta kõigi punktide seeria ja joonda täpselt 7 päevale ---
all_series = []
for name, pt in points.items():
    s = daily_avg_temp_utc(pt, start_dt, end_dt)
    s.name = name
    all_series.append(s)

days_utc = pd.date_range(start=start_utc, periods=7, freq="D", tz="UTC")
result = (pd.concat(all_series, axis=1)
          if all_series else pd.DataFrame(index=days_utc))
result = result.reindex(days_utc)

# Väljund: YYYY-MM-DD, ümardus 1 koht
out = result.round(1).reset_index().rename(columns={"index": "date_utc"})
out["date_utc"] = out["date_utc"].dt.strftime("%Y-%m-%d")

print("\nJärgmise 7 päeva päevakeskmised temperatuurid (°C) — UTC:")
print(out.to_string(index=False))
