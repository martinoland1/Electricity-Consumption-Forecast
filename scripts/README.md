## Modules

| Script | Description | Python |
|---|---|---|
| `ecf_runner.ipynb` | Interactive orchestration notebook for the full ECF pipeline (Europe/Tallinn). Runs: consumption & temperature snapshots → regression & bias → 7-day temperature forecast → 7-day daily forecast → weekday profiles → hourly split → hourly forecast. Includes previews, charts, and optional CSV exports to `./output/`. | N/A — open in Jupyter (not a module) |
| `elering_consumption.py` | Load **hourly** and **daily** consumption (EE calendar). Handles tz, optional weekday/holiday flags, and gap imputation. | `from elering_consumption import get_hourly_consumption, get_daily_consumption` |
| `meteostat_temperature.py` | Historical temperature: **hourly** → `hour_temp_time`, `hour_temp_value`; **daily** → `avg_day_temp_date`, `hour_day_value` (EE calendar). | `from meteostat_temperature import get_hourly_temperature, get_daily_temperature` |
| `regression_analysis.py` | Merge daily consumption + daily temperature and fit a **linear** model (`y = a + b·T`); output metrics (R², RMSE, MAE) and optional plots. | `from regression_analysis import load_daily_frames, run_linreg` |
| `bias_analysis.py` | Compute **bias factors** by **season**/**month**, optionally **segmented** (workday/offday). Saves factor **table (CSV)** + **map (JSON)**. | `from bias_analysis import get_bias_factors, apply_bias_to_forecast` |
| `temp_forecast.py` | **7-day temperature forecast** (exactly **tomorrow → +6**). Returns daily averages per city + `EE_avg` on the EE calendar. | `from temp_forecast import get_next7_forecast` |
| `el_consumption_forecast.py` | **Daily 7-day consumption forecast** using regression + bias; supports segmented bias and custom temperature source. | `from el_consumption_forecast import forecast_next7, plot_dual_axis_bars` |
| `weekday_profile.py` | Build **24×7 weekday share matrix** (DST-aware, EE public holidays filtered) & **split** daily totals into hourly values. | `from weekday_profile import get_weekday_hour_share_matrix, get_weekday_days_used, split_daily_forecast_to_hourly` |
| `electricity_hourly_forecast.py` | Full **hourly 7-day** forecast: compute/ingest daily, split by weekday profiles, reconcile sums, save CSV. | `from electricity_hourly_forecast import main as run_hourly` |
