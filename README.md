# Final Project: Electricity Consumption Forecast

## About the Final Project  
- This is the final project of the **Vali-IT Data Analytics 6-week bootcamp**: [vali-it.ee/andmetarkus](https://vali-it.ee/andmetarkus)  
- **Tech stack**: Python, Visual Studio Code, Jupyter Notebook, Git, GitHub  
- **Project authors**:  
  - Johannes Kauksi  
    - Email: johanneskauksi@gmail.com  
    - LinkedIn:[Johannes' linkedin](https://www.linkedin.com/in/johannes-kauksi/)
  - Sergei Erbin  
    - Email: sergei.erbin@gmail.com  
    - LinkedIn: [https://www.linkedin.com/in/sergei-erbin/](https://www.linkedin.com/in/sergei-erbin/)   
  - Tarmo Gede  
    - Email: tarmo.gede@gmail.com  
    - LinkedIn: https://www.linkedin.com/in/tarmo-gede/  

## Introduction of the Company and Research Problem  

Company X is an electricity sales company that needs **to forecast electricity consumption in Estonia at the hourly level** in order to support electricity planning and purchasing decisions.  

## Research plan

1. **Analyze correlation:** Examine the relationship between electricity consumption and temperature using historical data.
2. **Build forecast:** Develop an **hourly** electricity consumption forecast model.
3. **Validate:** Test the forecast **against actuals** and assess accuracy.

### 📌 Project Backlog
[View the backlog here](https://github.com/martinoland1/Electricity-Consumption-Forecast/blob/main/backlog.md)

## Data protection description – on what basis are the data processed?

## Business glossary, data model, data dictionary

### 1. Business Glossary

- **Electricity Consumption (MWh)** – Hourly electricity demand in megawatt-hours, measured by Elering.  
- **Temperature (°C)** – Hourly outdoor air temperature from Meteostat or University of Tartu station.  
- **15-Minute Measurement** – Underlying measurement interval for actual load, aggregated to hourly for forecasting.  
- **Regression Model** – Statistical model linking consumption with temperature.  
- **ARIMA Model** – Alternative time-series model considered for comparison.  
- **Bias Coefficient** – Seasonal correction factor applied to regression forecasts.  
- **Growth Coefficient** – Baseline trend showing long-term consumption growth.  
- **Day-Curve Profile** – 24-hour consumption distribution expressed as percentages.  
- **Workday Profile** – Daily curve for weekdays.  
- **Offday Profile** – Daily curve for weekends and holidays.  
- **Forecast Horizon** – Prediction window, 7 days ahead.  
- **date_local** – Local forecast date (Europe/Tallinn).  
- **datetime_local** – Local timestamp for forecasted hour.  
- **consumption_hourly** – Hourly forecast consumption in MWh.  
- **weekday** – Day of week (0=Mon..6=Sun).  
- **hour_local** – Hour of day in local time.  
- **segment** – Classification: workday, weekend, holiday.  
- **season** – Season of year.  
- **is_weekend** – Weekend flag.  
- **is_holiday** – Holiday flag.  
- **month_num** – Month number.  
- **EE_avg_temp_C** – Average daily temperature.  
- **bias_key** – Key for bias factor.  
- **bias_factor** – Bias correction value.  
- **yhat_base** – Baseline forecast.  
- **yhat_consumption** – Final forecast.  
- **DataFrame** – Python tabular data structure.  
- **ETL** – Extract, Transform, Load process.  
- **Stand-up Meeting** – Daily coordination meeting.  
- **Product Owner** – Role responsible for planning.  
- **Tester** – Role verifying code before merging.  

---

### 2. Data Model (verbal explanation)

See the **DAta model diagram and Data flow** in the dedicated section below.  
The logical structure can be described as follows:

- **Input layer**: Electricity consumption (Elering JSON/CSV) and temperature data (Meteostat API, Tartu University CSV).  
- **Transformation layer**: Data cleaning, time standardization (Europe/Tallinn), derived columns (`weekday`, `season`, `growth coefficient`).  
- **Model layer**: Regression model linking temperature with consumption, corrected by seasonal *bias coefficients*.  
- **Profile layer**: Daily consumption curves (workday, weekend, holiday) providing *hourly distribution coefficients*.  
- **Forecast layer**: Daily forecast values are distributed into hourly forecasts using the profiles.  
- **Output layer**: Final 7-day forecast table containing hourly consumption (`consumption_hourly`) and supporting metadata (`segment`, `season`, `bias_factor`, etc.).  

---

### 3. Data Dictionary

| Column             | Format              | Description |
|--------------------|--------------------|-------------|
| date_local         | String (YYYY-MM-DD) | Local forecast date |
| datetime_local     | Timestamp           | Forecasted hour in local time |
| consumption_hourly | Float (MWh)         | Forecasted hourly consumption |
| weekday            | Integer (0–6)       | Day of week (0=Mon..6=Sun) |
| hour_local         | Integer (0–23)      | Hour of day in local time |
| segment            | String              | Workday / weekend / holiday classification |
| season             | String              | Season: winter, spring, summer, autumn |
| is_weekend         | Boolean             | Weekend flag |
| is_holiday         | Boolean             | Holiday flag |
| month_num          | Integer (1–12)      | Month number |
| EE_avg_temp_C      | Float (°C)          | Average daily temperature in Estonia |
| bias_key           | String              | Key for bias factor (e.g., segment + season) |
| bias_factor        | Float               | Bias correction factor |
| yhat_base          | Float               | Baseline forecast without daily curve adjustment |
| yhat_consumption   | Float               | Final daily forecast value |


### Data model diagram

[![Data model](docs/data_model.png)](https://raw.githubusercontent.com/martinoland1/Electricity-Consumption-Forecast/main/docs/data_model.png)

## process/data flow diagram
<img width="1065" height="363" alt="image" src="https://github.com/user-attachments/assets/ea669326-ffac-4d55-a19c-2ca80ae854f0" />

### Forecasting Pipeline Overview
The model integrates electricity consumption data from Elering and weather information from Meteostat into a combined dataframe. This serves as the central input for the forecasting pipeline:
- A regression formula is applied to capture the relationship between daily average temperature and electricity consumption.
- A bias correction module adjusts the regression output to reduce systematic errors (e.g., seasonal or structural effects).
- A weekday profile provides the hourly load distribution, refining daily forecasts into hourly demand curves.
- The pipeline first produces a daily forecast and then disaggregates it into an hourly forecast using the weekday/hourly patterns.

#### Elering API
| Source System | Source Column | Python pipeline DataFrame column | Column Format | Description |
|---------------|---------------|----------------------------------|---------------|-------------|
| Elering API   | timestamp     | sum_cons_time                    | datetime (tz-aware, Europe/Tallinn) | Measurement time (hourly granularity) |
| Elering API   | consumption   | sum_el_hourly_value              | float (MWh)   | Hourly electricity consumption in MWh |

#### Hourly (get_hourly_consumption)
| Source System | Source Column | Python pipeline DataFrame column | Column Format                                    | Description |
|---------------|---------------|----------------------------------|--------------------------------------------------|-------------|
| Elering API   | timestamp     | sum_cons_time                    | datetime (tz-aware, Europe/Tallinn)              | Measurement time (hourly granularity) |
| Elering API   | consumption   | sum_el_hourly_value              | float (MWh)                                      | Hourly electricity consumption in MWh |
| — (derived)   | —             | imputed                          | boolean                                          | True, if value was filled by neighbor interpolation |
| — (derived)   | —             | weekday                          | string                                           | Day name (e.g., Monday, Tuesday) |
| — (derived)   | —             | is_weekend                       | boolean                                          | True if Saturday or Sunday |
| — (derived)   | —             | is_holiday                       | boolean                                          | True if Estonian public holiday |

#### Daily (get_daily_consumption)
| Source System | Source Column | Python pipeline DataFrame column | Column Format                               | Description |
|---------------|---------------|----------------------------------|---------------------------------------------|-------------|
| Aggregated    | —             | sum_cons_date                    | date (local, Europe/Tallinn)                | Local calendar day (aggregation bucket) |
| Aggregated    | —             | sum_el_daily_value               | float (MWh)                                  | Daily electricity consumption in MWh (sum of hourly values) |
| — (derived)   | —             | weekday                          | string                                       | Day name (e.g., Monday, Tuesday) |
| — (derived)   | —             | is_weekend                       | boolean                                      | True if Saturday or Sunday |
| — (derived)   | —             | is_holiday                       | boolean                                      | True if Estonian public holiday |


#### Meteostat API
| Source System     | Source Column | Python pipeline DataFrame column | Column Format | Description |
|-------------------|---------------|----------------------------------|---------------|-------------|
| Meteostat API     | datetime      | hour_temp_time                   | datetime (tz-aware, Europe/Tallinn) | Measurement time (hourly granularity) |
| Meteostat API     | temp          | hour_temp_value                  | float (°C)    | Hourly average temperature across Estonian points |

#### 

## Creation of a sample dataset

### Data Sources

- **Electricity Consumption (JSON)** – [Elering Swagger](https://dashboard.elering.ee/assets/swagger-ui/index.html)  
- **Electricity Consumption (CSV)** – [Elering LIVE](https://dashboard.elering.ee/et/system/with-plan/production-consumption?interval=minute&period=days&start=2025-09-14T21:00:00.000Z&end=2025-09-15T20:59:59.999Z)  

- **Hourly Temperature (Meteostat Python Library)** – [Meteostat Python Library](https://dev.meteostat.net/python/hourly.html)  
- **Hourly temperature (CSV)** – [University of Tartu, Institute of Physics Weather Station](https://meteo.physic.ut.ee/)  

## Data quality check

## Exploratory data analysis
### 1. Yearly consumption and average temperature
This chart shows the total electricity consumption per year alongside the average air temperature.
- From 2020 to 2024, consumption remained relatively stable around 8 million MWh, while temperatures varied.
- The highest consumption was observed in 2021 (8.4M MWh).
- Data for 2025 is currently incomplete, showing a lower total (5.6M MWh).
- In general, colder years correspond with higher electricity usage.
<img width="1621" height="728" alt="image" src="https://github.com/user-attachments/assets/57d1f355-6ade-4590-840c-08749fef2fb0" />

### 2. Monthly consumption and temperature
This visualization illustrates the relationship between consumption and temperature on a monthly level.
- Winter months (December–February): higher consumption due to heating demand, with temperatures dropping below zero.
- Summer months (June–August): consumption decreases, while average temperatures peak at 16–19 °C.
- The inverse relationship is clear – colder months drive higher demand, warmer months lower demand.
<img width="1381" height="741" alt="image" src="https://github.com/user-attachments/assets/9bcdd81e-6dc6-42d9-834e-b0491fef6d34" />

### 3. Year-to-year comparison (2023 vs 2024)
This comparison highlights monthly consumption and temperatures across two consecutive years.
- Both years show the same seasonal pattern: higher demand in winter, lower in summer.
- Temperatures follow a similar curve, but January 2023 was colder than January 2024.
- The comparison suggests that even small differences in temperature (e.g., colder January) can cause significant changes in electricity demand.
<img width="1318" height="741" alt="image" src="https://github.com/user-attachments/assets/2b79b65a-62d8-434c-a092-94caa9d7e0e8" />

### 4. Average consumption daily profile
Daily Consumption Profiles (by Day of the Week)
This visualization shows the average daily consumption patterns broken down by each day of the week.
- The chart highlights how electricity usage typically evolves hour by hour across weekdays and weekends:
- Workdays (Mon–Fri) follow a consistent pattern, with a clear morning peak (around 7–10 AM) and a gradual evening decline.
- Saturdays display a flatter profile, with less pronounced peaks, indicating more evenly distributed consumption throughout the day.
- Sundays show the lowest overall demand, especially during the morning hours, reflecting reduced industrial and commercial activity.
- This analysis provides insights into behavioral and operational differences between weekdays and weekends, which is essential for forecasting hourly demand and adjusting models for different calendar profiles.
<img width="1303" height="738" alt="image" src="https://github.com/user-attachments/assets/164354f1-1cc7-4e83-9c7b-fc5b3557757b" />



## Statistical data analysis
### Regression Analysis (Daily Consumption vs. Temperature)
We fitted linear regression models to study the relationship between average daily temperature and electricity consumption in Estonia.
All Days: A strong negative correlation was observed between temperature and consumption. As temperature rises, electricity usage decreases, reflecting reduced heating demand. (Slope ≈ –395 MWh/°C, R² ≈ 0.75).
Workdays (Mon–Fri, non-holiday): The model shows an even stronger relationship (Slope ≈ –404 MWh/°C, R² ≈ 0.82), indicating that weekday consumption patterns are highly temperature-dependent.
Weekends & Holidays: The slope is slightly lower (≈ –382 MWh/°C, R² ≈ 0.81), but the correlation remains strong, showing that leisure days also follow the same inverse trend.
These models will later be integrated into the forecasting pipeline, where the segmented regressions (workdays vs. weekends/holidays) allow us to capture different behavioral patterns and improve accuracy in consumption prediction.
<img width="713" height="496" alt="image" src="https://github.com/user-attachments/assets/08ea2dcc-25ea-4fa6-8710-5c3832802e99" />
<img width="713" height="496" alt="image" src="https://github.com/user-attachments/assets/9f00be4e-60bc-4fc2-ad17-ee38230995ec" />
<img width="728" height="506" alt="image" src="https://github.com/user-attachments/assets/6787e7f2-7a73-43c7-aa26-14b7ab7a67eb" />

### Bias calculation
Seasonal Bias Correction Factors
To further refine forecasts, we computed bias correction factors by season (Spring, Summer, Autumn, Winter).
Spring (March–May) and Autumn (September–November): Transitional periods, where actual consumption often deviates from regression estimates due to rapid shifts in heating/cooling needs.
Winter (December–February): Highest bias adjustments, since cold extremes drive significantly higher electricity usage than predicted by a simple linear trend.
Summer (June–August): Lowest adjustments, as electricity consumption stabilizes and is less variable with temperature.
These factors are applied as multipliers to regression-based forecasts, either by season or by month, and can be segmented further into workdays vs. off-days.
<img width="989" height="390" alt="bias" src="https://github.com/user-attachments/assets/0847b029-2811-4cfe-bb61-a21c8af894e6" />

### Weekday Load Profiles (24×7 Hourly Share Matrix)

We analyzed the average daily consumption patterns across weekdays by building a 24×7 share matrix. Each column (weekday) sums to 1.0, showing how electricity demand is distributed across the 24 hours of the day.
Key insights:
Early morning (0–5h): Lowest relative consumption across all weekdays (~3–3.5% of daily total per hour).
Morning ramp-up (6–9h): Sharp increase, peaking around 8–10h (up to ~5% of daily total per hour), especially on workdays.
Daytime plateau (10–16h): Stable, elevated demand throughout business hours.
Evening peak (18–21h): Second wave of demand, stronger on weekends, reflecting household activity.
Weekday vs Weekend: Weekdays show stronger morning peaks due to industrial and office usage, while weekends flatten daytime demand and shift load toward later hours.
This matrix is DST-aware and based on the Europe/Tallinn timezone, with public holidays excluded from the training sample. It is later used to adjust hourly forecast distributions once daily consumption is predicted.
<img width="772" height="440" alt="weekday_profile" src="https://github.com/user-attachments/assets/b090625f-3d6c-4149-bc68-f11673a26697" />


## Descriptive report / analysis

## Data story, conclusions
