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

### Backlog
#### 13.09
- [ ] **Build an Excel prototype for electricity consumption forecasting (import data, run regression, produce forecast, format output).** — *Owner:* @sergeierbin
#### 15.09
- [x] **Create the data model diagram and XML.** — *Owners:* @martinoland1 @sergeierbin *Owner:* @tarmogede-dev
- [x] **Import the electricity consumption JSON dataset into Python.** — *Owner:* @sergeierbin
- [x] **Import the temperature dataset into Python using the Meteostat library.** — *Owner:* @tarmogede-dev
- [x] **Perform a regression analysis in Python to assess the relationship between consumption and temperature.** — *Owner:* @martinoland1
#### 16.09
- [ ] **Update the DataFrame & data model** — add `imputed` column, consumption growth, weekday, day-curve coefficient, and consumption forecast. — *Owners:* @martinoland1 @sergeierbin *Owner:* @tarmogede-dev
- [x] **Estimate temperature-independent consumption trend** — compute the baseline growth coefficient (preferably monthly; optionally daily). — *Owner:* @martinoland1
- [ ] **Analyze daily consumption curves and identify typical daily load patterns.** — *Owner:* @tarmogede-dev
- [x] **Import the temperature forecast dataset into Python using the Meteostat library.** — *Owner:* @martinoland1
- [x] **Build next-day daily forecast** — combine regression and, baseline growth; use the average-day temperature forecast. — *Owner:* @sergeierbin
#### 17.09
- [x] **Compute a curve coefficient for each weekday.** — *Owner:* @sergeierbin
- [x] **Build next-day hourly forecast** — combine next-day daily forecast and day-curve. — *Owner:* @martinoland1
- [ ] **Create a business glossary and data table descriptions** — *Owner:* @tarmogede-dev
- [ ] **Describe the Excel prototype** — *Owner:* @sergeierbin
- [ ] **Test the forecasting model** — *Owner:* @tarmogede-dev
- [ ] **Investigate importing Python code into Power BI** — *Owner:* @martinoland1
- [x] **Refactor scripts to consistently use Europe/Tallinn time** — *Owner:* @sergeierbin

#### 18.09
- [ ] **Validate forecast vs actuals** — compare errors and summarize. — *Owner:*
- [ ] **Create Power BI reports** — consumption comparison (year/month), day-curve comparison, temperature comparison (year/month), etc. — *Owner:*
- [ ] **Prepare the PowerPoint presentation** — summarize methodology, metrics, visuals, and key takeaways. — *Owner:*


### Open Question  
Should additional factors be considered in the research plan?  

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

See the **Data model diagram** in the dedicated section below.  
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

## Data flow
[![Data model](docs/el_cons_data_model-data_lineage_model.jpg)](https://github.com/martinoland1/Electricity-Consumption-Forecast/blob/main/docs/el_cons_data_model-data_lineage_model.jpg)


## Creation of a sample dataset

### Data Sources

- **Electricity Consumption (JSON)** – [Elering Swagger](https://dashboard.elering.ee/assets/swagger-ui/index.html)  
- **Electricity Consumption (CSV)** – [Elering LIVE](https://dashboard.elering.ee/et/system/with-plan/production-consumption?interval=minute&period=days&start=2025-09-14T21:00:00.000Z&end=2025-09-15T20:59:59.999Z)  

- **Hourly Temperature (Meteostat Python Library)** – [Meteostat Python Library](https://dev.meteostat.net/python/hourly.html)  
- **Hourly temperature (CSV)** – [University of Tartu, Institute of Physics Weather Station](https://meteo.physic.ut.ee/)  

## Data quality check

## Exploratory data analysis

## Statistical data analysis

## Descriptive report / analysis

## Data story, conclusions
