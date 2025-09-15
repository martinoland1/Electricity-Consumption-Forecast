# Final Project: Electricity Consumption Forecast

## About the Final Project  
- This is the final project of the **Vali-IT Data Analytics 6-week bootcamp**: [vali-it.ee/andmetarkus](https://vali-it.ee/andmetarkus)  
- **Tech stack**: Power BI, Excel, SQL, PostgreSQL, DBeaver, Python, Jupyter Notebook, Visual Studio Code, Git, GitHub  
- **Project authors**:  
  - Johannes Kauksi  
    - Email: *...*  
    - LinkedIn: *link* 
  - Sergei Erbin  
    - Email: sergei.erbin@gmail.com  
    - LinkedIn: [https://www.linkedin.com/in/sergei-erbin/](https://www.linkedin.com/in/sergei-erbin/)   
  - Tarmo Gede  
    - Email: *...*  
    - LinkedIn: *link*  

## Introduction of the Company and Research Problem  

Company X is an electricity sales company that needs to forecast electricity consumption in Estonia at the **monthly, daily, and hourly level** in order to support electricity purchasing and planning decisions.  

The research problem is twofold:  

### Part 1. Identifying the drivers of electricity consumption in Estonia

The aim is to understand which factors influence electricity demand, such as:  
- **Weather conditions** – temperature, precipitation, wind speed, solar radiation  
- **Calendar effects** – month, day of the week (working day, weekend, public holiday)  
- **Other external factors** – X (to be identified during the analysis)  

### Part 2. Forecasting electricity consumption  
Based on the identified drivers, the goal is to produce a **forecast of electricity consumption for October 2025**, broken down into months, days, and hours. And test the model against real data.

## Research plan

1. **Find the correlation between electricity consumption and temperature at the monthly level**  
   - Use the last 5 years of data (or more if available).  
   - Outcome: estimate monthly electricity consumption based on temperature forecasts.  

2. **Find the correlation between electricity consumption and temperature at the daily level**  
   - Focus on data from a specific month.  
   - Outcome: estimate daily electricity consumption based on daily temperature forecasts.  

3. **Analyze daily consumption curves**  
   - Identify typical daily load patterns.  
   - Outcome: allocate the daily consumption forecast across the 24 hours of the day.  

## To-Do List

- [ ] **Import the electricity consumption JSON dataset into Python.** — *Owner:* @sergeierbin
- [ ] **Import the temperature dataset into Python using the Meteostat library.** — *Owner:* @tarmogede-dev
- [ ] **Perform a regression analysis in Python to assess the relationship between consumption and temperature.** — *Owner:* @martinoland1

### Open Question  
Should additional factors be considered in the research plan?  

## Data protection description – on what basis are the data processed?

## Business glossary, data model, data dictionary

### Data model

[![Data model](docs/data_model.png)](https://raw.githubusercontent.com/martinoland1/Electricity-Consumption-Forecast/main/docs/data_model.png)

## Data flow

## Creation of a sample dataset

### Data Sources  
- **Electricity Consumption** – [Elering](https://dashboard.elering.ee/et/system/with-plan/production-consumption?interval=minute&period=search&start=2024-08-31T21:00:00.000Z&end=2025-08-31T20:59:59.000Z&show=table)  
- **Temperature Data** – [University of Tartu, Institute of Physics Weather Station](https://meteo.physic.ut.ee/)  
 
## Data quality check

## Exploratory data analysis

## Statistical data analysis

## Descriptive report / analysis

## Data story, conclusions
