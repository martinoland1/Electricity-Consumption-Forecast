# Excel prototype

## Purpose of the Excel Prototype  
The goal of the Excel prototype was to explore the correlation between temperature and electricity consumption and to assess its suitability for forecasting energy demand.

## Prototype Development Steps

1. Consumption data (CSV) was taken for the period **1 Sep 2024 – 31 Aug 2025** from the Elering Live page:  
   <https://dashboard.elering.ee/et/system/with-plan/production-consumption?interval=minute&period=search&start=2024-08-31T21:00:00.000Z&end=2025-08-31T20:59:59.000Z&show=table>

2. Temperature data (CSV) was taken for **1 Sep 2024 – 31 Aug 2025** from the University of Tartu site:  
   <https://meteo.physic.ut.ee/>

3. The consumption and temperature CSVs were imported into Excel.  

4. **Daily total consumption** and **daily average temperature** were calculated.  

5. A **scatter chart** of daily total consumption vs. daily average temperature showed a strong **linear relationship**.
     
   [![Scatter Chart](https://github.com/martinoland1/Electricity-Consumption-Forecast/blob/main/excel_prototype/scatter_chart.png)](https://github.com/martinoland1/Electricity-Consumption-Forecast/blob/main/excel_prototype/scatter_chart.png)

7. An **Excel Regression** analysis was performed on the daily consumption vs. temperature data.  

![Regression Analysis](https://github.com/martinoland1/Electricity-Consumption-Forecast/blob/main/excel_prototype/regression_analysis.png)

8. Using the resulting regression equation, a **daily consumption forecast** was produced for **6 Sep 2025 – 12 Sep 2025**.  

![Daily Consumption Forecast](https://github.com/martinoland1/Electricity-Consumption-Forecast/blob/main/excel_prototype/daily_consumption_forecast.png)

9. **Daily load profiles** (day curves) were derived based on data from **1 Sep 2024 – 31 Aug 2025**, showing the typical hourly distribution of electricity consumption within a day.

![Daily Profile](https://github.com/martinoland1/Electricity-Consumption-Forecast/blob/main/excel_prototype/daily_ptofile.png)

11. The **daily consumption** was **distributed into hourly values** using the derived day profiles.
   
![Hourly Consumption Forecast](https://github.com/martinoland1/Electricity-Consumption-Forecast/blob/main/excel_prototype/hourly_consumption_forecast.png)

12. **Forecast Error Rate** between actual and forecasted electricity consumption.  
 
![Consumption vs Forecast % Graphs](https://github.com/martinoland1/Electricity-Consumption-Forecast/blob/main/excel_prototype/consumption_vs_forecast_%25.png)

13. **Day-by-day comparison** of actual vs. forecasted consumption.

[![Consumption vs Forecast Graphs](https://github.com/martinoland1/Electricity-Consumption-Forecast/blob/main/excel_prototype/consumption_vs_forecast_graphs.png)](https://github.com/martinoland1/Electricity-Consumption-Forecast/blob/main/excel_prototype/consumption_vs_forecast_graphs.png)

14. **Data and calculations** are available here: [Excel Prototype Workbook](https://github.com/martinoland1/Electricity-Consumption-Forecast/blob/main/excel_prototype/excel_prototype.xlsx)

## Conclusion

Temperature is a strong driver of daily electricity consumption: the model explains ~65% of the variance (R²=0.646), and each +1 °C is associated with ~331 units lower daily demand (95% CI: −357…−306). The regression is highly significant (F-test p<1e−83). A simple linear model, **Consumption ≈ 24,131.9 − 331.4 × AvgDailyTemp(°C)**, provides a solid baseline; the remaining ~35% likely reflects weekday and holiday effects, trends, and seasonality.  

A better correlation and more accurate daily forecasts are likely if calculations are performed separately for weekdays and weekends/holidays. Forecast accuracy can also be improved with an adjustment coefficient that accounts for overall demand growth, seasonality, and other unidentified factors. Hourly forecasts can be made more precise by analyzing which daily load profile trend to apply (e.g., from the most recent month, quarter, etc.). The Excel prototype demonstrated that this model is valuable and worth further exploration.
