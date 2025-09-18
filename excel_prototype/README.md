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

8. Using the resulting regression equation, a **daily consumption forecast** was produced for **5 Sep 2025 – 12 Sep 2025**.  

9. **Daily load profiles** (day curves) were derived for the period **X**.  

10. The **daily consumption** was **distributed into hourly values** using the derived day profiles.  

## Conclusion

Temperature is a strong driver of daily electricity consumption: the model explains ~65% of the variance (R²=0.646), and each +1 °C is associated with ~331 units lower daily demand (95% CI: −357…−306). The regression is highly significant (F-test p<1e−83). A simple linear model, **Consumption ≈ 24,131.9 − 331.4 × AvgDailyTemp(°C)**, provides a solid baseline; the remaining ~35% likely reflects weekday and holidays effects, trends, seasonality.

