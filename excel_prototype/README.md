1. Consumption data (CSV) was taken for the period **1 Sep 2024 – 31 Aug 2025** from the Elering Live page: <https://dashboard.elering.ee/et/system/with-plan/production-consumption?interval=minute&period=search&start=2024-08-31T21:00:00.000Z&end=2025-08-31T20:59:59.000Z&show=table>
2. Temperature data (CSV) was taken for **1 Sep 2024 – 31 Aug 2025** from the University of Tartu site: <https://meteo.physic.ut.ee/>
3. The consumption and temperature CSVs were imported into Excel and joined using the **VLOOKUP** function.
4. From the joined data, the **daily total consumption** and **daily average temperature** were calculated.
5. A **scatter chart** of daily total consumption vs. daily average temperature showed a strong **linear relationship**.
6. An **Excel Regression** analysis was performed on the daily consumption vs. temperature data.
7. Using the resulting regression equation, a **daily consumption forecast** was produced for **5 Sep 2025 – 12 Sep 2025**.
8. **Daily load profiles** (day curves) were derived for period **X**.
9. The **daily consumption** was **distributed to hourly values** using the derived day profiles.
