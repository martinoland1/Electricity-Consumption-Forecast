# Backlog

## 13.09
- [x] **Build an Excel prototype for electricity consumption forecasting (import data, run regression, produce forecast, format output).** — *Owner:* Sergei

## 15.09
- [x] **Create the initial data model diagram and XML.** — *Owners:* Joahnnes, Sergei, Tarmo
- [x] **Import the electricity consumption JSON dataset into Python.** — *Owner:* Sergei
- [x] **Import the temperature dataset into Python using the Meteostat library.** — *Owner:* Tarmo
- [x] **Perform a regression analysis in Python to assess the relationship between consumption and temperature.** — *Owner:* Johannes

## 16.09
- [x] **Update the DataFrame & data model** — add `imputed` column, consumption growth, weekday, day-curve coefficient, and consumption forecast. — *Owners:* Johannes, Sergei, Tarmo
- [x] **Estimate temperature-independent consumption trend** — compute the baseline growth coefficient (preferably monthly; optionally daily). — *Owner:* Johannes
- [x] **Analyze daily consumption curves and identify typical daily load patterns.** — *Owner:* Tarmo
- [x] **Import the temperature forecast dataset into Python using the Meteostat library.** — *Owner:* Johannes
- [x] **Build next-day daily forecast** — combine regression and, baseline growth; use the average-day temperature forecast. — *Owner:* Sergei

## 17.09
- [x] **Compute a curve coefficient for each weekday.** — *Owner:* Sergei
- [x] **Build next-day hourly forecast** — combine next-day daily forecast and day-curve. — *Owner:* Johannes
- [x] **Create a business glossary and data table descriptions** — *Owner:* Tarmo
- [x] **Investigate importing Python code into Power BI** — *Owner:* Johannes
- [x] **Refactor scripts to consistently use Europe/Tallinn time** — *Owner:* Sergei

## 18.09
- [x] **Describe the Excel prototype** — *Owner:* Sergei
- [ ] **Validate forecast vs actuals** — compare errors and summarize. — *Owner:* Tarmo
- [ ] **Create Power BI reports** — consumption comparison (year/month), day-curve comparison, temperature comparison (year/month), etc. — *Owner:* Johannes

## 19.09


## Next Tasks  
- [ ] **User-Friendly Execution** – provide an easy-to-run Python launcher script or Jupyter Notebook. — *Owner:* Sergei
- [ ] **Document Scripts in README** – describe how to use each script. — *Owner:* Sergei
- [ ] **Add DataFrame Descriptions** – include DataFrame explanations in the script documentation.  — *Owner:* Sergei 

- [ ] **Quality Check Script** – create a Jupyter Notebook to investigate errors and validate data quality.  
- [ ] **Python Review** – ask Kaido to review the Python scripts.  
- [ ] **Data Flow Diagram** – create a high-level data flow visualization.  
- [ ] **Extend Data Models** – update and refine Python DataFrames.  
- [ ] **Power BI Data Model** – build a Power BI data model for reporting.  
- [ ] **Translate Scripts into English** – ensure consistency and accessibility.  
- [ ] **Error Warning** – add a notification when the forecast error exceeds a defined threshold (e.g., X%).  
- [ ] **Prepare the PowerPoint presentation** — summarize methodology, metrics, visuals, and key takeaways. — *Owner:*

## Open Question  
Should additional factors be considered in the research plan?  
