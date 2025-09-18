# Backlog

## 13.09
- [ ] **Build an Excel prototype for electricity consumption forecasting (import data, run regression, produce forecast, format output).** — *Owner:* @sergeierbin

## 15.09
- [x] **Create the data model diagram and XML.** — *Owners:* @martinoland1 @sergeierbin *Owner:* @tarmogede-dev
- [x] **Import the electricity consumption JSON dataset into Python.** — *Owner:* @sergeierbin
- [x] **Import the temperature dataset into Python using the Meteostat library.** — *Owner:* @tarmogede-dev
- [x] **Perform a regression analysis in Python to assess the relationship between consumption and temperature.** — *Owner:* @martinoland1

## 16.09
- [ ] **Update the DataFrame & data model** — add `imputed` column, consumption growth, weekday, day-curve coefficient, and consumption forecast. — *Owners:* @martinoland1 @sergeierbin *Owner:* @tarmogede-dev
- [x] **Estimate temperature-independent consumption trend** — compute the baseline growth coefficient (preferably monthly; optionally daily). — *Owner:* @martinoland1
- [ ] **Analyze daily consumption curves and identify typical daily load patterns.** — *Owner:* @tarmogede-dev
- [x] **Import the temperature forecast dataset into Python using the Meteostat library.** — *Owner:* @martinoland1
- [x] **Build next-day daily forecast** — combine regression and, baseline growth; use the average-day temperature forecast. — *Owner:* @sergeierbin

## 17.09
- [x] **Compute a curve coefficient for each weekday.** — *Owner:* @sergeierbin
- [x] **Build next-day hourly forecast** — combine next-day daily forecast and day-curve. — *Owner:* @martinoland1
- [ ] **Create a business glossary and data table descriptions** — *Owner:* @tarmogede-dev
- [ ] **Describe the Excel prototype** — *Owner:* @sergeierbin
- [ ] **Test the forecasting model** — *Owner:* @tarmogede-dev
- [ ] **Investigate importing Python code into Power BI** — *Owner:* @martinoland1
- [x] **Refactor scripts to consistently use Europe/Tallinn time** — *Owner:* @sergeierbin

## 18.09
- [ ] **Validate forecast vs actuals** — compare errors and summarize. — *Owner:*
- [ ] **Create Power BI reports** — consumption comparison (year/month), day-curve comparison, temperature comparison (year/month), etc. — *Owner:*
- [ ] **Prepare the PowerPoint presentation** — summarize methodology, metrics, visuals, and key takeaways. — *Owner:*

## Open Question  
Should additional factors be considered in the research plan?  