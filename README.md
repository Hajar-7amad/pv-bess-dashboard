# PV-BESS Explainable Analytics & Battery Performance Platform

An interactive Streamlit application that turns the methodology of

> H. Albutuwaybah, H. Bassi, A. Alghamdi, *"An Explainable Machine Learning Tool for Capacity-Normalized
> Battery Charging Analysis in Photovoltaic Energy Storage Systems"* (ICAT2I)

into an engineering tool: an analytical layer above the EPower Monitor EMS for monitoring, charging analysis,
battery comparison, XGBoost prediction and SHAP explainability. **It is read-only: it never issues battery
control commands.**

| System (from the paper) | |
|---|---|
| PV array | 260 kWp, 1000 VDC bus |
| EMS | EPower Monitor |
| Battery 1 | Lead-Acid, OPzV VRLA Gel Deep Cycle, **864 kWh** |
| Battery 2 | Lithium-Ion, Huawei LUNA2000-200 (LFP), **510 kWh** |
| Future | e.g. Vanadium Redox Flow Battery, user-defined capacity (Settings → Add battery) |

---

## Installation

Python 3.10 or newer.

```bash
cd pv_bess_dashboard
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The browser opens at http://localhost:8501. To try every page before loading real data, open
**Data Import & Mapping → Load SYNTHETIC demo data**. A banner then marks every page as synthetic; these
values are artificial and must never be reported.

## Tests

```bash
python -m pytest -q tests/
```

`tests/test_pipeline.py` checks the column mapper, R_cn, the charging filter, events, model training, SHAP
and comparison logic. `tests/test_app.py` renders every page headlessly, trains both models and builds the
report. Both use only synthetic data.

---

## Workflow (same sequence as the paper)

1. **Import** – upload CSV files per battery plus shared plant files; confirm the timestamp and column mapping.
2. **Synchronize** – resample to a common interval (default hourly, as in the paper), nearest-timestamp
   matching with a tolerance, or exact matching.
3. **Preprocess** – sort, remove duplicate timestamps, flag invalid values (SOC outside 0–100 %, impossible
   negatives, out-of-range readings; screened at native resolution *before* resampling), detect outliers
   (retain or remove), report missing values. Every step is logged with observation counts; nothing is
   removed silently.
4. **Charging selection** – `P_charge > 0` **and** `SOC < 80 %` (both thresholds configurable).
5. **Target** – `normalized_charging_rate` = R_cn = P_charge / E_nominal [kW/kWh = h⁻¹].
6. **Model** – one XGBoost model per battery (never pooled), 80/20 hold-out (chronological by default, or
   random), 5-fold cross-validation (shuffled K-fold, as a stability check, or time-series split), R² / MAE /
   RMSE.
7. **Explain** – TreeSHAP: global mean |SHAP|, beeswarm, dependence, waterfall per timestamp, and
   cross-battery ranking comparison.

## Pages

| Page | Contents |
|---|---|
| Data Import & Mapping | uploads per battery and shared, automatic column matching with manual override, unit scale factors, timestamp options, synchronization statistics, CSV guidance |
| System Overview | instantaneous KPI cards, energy-flow schematic with live values, dataset summary, recent operation |
| Data Explorer | time-series analysis (any period, any parameters, raw/5 min/15 min/30 min/hourly/daily), chart explorer, correlation matrix |
| Custom Plot Builder | chart type, X, Y, extra Y-series, second axis, colour variable, batteries, period, SOC range, mode filter, aggregation, title, axis labels, log scales, legend, PNG/SVG/HTML/CSV export |
| Data Quality | missing values, availability heatmap, sampling intervals per file, gaps, invalid values, outliers, frozen sensors, sudden jumps, warnings, battery data-quality comparison |
| Battery Analysis | SOC, power, R_cn time series, daily energy, cumulative throughput, relationship scatter plots, operating-state and SOC-band shares |
| Charging Analysis (R_cn) | formula, preprocessing log, filter funnel, operating-state toggles, R_cn statistics and plots, charging-event table and inspector |
| Battery Performance Comparison | side-by-side KPI cards, scorecard with safe relative differences, normalized indicators, time series, box/histogram/CDF, scatter, **matched operating conditions**, daily/weekly/monthly trends, energy balance, utilization, charging-event comparison, ML comparison, descriptive interpretation |
| Machine Learning | predictor selection with leakage checks, validation design, training with progress, metrics, metadata, actual vs predicted, residuals, error over time, CV folds, feature importance, train vs test |
| SHAP Explainability | global importance, beeswarm, dependence, individual prediction (waterfall), battery comparison with Top-N common features |
| Report / Export | every table as CSV, ZIP bundle, self-contained HTML engineering report |
| Live Monitor (preview) | auto-refresh architecture (replay of historical data today) |
| Settings | battery registry (add/remove technologies, capacities), every analysis parameter, JSON save/load |

## Scientific safeguards built in

* **R_cn definition is fixed** in `modules/normalized_rate.py`; only E_nominal and the SOC cutoff are editable.
* **Target leakage:** charge power, signed battery power, interval charge energy and R_cn are blocked as
  predictors. Discharge power and near-perfectly correlated features raise warnings. When PV, load **and**
  grid power are all selected, the app fits the energy-balance identity and warns if they can reconstruct
  the charge power (check this on real data: if it holds, one of the three should be dropped).
* **SHAP is labelled as association, not causality** on every SHAP view and in the report.
* **No automatic chemistry ranking:** the scorecard shows separate metrics; a preference tally exists only
  if the user enables it in Settings and defines each metric's preferred direction.
* **Fair-comparison notice** on the comparison page; relative differences are suppressed (with reason) for
  %-quantities, °C and R²; coverage-dependent metrics are flagged.
* **Unavailable variables are shown as unavailable**, never estimated.
* Data stages are kept distinct: raw synchronized → processed → filtered charging → training/testing →
  predictions.

## CSV format guidance

* One timestamp column (ISO, `DD/MM/YYYY HH:MM`, etc.) **or** separate date and time columns.
  Day-first vs month-first is detected automatically and can be overridden; an explicit strftime format can
  be entered.
* Delimiter (`,` `;` tab `|`), encoding (UTF-8/UTF-16/CP1252) and decimal comma are detected.
  Leading metadata lines before the header are skipped.
* Expected units: kW, kWh, W/m², °C, m/s, degrees, %. Use the scale factor to convert (W → kW: 0.001).
* Parameters can be spread over several files with different sampling intervals.
* Put **battery-specific** files under the battery's tab and **plant-level** files (weather, PV, load, grid)
  under *Shared plant files*. Shared data are clipped to each battery's own monitoring period, because the
  banks were dispatched sequentially.
* A single signed battery-power column can replace separate charge/discharge columns (sign convention in
  Settings).

Example (any column names work; the mapper suggests matches):

```text
Date Time;State of Charge (%);Charging Power (kW);Discharging Power (kW)
06/01/2025 00:00;45.0;0.0;12.4
```

### Standardized schema

| Key | Quantity | Unit | Resampling |
|---|---|---|---|
| `solar_irradiance` | Solar irradiance | W/m² | mean |
| `ambient_temperature` | Ambient temperature | °C | mean |
| `pv_cell_temperature` | PV cell temperature | °C | mean |
| `wind_speed` | Wind speed | m/s | mean |
| `wind_direction` | Wind direction | ° | circular mean |
| `battery_soc` | Battery SOC | % | mean |
| `battery_charge_power` | Battery charge power | kW | mean |
| `battery_discharge_power` | Battery discharge power | kW | mean |
| `battery_power` | Signed battery power (optional) | kW | mean |
| `pv_power` | PV power | kW | mean |
| `pv_energy` | PV energy (cumulative register) | kWh | last value, never averaged |
| `total_energy` | Total energy (cumulative register) | kWh | last value, never averaged |
| `load_power` | Load power | kW | mean |
| `grid_power` | Grid power (+ import by default) | kW | mean |
| `inverter_temperature` | Inverter temperature | °C | mean |
| `normalized_charging_rate` | **R_cn** (derived) | h⁻¹ | mean |

## Configuration

All defaults live in `config.py` (`DEFAULT_SETTINGS`, `DEFAULT_BATTERIES`). `settings_example.json` shows a
full configuration including an example VRFB entry (its 400 kWh capacity is a placeholder). Load or save
JSON configurations on the Settings page.

## Architecture

```text
pv_bess_dashboard/
├── app.py                    entry point, navigation, sidebar
├── config.py                 system, battery registry, schema, defaults
├── requirements.txt
├── settings_example.json
├── .streamlit/config.toml    theme
├── data/sample/              SYNTHETIC demo files only (see its README)
├── models/                   saved XGBoost models (+ metadata JSON)
├── outputs/
├── modules/
│   ├── data_sources.py       CSVDataSource (implemented); Database / API / LiveEMS placeholders
│   ├── data_loader.py        robust CSV reading, timestamp detection, column mapping
│   ├── synchronization.py    resample / nearest / exact merge, physically correct aggregation
│   ├── preprocessing.py      logged cleaning pipeline, derived variables
│   ├── normalized_rate.py    R_cn, operating modes, paper charging filter
│   ├── charging_events.py    event detection and statistics
│   ├── battery_analysis.py   KPIs, durations, SOC bands, period summaries, energy balance
│   ├── battery_comparison.py scorecard, matched conditions, interpretation, disclaimers
│   ├── ml_model.py           leakage checks, dataset, XGBoost training, CV, metrics
│   ├── shap_analysis.py      TreeSHAP, rankings, comparison, plots
│   ├── data_quality.py       coverage, gaps, frozen sensors, jumps, warnings
│   ├── plotting.py           Plotly figure factory
│   ├── export.py / reporting.py
│   ├── state.py              session state + cached end-to-end pipeline
│   ├── explorer.py / ui.py   shared controls and components
│   └── synthetic.py          SYNTHETIC demo generator
├── pages/                    one file per page
└── tests/
```

**Caching:** the pipeline result for each battery is memoised on a fingerprint of the file bytes, mappings,
relevant settings and capacity; models are retrained only when the data, predictors or ML settings change.

**Adding a battery technology:** Settings → *Add a battery technology* (name, capacity). It gets its own
upload tab, its own pipeline, its own model and appears in every comparison. No code change needed.

**Connecting the EMS live:** implement `LiveEMSDataSource.latest()` / `read_table()` in
`modules/data_sources.py` (Modbus-TCP / OPC-UA / MQTT / REST). The Live Monitor page already polls the source
on the configured refresh interval.

## Figure export

Every Plotly chart has a toolbar camera icon that downloads PNG or SVG (choose in Settings → *Data quality &
display*) and an *Interactive HTML* button. The Plot Builder can also render static images server-side when
Kaleido and a Chrome/Chromium browser are installed. SHAP beeswarm and waterfall plots download as PNG/SVG.
