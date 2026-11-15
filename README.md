# ⚡ SmartPowerTracker
## Smart Home Energy Watch

An intelligent household electricity monitoring prototype using Python, data analytics, machine learning and Streamlit.

---

## 1. Project Overview

SmartPowerTracker is a mini project designed to analyse household electricity consumption and identify unusual electricity readings.

It uses data analysis and anomaly detection techniques to help users understand consumption patterns and investigate unusual observations.

The project provides an interactive dashboard to display electricity statistics, visualisations and anomaly detection results.

## 2. Problem Statement

Households may not always notice changes or unusual patterns in electricity consumption.

SmartPowerTracker explores how data analytics and machine learning can help monitor electricity usage and highlight readings that may require further investigation.

## 3. Project Objectives

- Analyse household electricity consumption.
- Calculate electricity usage statistics.
- Visualise consumption patterns.
- Identify unusual electricity readings.
- Compare statistical and machine-learning anomaly detection.
- Display results through an interactive dashboard.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Matplotlib | Data visualisation |
| Scikit-learn | Machine learning |
| Streamlit | Interactive dashboard |
| Pytest | Automated testing |
| GitHub Codespaces | Development environment |
| Git | Version control |

## 5. Dataset

**Dataset:** Individual Household Electric Power Consumption

**Source:** UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The project uses 200 consecutive minute-level readings from 16 December 2006, between 17:24 and 20:43.

Selected attributes:

- Timestamp
- Global_active_power (kW)
- Voltage (V)

## 6. Project Workflow

1. Load the original household electricity dataset.
2. Select the required time period and columns.
3. Clean missing and duplicate records.
4. Analyse electricity consumption.
5. Generate statistical summaries and graphs.
6. Detect unusual readings using two methods.
7. Display the results in the Streamlit dashboard.
8. Validate the project using automated tests.

## 7. Anomaly Detection Methods

### Statistical Threshold Method

A reading is flagged when its active power exceeds:

Threshold = Mean + (2 × Standard Deviation)

### Isolation Forest

Isolation Forest is used to identify unusual observations using:

- Global active power
- Voltage

The contamination parameter is configured to 0.05.

Both methods are compared to explore their detection results.

## 8. Dashboard Features

The Streamlit dashboard contains four sections:

### Overview
- Total readings
- Average power
- Peak power
- Unusual readings
- Date-based filtering
- Electricity consumption graph

### Consumption Analysis
- Power consumption distribution
- Power statistics
- Voltage statistics
- Consumption over time

### Anomaly Detection
- Statistical anomaly count
- Isolation Forest anomaly count
- Combined flagged readings
- Unusual reading details

### Dataset
- Dataset preview
- Dataset information
- CSV download option

## 9. Project Structure

```text
SmartPowerTracker/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── __init__.py
│   ├── load_data.py
│   ├── clean_data.py
│   ├── analysis.py
│   └── anomaly_detection.py
│
├── dashboard/
│   └── app.py
│
├── tests/
│   └── test_analysis.py
│
├── docs/
│   └── project_methodology.md
│
├── reports/
│   ├── analysis_summary.csv
│   ├── hourly_power_summary.csv
│   ├── anomaly_results.csv
│   └── figures/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 10. Installation and Setup

Open the project in GitHub Codespaces.

Create a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Download and extract the original dataset from the UCI repository before running the data-processing scripts.

## 11. Run the Project

Run the Python scripts in the following order:

```bash
python src/load_data.py
python src/clean_data.py
python src/analysis.py
python src/anomaly_detection.py
```

Start the dashboard:

```bash
streamlit run dashboard/app.py --server.address 0.0.0.0
```

Open port 8501 in GitHub Codespaces to view the dashboard.

## 12. Run Automated Tests

Execute:

```bash
pytest -v
```

The tests check dataset availability, required columns, missing values, timestamp consistency, valid measurements and anomaly result consistency.

## 13. Limitations

- The analysis uses only 200 consecutive minutes of household data.
- The sample cannot represent long-term household electricity usage.
- An unusual reading does not necessarily indicate electricity wastage.
- An anomaly does not prove an appliance fault.
- The Isolation Forest contamination value is an assumed parameter.
- Appliance-level identification is not possible with the selected columns.
- Actual energy savings have not been measured or validated.

## 14. Future Enhancements

- Analyse longer periods of electricity consumption.
- Compare additional anomaly detection algorithms.
- Introduce configurable alert thresholds.
- Integrate appliance-level electricity readings.
- Evaluate model performance using labelled data.
- Conduct a real-world household pilot.

## 15. Expected Impact

SmartPowerTracker demonstrates how data analytics and machine learning can support household electricity monitoring.

It provides a foundation for exploring more informed electricity usage and investigating unusual consumption patterns.

---

**Project Name:** SmartPowerTracker  
**Project Title:** Smart Home Energy Watch  
**Category:** Data Analytics and Machine Learning
