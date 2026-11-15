
# SmartPowerTracker – Project Methodology

## 1. Data Collection

The project uses the UCI Individual Household Electric Power Consumption dataset.

## 2. Data Preparation

A 200-minute sample from 16 December 2006 is selected.

The data preparation process includes:
- Timestamp conversion.
- Numeric data conversion.
- Missing-value handling.
- Duplicate timestamp removal.
- Invalid measurement removal.
- Saving the cleaned dataset.

## 3. Exploratory Data Analysis

The project calculates:
- Mean power consumption.
- Median power consumption.
- Minimum and maximum power.
- Standard deviation.
- Average voltage.
- First and last 40-reading averages.

Visualisations are generated to understand electricity consumption patterns.

## 4. Anomaly Detection

Two approaches are used.

### Statistical Threshold

Readings above the mean plus two standard deviations are flagged.

### Isolation Forest

A machine-learning algorithm is applied to active power and voltage.

The contamination parameter is set to 0.05.

## 5. Dashboard Development

Streamlit is used to create an interactive dashboard with four sections:
- Overview.
- Consumption Analysis.
- Anomaly Detection.
- Dataset.

## 6. Testing

Pytest is used to check dataset integrity, required columns, valid measurements, timestamps and anomaly result consistency.

## 7. Limitations and Responsible Use

The dataset is a short sample and does not establish long-term consumption patterns.

Flagged readings are indicators for further investigation, not confirmed appliance faults or energy wastage.

Household energy data should be handled with appropriate consent, privacy and security measures.
