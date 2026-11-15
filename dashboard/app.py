
from pathlib import Path

import pandas as pd
import streamlit as st


# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="SmartPowerTracker",
    page_icon="⚡",
    layout="wide"
)

# Project file paths
CLEAN_FILE = Path(
    "data/processed/household_power_cleaned.csv"
)

ANOMALY_FILE = Path(
    "reports/anomaly_results.csv"
)


# -----------------------------------------
# LOAD DATA
# -----------------------------------------

@st.cache_data
def load_data():

    if ANOMALY_FILE.exists():
        df = pd.read_csv(ANOMALY_FILE)
    elif CLEAN_FILE.exists():
        df = pd.read_csv(CLEAN_FILE)
    else:
        return None

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"],
        errors="coerce"
    )

    df["Voltage"] = pd.to_numeric(
        df["Voltage"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "timestamp",
            "Global_active_power",
            "Voltage"
        ]
    )

    return df.sort_values("timestamp")


# -----------------------------------------
# DASHBOARD HEADER
# -----------------------------------------

st.title("⚡ SmartPowerTracker")

st.subheader(
    "Smart Household Electricity Monitoring Dashboard"
)

st.markdown(
    "Monitor electricity consumption, "
    "understand usage patterns and identify "
    "unusual power readings."
)

st.divider()


# -----------------------------------------
# LOAD PROJECT DATA
# -----------------------------------------

df = load_data()

if df is None or df.empty:

    st.error(
        "Dataset not found. Please run the "
        "data loading, cleaning and anomaly "
        "detection programs first."
    )

    st.stop()


# -----------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------

st.sidebar.header("Dashboard Filters")

min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()

selected_dates = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(selected_dates, (tuple, list)):

    if len(selected_dates) == 2:

        start_date, end_date = selected_dates

        filtered_df = df[
            (df["timestamp"].dt.date >= start_date)
            &
            (df["timestamp"].dt.date <= end_date)
        ]

    else:
        filtered_df = df[
            df["timestamp"].dt.date == selected_dates[0]
        ]

else:

    filtered_df = df[
        df["timestamp"].dt.date == selected_dates
    ]


if filtered_df.empty:

    st.warning("No readings found for this date range.")
    st.stop()


# -----------------------------------------
# KEY PERFORMANCE INDICATORS
# -----------------------------------------

average_power = filtered_df[
    "Global_active_power"
].mean()

maximum_power = filtered_df[
    "Global_active_power"
].max()

minimum_power = filtered_df[
    "Global_active_power"
].min()

average_voltage = filtered_df[
    "Voltage"
].mean()

total_readings = len(filtered_df)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Average Power",
    f"{average_power:.2f} kW"
)

col2.metric(
    "Peak Power",
    f"{maximum_power:.2f} kW"
)

col3.metric(
    "Average Voltage",
    f"{average_voltage:.2f} V"
)

col4.metric(
    "Total Readings",
    total_readings
)


st.divider()


# -----------------------------------------
# ELECTRICITY CONSUMPTION CHART
# -----------------------------------------

st.subheader("Electricity Consumption Over Time")

chart_data = filtered_df.set_index("timestamp")[
    ["Global_active_power"]
]

st.line_chart(
    chart_data,
    y="Global_active_power",
    x_label="Time",
    y_label="Active Power (kW)"
)


# -----------------------------------------
# POWER DISTRIBUTION
# -----------------------------------------

st.subheader("Power Consumption Distribution")

histogram = pd.cut(
    filtered_df["Global_active_power"],
    bins=10
).value_counts().sort_index()

histogram_data = pd.DataFrame({
    "Power Range": histogram.index.astype(str),
    "Readings": histogram.values
}).set_index("Power Range")

st.bar_chart(histogram_data)


# -----------------------------------------
# ANOMALY DETECTION
# -----------------------------------------

st.divider()

st.subheader("Unusual Consumption Detection")

if "statistical_anomaly" in filtered_df.columns:

    statistical_count = int(
        filtered_df["statistical_anomaly"]
        .astype(bool)
        .sum()
    )

    st.metric(
        "Statistical Unusual Readings",
        statistical_count
    )

    unusual_df = filtered_df[
        filtered_df["statistical_anomaly"].astype(bool)
    ]

    if not unusual_df.empty:

        st.warning(
            "Some readings are above the statistical "
            "power threshold. Review these readings."
        )

        st.dataframe(
            unusual_df[
                [
                    "timestamp",
                    "Global_active_power",
                    "Voltage"
                ]
            ],
            use_container_width=True
        )

    else:

        st.success(
            "No readings exceeded the statistical threshold."
        )

else:

    st.info(
        "Run the anomaly detection program to view alerts."
    )


if "isolation_forest_anomaly" in filtered_df.columns:

    ml_count = int(
        filtered_df["isolation_forest_anomaly"]
        .astype(bool)
        .sum()
    )

    st.metric(
        "Isolation Forest Unusual Readings",
        ml_count
    )


# -----------------------------------------
# COMPLETE DATA TABLE
# -----------------------------------------

st.divider()

st.subheader("Electricity Reading Details")

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)


# -----------------------------------------
# DOWNLOAD RESULTS
# -----------------------------------------

csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Filtered Electricity Data",
    data=csv_data,
    file_name="smart_power_readings.csv",
    mime="text/csv"
)


# -----------------------------------------
# FOOTER
# -----------------------------------------

st.divider()

st.caption(
    "SmartPowerTracker | Household Electricity "
    "Monitoring and Anomaly Detection Project"
)

st.caption(
    "Research dataset: UCI Individual Household "
    "Electric Power Consumption. Unusual readings "
    "are indicators for review, not proof of faults."
)
