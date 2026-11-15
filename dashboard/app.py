
from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

CLEAN_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "household_power_cleaned.csv"
)

ANOMALY_FILE = (
    PROJECT_DIR
    / "reports"
    / "anomaly_results.csv"
)


# --------------------------------------------------
# 2. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SmartPowerTracker",
    page_icon="⚡",
    layout="wide"
)


# --------------------------------------------------
# 3. CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        background-color: #f5f9f6;
    }

    h1 {
        color: #176b45;
    }

    h2, h3 {
        color: #247a52;
    }

    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #dce9df;
        padding: 15px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 4. LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    if ANOMALY_FILE.exists():
        df = pd.read_csv(ANOMALY_FILE)

    elif CLEAN_FILE.exists():
        df = pd.read_csv(CLEAN_FILE)

    else:
        return pd.DataFrame()

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # Convert numeric columns
    numeric_columns = [
        "Global_active_power",
        "Voltage"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove invalid rows
    df = df.dropna(
        subset=[
            "timestamp",
            "Global_active_power",
            "Voltage"
        ]
    )

    # Sort by timestamp
    df = df.sort_values("timestamp")

    # Convert anomaly columns into Boolean values
    anomaly_columns = [
        "statistical_anomaly",
        "isolation_forest_anomaly"
    ]

    for column in anomaly_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "1", "yes"])
            )

        else:
            df[column] = False

    # Identify readings flagged by either method
    df["any_anomaly"] = (
        df["statistical_anomaly"]
        | df["isolation_forest_anomaly"]
    )

    return df


df = load_data()


# --------------------------------------------------
# 5. CHECK DATA AVAILABILITY
# --------------------------------------------------

if df.empty:

    st.error(
        "No dataset found. Please run the data "
        "loading, cleaning and anomaly detection "
        "programs first."
    )

    st.stop()


# --------------------------------------------------
# 6. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("⚡ SmartPowerTracker")

st.sidebar.write(
    "Smart Home Energy Watch"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Consumption Analysis",
        "Anomaly Detection",
        "Dataset"
    ]
)


# --------------------------------------------------
# 7. MAIN TITLE
# --------------------------------------------------

st.title("⚡ SmartPowerTracker")

st.write(
    "Household Electricity Monitoring "
    "and Anomaly Detection Dashboard"
)

st.info(
    "Monitor household electricity readings, "
    "understand consumption patterns and review "
    "unusual observations."
)


# ==================================================
# PAGE 1: OVERVIEW
# ==================================================

if page == "Overview":

    st.header("Energy Consumption Overview")

    # Calculate metrics
    total_readings = len(df)

    average_power = df[
        "Global_active_power"
    ].mean()

    maximum_power = df[
        "Global_active_power"
    ].max()

    unusual_readings = int(
        df["any_anomaly"].sum()
    )

    # Display four metric cards
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Readings",
        total_readings
    )

    col2.metric(
        "Average Power",
        f"{average_power:.2f} kW"
    )

    col3.metric(
        "Peak Power",
        f"{maximum_power:.2f} kW"
    )

    col4.metric(
        "Unusual Readings",
        unusual_readings
    )

    st.divider()

    # ----------------------------------------------
    # DATE FILTER
    # ----------------------------------------------

    st.subheader("Select a Time Period")

    start_date = df["timestamp"].min().date()

    end_date = df["timestamp"].max().date()

    selected_dates = st.date_input(
        "Filter readings by date",
        value=(start_date, end_date),
        min_value=start_date,
        max_value=end_date
    )

    if len(selected_dates) == 2:

        filtered_df = df[
            (
                df["timestamp"].dt.date
                >= selected_dates[0]
            )
            &
            (
                df["timestamp"].dt.date
                <= selected_dates[1]
            )
        ]

        if not filtered_df.empty:

            st.write(
                f"Showing {len(filtered_df)} readings"
            )

            st.subheader(
                "Filtered Electricity Consumption"
            )

            st.line_chart(
                filtered_df.set_index("timestamp")[
                    ["Global_active_power"]
                ]
            )

        else:

            st.warning(
                "No readings found for this date range."
            )

    else:

        st.info(
            "Please select both a start date and an end date."
        )

    st.divider()

    # ----------------------------------------------
    # VOLTAGE INFORMATION
    # ----------------------------------------------

    st.subheader("Voltage Information")

    average_voltage = df["Voltage"].mean()

    st.metric(
        "Average Voltage",
        f"{average_voltage:.2f} V"
    )

    st.caption(
        "The displayed statistics describe the available "
        "dataset sample."
    )


# ==================================================
# PAGE 2: CONSUMPTION ANALYSIS
# ==================================================

elif page == "Consumption Analysis":

    st.header("Electricity Consumption Analysis")

    st.subheader("Power Consumption Distribution")

    st.bar_chart(
        df["Global_active_power"].value_counts(
            bins=10,
            sort=False
        )
    )

    st.subheader("Power Consumption Statistics")

    power_statistics = (
        df["Global_active_power"]
        .describe()
        .to_frame(name="Power (kW)")
    )

    st.dataframe(
        power_statistics,
        use_container_width=True
    )

    st.subheader("Voltage Statistics")

    voltage_statistics = (
        df["Voltage"]
        .describe()
        .to_frame(name="Voltage (V)")
    )

    st.dataframe(
        voltage_statistics,
        use_container_width=True
    )

    st.subheader("Power Consumption Over Time")

    st.line_chart(
        df.set_index("timestamp")[
            ["Global_active_power"]
        ]
    )


# ==================================================
# PAGE 3: ANOMALY DETECTION
# ==================================================

elif page == "Anomaly Detection":

    st.header("Anomaly Detection Results")

    statistical_count = int(
        df["statistical_anomaly"].sum()
    )

    ml_count = int(
        df["isolation_forest_anomaly"].sum()
    )

    combined_count = int(
        df["any_anomaly"].sum()
    )

    # Display anomaly metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Statistical Anomalies",
        statistical_count
    )

    col2.metric(
        "Isolation Forest Anomalies",
        ml_count
    )

    col3.metric(
        "Readings Flagged by Either Method",
        combined_count
    )

    st.divider()

    # ----------------------------------------------
    # ANOMALY GRAPH
    # ----------------------------------------------

    st.subheader(
        "Power Consumption with Unusual Readings"
    )

    normal_df = df[
        ~df["any_anomaly"]
    ]

    anomaly_df = df[
        df["any_anomaly"]
    ]

    st.line_chart(
        df.set_index("timestamp")[
            ["Global_active_power"]
        ]
    )

    if not anomaly_df.empty:

        st.subheader("Flagged Readings")

        st.dataframe(
            anomaly_df[
                [
                    "timestamp",
                    "Global_active_power",
                    "Voltage",
                    "statistical_anomaly",
                    "isolation_forest_anomaly"
                ]
            ],
            use_container_width=True
        )

    else:

        st.success(
            "No unusual readings were flagged in this dataset."
        )

    st.info(
        "An anomaly is an unusual observation identified "
        "by the selected methods. It does not automatically "
        "indicate electricity wastage, an appliance fault "
        "or a safety problem."
    )


# ==================================================
# PAGE 4: DATASET
# ==================================================

elif page == "Dataset":

    st.header("Household Electricity Dataset")

    st.write(
        "Explore the cleaned household electricity readings "
        "used by SmartPowerTracker."
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        len(df)
    )

    col2.metric(
        "Total Columns",
        len(df.columns)
    )

    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    # ----------------------------------------------
    # CSV DOWNLOAD
    # ----------------------------------------------

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Dataset as CSV",
        data=csv_data,
        file_name="smartpowertracker_results.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# 8. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "SmartPowerTracker | Smart Home Energy Watch | "
    "Python • Pandas • Scikit-learn • Streamlit"
)
