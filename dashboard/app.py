
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ----------------------------------------
# PROJECT FILE LOCATIONS
# ----------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
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


# ----------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------

st.set_page_config(
    page_title="SmartPowerTracker",
    page_icon="⚡",
    layout="wide"
)


# ----------------------------------------
# CUSTOM DESIGN
# ----------------------------------------

st.markdown("""
<style>
    .main {
        background-color: #F5F8F5;
    }

    h1 {
        color: #174D38;
    }

    h2, h3 {
        color: #246B4B;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #DCE8DF;
    }

    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------
# LOAD DATA
# ----------------------------------------

@st.cache_data
def load_data():

    if ANOMALY_FILE.exists():
        df = pd.read_csv(ANOMALY_FILE)
    elif DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
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

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Combine both anomaly detection methods
    if (
        "statistical_anomaly" in df.columns
        and "isolation_forest_anomaly" in df.columns
    ):

        for column in [
            "statistical_anomaly",
            "isolation_forest_anomaly"
        ]:

            df[column] = (
                df[column]
                .astype(str)
                .str.lower()
                .isin(["true", "1"])
            )

        df["any_anomaly"] = (
            df["statistical_anomaly"]
            | df["isolation_forest_anomaly"]
        )

    elif "statistical_anomaly" in df.columns:

        df["any_anomaly"] = (
            df["statistical_anomaly"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        )

    else:

        df["any_anomaly"] = False

    return df


df = load_data()


# ----------------------------------------
# SIDEBAR
# ----------------------------------------

st.sidebar.title("⚡ SmartPowerTracker")

st.sidebar.write(
    "Household Electricity Monitoring System"
)

st.sidebar.divider()

st.sidebar.subheader("Dashboard Navigation")

page = st.sidebar.radio(
    "Select section",
    [
        "Overview",
        "Consumption Analysis",
        "Anomaly Detection",
        "Dataset"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Project: SmartPowerTracker"
)

st.sidebar.caption(
    "Built using Python and Streamlit"
)


# ----------------------------------------
# CHECK DATA
# ----------------------------------------

if df is None or df.empty:

    st.error(
        "Dataset not found or contains no valid readings. "
        "Run the data cleaning and anomaly detection programs first."
    )

    st.stop()


# ----------------------------------------
# DASHBOARD HEADER
# ----------------------------------------

st.title("⚡ SmartPowerTracker")

st.write(
    "Smart Household Electricity Monitoring "
    "and Anomaly Detection Dashboard"
)

st.caption(
    "Dataset: Individual Household Electric Power Consumption "
    "| UCI Machine Learning Repository"
)

st.divider()


# ----------------------------------------
# OVERVIEW PAGE
# ----------------------------------------

if page == "Overview":

    st.header("Electricity Usage Overview")

    total_readings = len(df)

    average_power = df[
        "Global_active_power"
    ].mean()

    maximum_power = df[
        "Global_active_power"
    ].max()

    average_voltage = df["Voltage"].mean()

    anomaly_count = int(
        df["any_anomaly"].sum()
    )

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
        anomaly_count
    )

    st.subheader("Electricity Consumption Trend")

    chart_data = df.set_index("timestamp")[
        ["Global_active_power"]
    ]

    st.line_chart(
        chart_data,
        height=350
    )

    st.subheader("Voltage Overview")

    st.metric(
        "Average Voltage",
        f"{average_voltage:.2f} V"
    )

    st.caption(
        "All metrics describe the selected dataset sample, "
        "not long-term household consumption."
    )


# ----------------------------------------
# CONSUMPTION ANALYSIS PAGE
# ----------------------------------------

elif page == "Consumption Analysis":

    st.header("Consumption Analysis")

    st.subheader("Power Distribution")

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(
        df["Global_active_power"],
        bins=20,
        edgecolor="black"
    )

    ax.set_xlabel("Active Power (kW)")
    ax.set_ylabel("Number of Readings")
    ax.set_title("Distribution of Electricity Consumption")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Power Statistics")

    statistics = df[
        "Global_active_power"
    ].describe().to_frame(
        name="Active Power (kW)"
    )

    st.dataframe(
        statistics,
        use_container_width=True
    )

    st.subheader("Voltage Statistics")

    st.write(
        f"Minimum Voltage: {df['Voltage'].min():.2f} V"
    )

    st.write(
        f"Maximum Voltage: {df['Voltage'].max():.2f} V"
    )

    st.write(
        f"Average Voltage: {df['Voltage'].mean():.2f} V"
    )


# ----------------------------------------
# ANOMALY DETECTION PAGE
# ----------------------------------------

elif page == "Anomaly Detection":

    st.header("Unusual Electricity Readings")

    statistical_count = 0
    isolation_count = 0

    if "statistical_anomaly" in df.columns:

        statistical_count = int(
            df["statistical_anomaly"].sum()
        )

    if "isolation_forest_anomaly" in df.columns:

        isolation_count = int(
            df["isolation_forest_anomaly"].sum()
        )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Unusual Readings",
        int(df["any_anomaly"].sum())
    )

    col2.metric(
        "Statistical Method",
        statistical_count
    )

    col3.metric(
        "Isolation Forest",
        isolation_count
    )

    st.subheader("Anomaly Detection Graph")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df["timestamp"],
        df["Global_active_power"],
        label="Electricity Power"
    )

    unusual = df[df["any_anomaly"]]

    ax.scatter(
        unusual["timestamp"],
        unusual["Global_active_power"],
        marker="x",
        s=60,
        label="Flagged Readings"
    )

    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Active Power (kW)")
    ax.legend()
    ax.set_title("Electricity Anomaly Detection")

    plt.xticks(rotation=30)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Unusual Reading Details")

    if not unusual.empty:

        st.dataframe(
            unusual[
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
            "No readings were flagged by the selected detection methods."
        )

    st.info(
        "These are algorithmically flagged observations, "
        "not confirmed appliance faults or energy wastage."
    )


# ----------------------------------------
# DATASET PAGE
# ----------------------------------------

elif page == "Dataset":

    st.header("Electricity Dataset")

    st.write(
        "Explore the cleaned household electricity readings."
    )

    st.dataframe(
        df,
        use_container_width=True,
        height=400
    )

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Dataset CSV",
        data=csv_data,
        file_name="smartpowertracker_results.csv",
        mime="text/csv"
    )


# ----------------------------------------
# FOOTER
# ----------------------------------------

st.divider()

st.caption(
    "SmartPowerTracker | Mini Project | "
    "Python • Pandas • Matplotlib • Scikit-learn • Streamlit"
)
