
from pathlib import Path
import pandas as pd


# Project directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Dataset paths
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


# Test 1: Check whether cleaned dataset exists
def test_cleaned_dataset_exists():
    assert CLEAN_FILE.exists()


# Test 2: Check required columns
def test_cleaned_dataset_has_required_columns():
    df = pd.read_csv(CLEAN_FILE)

    required_columns = [
        "timestamp",
        "Global_active_power",
        "Voltage"
    ]

    for column in required_columns:
        assert column in df.columns


# Test 3: Check missing values
def test_cleaned_dataset_has_no_missing_values():
    df = pd.read_csv(CLEAN_FILE)

    required_columns = [
        "timestamp",
        "Global_active_power",
        "Voltage"
    ]

    assert not df[required_columns].isnull().any().any()


# Test 4: Check timestamp order and uniqueness
def test_timestamps_are_unique_and_ordered():
    df = pd.read_csv(CLEAN_FILE)

    timestamps = pd.to_datetime(df["timestamp"])

    assert timestamps.is_unique
    assert timestamps.is_monotonic_increasing


# Test 5: Check valid power and voltage
def test_power_and_voltage_are_valid():
    df = pd.read_csv(CLEAN_FILE)

    assert (df["Global_active_power"] >= 0).all()
    assert (df["Voltage"] >= 0).all()


# Test 6: Check anomaly results file
def test_anomaly_results_exist():
    assert ANOMALY_FILE.exists()


# Test 7: Check anomaly detection columns
def test_anomaly_results_have_detection_columns():
    df = pd.read_csv(ANOMALY_FILE)

    required_columns = [
        "statistical_anomaly",
        "isolation_forest_anomaly"
    ]

    for column in required_columns:
        assert column in df.columns


# Test 8: Check matching row counts
def test_anomaly_results_match_cleaned_dataset():
    clean_df = pd.read_csv(CLEAN_FILE)
    anomaly_df = pd.read_csv(ANOMALY_FILE)

    assert len(clean_df) == len(anomaly_df)
