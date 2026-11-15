
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Input and output locations
INPUT_FILE = Path("data/processed/household_power_cleaned.csv")

REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"

OUTPUT_FILE = REPORT_DIR / "anomaly_results.csv"


def detect_anomalies():

    print("Starting anomaly detection...")

    # Check input file
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found. Run clean_data.py first."
        )

    # Load cleaned data
    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Select features for detection
    features = [
        "Global_active_power",
        "Voltage"
    ]

    X = df[features].copy()

    # ------------------------------------------------
    # METHOD 1: STATISTICAL THRESHOLD
    # ------------------------------------------------

    mean_power = X["Global_active_power"].mean()
    std_power = X["Global_active_power"].std()

    threshold = mean_power + (2 * std_power)

    df["threshold"] = threshold

    df["statistical_anomaly"] = (
        X["Global_active_power"] > threshold
    )

    # ------------------------------------------------
    # METHOD 2: ISOLATION FOREST
    # ------------------------------------------------

    model = make_pipeline(
        StandardScaler(),
        IsolationForest(
            contamination=0.05,
            random_state=42
        )
    )

    predictions = model.fit_predict(X)

    # Isolation Forest returns:
    #  1 = Normal
    # -1 = Anomaly

    df["isolation_forest_anomaly"] = (
        predictions == -1
    )

    # Convert flags into readable labels
    df["statistical_status"] = df[
        "statistical_anomaly"
    ].map({
        True: "Unusual",
        False: "Normal"
    })

    df["ml_status"] = df[
        "isolation_forest_anomaly"
    ].map({
        True: "Unusual",
        False: "Normal"
    })

    # ------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    # ------------------------------------------------
    # DISPLAY SUMMARY
    # ------------------------------------------------

    statistical_count = int(
        df["statistical_anomaly"].sum()
    )

    isolation_count = int(
        df["isolation_forest_anomaly"].sum()
    )

    print("\n========== ANOMALY RESULTS ==========")

    print("Total readings:", len(df))

    print("Average power:",
          round(mean_power, 3), "kW")

    print("Standard deviation:",
          round(std_power, 3), "kW")

    print("Statistical threshold:",
          round(threshold, 3), "kW")

    print("\nStatistical anomalies:",
          statistical_count)

    print("Isolation Forest anomalies:",
          isolation_count)

    print("\nStatistical anomaly readings:")

    print(
        df.loc[
            df["statistical_anomaly"],
            ["timestamp", "Global_active_power", "Voltage"]
        ].to_string(index=False)
    )

    print("\nIsolation Forest anomaly readings:")

    print(
        df.loc[
            df["isolation_forest_anomaly"],
            ["timestamp", "Global_active_power", "Voltage"]
        ].to_string(index=False)
    )

    # ------------------------------------------------
    # CREATE VISUALIZATION
    # ------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["timestamp"],
        df["Global_active_power"],
        label="Active Power",
        linewidth=1.5
    )

    # Statistical anomalies
    statistical_points = df[
        df["statistical_anomaly"]
    ]

    plt.scatter(
        statistical_points["timestamp"],
        statistical_points["Global_active_power"],
        label="Statistical Anomalies",
        marker="x",
        s=65
    )

    # Isolation Forest anomalies
    ml_points = df[
        df["isolation_forest_anomaly"]
    ]

    plt.scatter(
        ml_points["timestamp"],
        ml_points["Global_active_power"],
        label="Isolation Forest Anomalies",
        marker="o",
        s=45
    )

    plt.axhline(
        threshold,
        linestyle="--",
        label="Statistical Threshold"
    )

    plt.title("SmartPowerTracker - Anomaly Detection")
    plt.xlabel("Timestamp")
    plt.ylabel("Active Power (kW)")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "anomaly_detection.png",
        dpi=150
    )

    plt.close()

    print("\nAnomaly detection completed!")

    print("Results saved to:", OUTPUT_FILE)

    print(
        "Graph saved to:",
        FIGURE_DIR / "anomaly_detection.png"
    )

    return df


if __name__ == "__main__":
    detect_anomalies()
