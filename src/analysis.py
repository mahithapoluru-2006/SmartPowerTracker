
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = Path("data/processed/household_power_cleaned.csv")
REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"


def perform_analysis():

    print("Starting exploratory data analysis...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found. Run clean_data.py first."
        )

    # Load cleaned data
    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    power = df["Global_active_power"]
    voltage = df["Voltage"]

    # Create output folders
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Basic statistics
    print("\n========== BASIC STATISTICS ==========")

    print("Total readings:", len(df))
    print("Average power:", round(power.mean(), 3), "kW")
    print("Median power:", round(power.median(), 3), "kW")
    print("Minimum power:", round(power.min(), 3), "kW")
    print("Maximum power:", round(power.max(), 3), "kW")
    print("Standard deviation:", round(power.std(), 3), "kW")
    print("Average voltage:", round(voltage.mean(), 3), "V")

    # 2. Find peak consumption
    peak_index = power.idxmax()
    peak_reading = df.loc[peak_index]

    print("\n========== PEAK CONSUMPTION ==========")

    print("Peak power:",
          round(peak_reading["Global_active_power"], 3), "kW")

    print("Peak timestamp:",
          peak_reading["timestamp"])

    print("Voltage at peak:",
          round(peak_reading["Voltage"], 3), "V")

    # 3. Compare first 40 and last 40 readings
    first_40_mean = power.head(40).mean()
    last_40_mean = power.tail(40).mean()

    print("\n========== CONSUMPTION COMPARISON ==========")

    print("First 40 readings average:",
          round(first_40_mean, 3), "kW")

    print("Last 40 readings average:",
          round(last_40_mean, 3), "kW")

    # 4. Readings below 4 kW
    below_4 = (power < 4).sum()
    percentage_below_4 = (below_4 / len(df)) * 100

    print("\nReadings below 4 kW:",
          below_4)

    print("Percentage below 4 kW:",
          round(percentage_below_4, 2), "%")

    # 5. Hourly average consumption
    df["hour"] = df["timestamp"].dt.floor("h")

    hourly_summary = (
        df.groupby("hour")["Global_active_power"]
        .mean()
        .reset_index()
    )

    hourly_summary.to_csv(
        REPORT_DIR / "hourly_power_summary.csv",
        index=False
    )

    # 6. Save complete statistical summary
    summary = pd.DataFrame({
        "Metric": [
            "Total readings",
            "Average power (kW)",
            "Median power (kW)",
            "Minimum power (kW)",
            "Maximum power (kW)",
            "Average voltage (V)",
            "First 40 average (kW)",
            "Last 40 average (kW)",
            "Readings below 4 kW"
        ],
        "Value": [
            len(df),
            power.mean(),
            power.median(),
            power.min(),
            power.max(),
            voltage.mean(),
            first_40_mean,
            last_40_mean,
            below_4
        ]
    })

    summary.to_csv(
        REPORT_DIR / "analysis_summary.csv",
        index=False
    )

    # 7. Electricity consumption over time
    plt.figure(figsize=(12, 5))

    plt.plot(
        df["timestamp"],
        power,
        label="Active Power"
    )

    plt.axhline(
        power.mean(),
        linestyle="--",
        label="Average Power"
    )

    plt.title("Household Electricity Consumption Over Time")
    plt.xlabel("Time")
    plt.ylabel("Active Power (kW)")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "power_consumption.png",
        dpi=150
    )

    plt.close()

    # 8. Power distribution histogram
    plt.figure(figsize=(9, 5))

    plt.hist(power, bins=20)

    plt.title("Distribution of Household Power Consumption")
    plt.xlabel("Active Power (kW)")
    plt.ylabel("Number of Readings")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "power_distribution.png",
        dpi=150
    )

    plt.close()

    # 9. Hourly average power chart
    plt.figure(figsize=(10, 5))

    plt.bar(
        hourly_summary["hour"].astype(str),
        hourly_summary["Global_active_power"]
    )

    plt.title("Average Power Consumption by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Average Power (kW)")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "hourly_consumption.png",
        dpi=150
    )

    plt.close()

    print("\n========== ANALYSIS COMPLETED ==========")

    print("Summary saved in reports/analysis_summary.csv")
    print("Hourly summary saved in reports/hourly_power_summary.csv")
    print("Graphs saved in reports/figures/")


if __name__ == "__main__":
    perform_analysis()
