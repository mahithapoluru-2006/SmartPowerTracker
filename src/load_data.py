from pathlib import Path
import pandas as pd

# Dataset file locations
RAW_FILE = Path("data/raw/household_power_consumption.txt")

OUTPUT_FILE = Path(
    "data/processed/household_power_sample.csv"
)


def load_and_prepare_data():

    # Check whether the dataset exists
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}"
        )

    print("Loading household electricity dataset...")

    # Read original dataset
    df = pd.read_csv(
        RAW_FILE,
        sep=";",
        na_values=["?"],
        low_memory=False
    )

    # Remove unwanted spaces from column names
    df.columns = df.columns.str.strip()

    # Convert required columns into numeric values
    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"],
        errors="coerce"
    )

    df["Voltage"] = pd.to_numeric(
        df["Voltage"],
        errors="coerce"
    )

    # Create timestamp
    dates = pd.to_datetime(
        df["Date"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    times = df["Time"].astype(str).str.strip()

    timestamps = pd.to_datetime(
        dates.dt.strftime("%Y-%m-%d") + " " + times,
        errors="coerce"
    )

    df["timestamp"] = timestamps

    # Select our project time period
    start_time = pd.Timestamp("2006-12-16 17:24")
    end_time = pd.Timestamp("2006-12-16 20:43")

    sample = df.loc[
        (df["timestamp"] >= start_time) &
        (df["timestamp"] <= end_time),
        ["timestamp", "Global_active_power", "Voltage"]
    ].copy()

    # Remove missing values in selected columns
    sample = sample.dropna()

    # Sort readings by time
    sample = sample.sort_values("timestamp")

    # Create output folder
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save cleaned dataset
    sample.to_csv(OUTPUT_FILE, index=False)

    # Display summary
    print("\nDATASET LOADED SUCCESSFULLY!")

    print("Original dataset rows:", len(df))
    print("Selected sample rows:", len(sample))

    print("\nFirst five readings:")
    print(sample.head())

    print("\nDataset information:")
    print(sample.info())

    print("\nMissing values:")
    print(sample.isnull().sum())

    print("\nAverage active power:",
          round(sample["Global_active_power"].mean(), 2),
          "kW")

    print("\nMaximum active power:",
          round(sample["Global_active_power"].max(), 2),
          "kW")

    if len(sample) != 200:
        print("\nWARNING: Expected 200 readings, but found",
              len(sample))

    print("\nCleaned dataset saved at:", OUTPUT_FILE)


if __name__ == "__main__":
    load_and_prepare_data()
