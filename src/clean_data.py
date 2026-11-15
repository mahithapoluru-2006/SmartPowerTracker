from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/processed/household_power_sample.csv")
OUTPUT_FILE = Path("data/processed/household_power_cleaned.csv")


def clean_data():

    print("Starting data cleaning...")

    # Check input file
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    # Load dataset
    df = pd.read_csv(INPUT_FILE)

    print("\nOriginal dataset shape:", df.shape)

    # Convert timestamp into datetime format
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # Convert measurement columns into numeric values
    numeric_columns = [
        "Global_active_power",
        "Voltage"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Check missing values before cleaning
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # Remove rows with missing required values
    df = df.dropna(
        subset=["timestamp"] + numeric_columns
    )

    # Remove duplicate timestamps
    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    # Remove physically invalid negative readings
    df = df[df["Global_active_power"] >= 0]
    df = df[df["Voltage"] >= 0]

    # Sort readings by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Check final dataset
    print("\nCleaned dataset shape:", df.shape)

    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    print("\nDuplicate timestamps:",
          df["timestamp"].duplicated().sum())

    # Save cleaned data
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nData cleaning completed!")
    print("Saved cleaned data to:", OUTPUT_FILE)

    return df


if __name__ == "__main__":
    clean_data()
