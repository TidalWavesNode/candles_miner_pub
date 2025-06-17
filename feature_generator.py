import pandas as pd

def generate_features(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    print(f"📥 Loaded {len(df)} rows from {input_csv}")

    # Ensure required columns exist
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"❌ Missing one of the required columns: {required}")

    # Feature engineering
    df["candle_body"] = df["close"] - df["open"]
    df["candle_range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    # Ratio features
    df["close_to_open_ratio"] = df["close"] / df["open"]
    df["high_to_low_ratio"] = df["high"] / df["low"]

    # Handle bad values
    df = df.replace([float('inf'), -float('inf')], pd.NA).dropna()

    # Reorder output columns if they exist
    output_cols = [
        "time", "open", "high", "low", "close", "volume",
        "candle_body", "candle_range", "upper_wick", "lower_wick",
        "close_to_open_ratio", "high_to_low_ratio"
    ]
    df = df[[col for col in output_cols if col in df.columns]]

    # Save output
    df.to_csv(output_csv, index=False)
    print(f"✅ Features saved to {output_csv}")
    print(f"🧪 Final columns: {list(df.columns)}")
    print(f"🧾 Total rows after cleaning: {len(df)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate engineered candle features.")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file (must include open, high, low, close)")
    parser.add_argument("--output", type=str, required=True, help="Output CSV with features")
    args = parser.parse_args()
    generate_features(args.input, args.output)
