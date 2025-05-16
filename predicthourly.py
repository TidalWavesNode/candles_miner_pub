import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class CandleNet(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.model(x)

def predict_24_hourly(csv_path="TVexport.csv", model_path="model.pth"):
    print("📊 Predicting the next 24 hourly candles...")

    df = pd.read_csv(csv_path)
    print("🧮 CSV Columns:", df.columns.tolist())
    df = df.fillna(method='ffill').fillna(method='bfill')

    df["candle_body"] = df["close"] - df["open"]
    df["candle_range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["close_to_open_ratio"] = df["close"] / df["open"]
    df["high_to_low_ratio"] = df["high"] / df["low"]

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    features = df[[
        "candle_body", "candle_range", "upper_wick", "lower_wick",
        "close_to_open_ratio", "high_to_low_ratio", "open", "close"
    ]].values.astype(np.float32)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    last_24 = torch.tensor(features[-24:], dtype=torch.float32)

    model = CandleNet(input_size=last_24.shape[1])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    predictions = torch.sigmoid(model(last_24)).detach().numpy().flatten()
    labels = ["Green" if p > 0.5 else "Red" for p in predictions]

    with open("predictions_hourly.txt", "w") as f:
        for i, label in enumerate(labels, 1):
            line = f"Hour {i}: {label}"
            print(line)
            f.write(line + "\n")

    print("✅ Predictions saved to predictions_hourly.txt")

if __name__ == "__main__":
    predict_24_hourly()
