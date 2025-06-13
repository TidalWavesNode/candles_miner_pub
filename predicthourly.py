import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class CandleNet(nn.Module):
    def __init__(self, input_size=10):
        super(CandleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

class TemperatureScaledModel(nn.Module):
    def __init__(self, base_model, temperature=2.0):
        super().__init__()
        self.base_model = base_model
        self.temperature = temperature

    def forward(self, x):
        logits = self.base_model(x)
        return logits / self.temperature

def load_data(csv_path, feature_cols):
    df = pd.read_csv(csv_path)
    df = df.dropna()
    df = df[feature_cols]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.values[-24:])  # Last 24 entries
    return torch.tensor(scaled_data, dtype=torch.float32)

def predict_24_hourly():
    print("📊 Predicting the next 24 hourly candles...")

    feature_cols = [
        "open", "high", "low", "close", "candle_body", "candle_range",
        "upper_wick", "lower_wick", "close_to_open_ratio", "high_to_low_ratio"
    ]

    df = pd.read_csv("TVexport_with_features.csv")
    print(f"🧮 CSV Columns: {list(df.columns)}")

    X = load_data("TVexport_with_features.csv", feature_cols)

    # Initialize and load model
    base_model = CandleNet(input_size=X.shape[1])
    model = TemperatureScaledModel(base_model, temperature=2.0)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()

    with torch.no_grad():
        outputs = model(X)
        predictions = torch.sigmoid(outputs).squeeze()

    with open("predictions_hourly.txt", "w") as f:
        for i, p in enumerate(predictions, 1):
            label = "Green" if p.item() > 0.5 else "Red"
            confidence = p.item()
            line = f"Hour {i}: {label} (Confidence: {confidence:.2f})"
            print(line)
            f.write(line + "\n")

if __name__ == "__main__":
    predict_24_hourly()
