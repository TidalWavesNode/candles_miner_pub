# predicthourly.py

import torch
import pandas as pd
import numpy as np
from torch import nn
from sklearn.preprocessing import MinMaxScaler

class CandleNet(nn.Module):
    def __init__(self, input_size=8):
        super(CandleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

def predict_24_hourly():
    print("📊 Predicting the next 24 hourly candles...")

    csv_path = "TVexport_with_features.csv"
    model_path = "model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(csv_path)
    print(f"🧮 CSV Columns: {list(df.columns)}")

    # Fill missing values without deprecation warning
    df = df.ffill().bfill()

    # Keep the last 24 rows for prediction
    last_24 = df.iloc[-24:].copy()

    # Match feature set used during training
    features = [
        "candle_body", "candle_range", "upper_wick", "lower_wick",
        "close_to_open_ratio", "high_to_low_ratio", "open", "close"
    ]

    X = last_24[features].values.astype(np.float32)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    # Load model
    model = CandleNet(input_size=len(features)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Predict
    with torch.no_grad():
        outputs = model(X_tensor).squeeze()
        predictions = torch.sigmoid(outputs).cpu().numpy()
        classes = ["Green" if p >= 0.5 else "Red" for p in predictions]

    # Output results
    for i, pred in enumerate(classes, 1):
        print(f"Hour {i}: {pred}")

    # Save results
    with open("predictions_hourly.txt", "w") as f:
        for i, pred in enumerate(classes, 1):
            f.write(f"Hour {i}: {pred}\n")

    print("✅ Predictions saved to predictions_hourly.txt")

if __name__ == "__main__":
    predict_24_hourly()
