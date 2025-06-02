import torch
import pandas as pd
import numpy as np
from torch import nn
from sklearn.preprocessing import StandardScaler

class CandleNet(nn.Module):
    def __init__(self, input_size=8):
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

def predict_24_hourly():
    print("📊 Predicting the next 24 hourly candles...")

    csv_path = "TVexport_with_features.csv"
    model_path = "model.pth"

    df = pd.read_csv(csv_path)
    print(f"🧮 CSV Columns: {list(df.columns)}")

    df = df.ffill().bfill()

    # Keep only the latest 24 candles
    last_24 = df.iloc[-24:].copy()

    features = [
        "candle_body", "candle_range", "upper_wick", "lower_wick",
        "close_to_open_ratio", "high_to_low_ratio", "open", "close"
    ]

    # Scale features
    scaler = StandardScaler()
    X = last_24[features].values.astype(np.float32)
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Load model
    model = CandleNet(input_size=len(features))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    with torch.no_grad():
        outputs = model(X_tensor).squeeze()
        probabilities = torch.sigmoid(outputs).numpy()
        classes = ["Green" if p >= 0.5 else "Red" for p in probabilities]

    for i, pred in enumerate(classes, 1):
        print(f"Hour {i}: {pred} ({probabilities[i-1]:.4f})")

    with open("predictions_hourly.txt", "w") as f:
        for i, pred in enumerate(classes, 1):
            f.write(f"Hour {i}: {pred} ({probabilities[i-1]:.4f})\n")

    print("✅ Predictions saved to predictions_hourly.txt")

if __name__ == "__main__":
    predict_24_hourly()
