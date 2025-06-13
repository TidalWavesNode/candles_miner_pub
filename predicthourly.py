import torch
import pandas as pd
import numpy as np
from torch import nn
from sklearn.preprocessing import StandardScaler

# 🧠 Updated Deep CandleNet Model
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

def predict_24_hourly():
    print("📊 Predicting the next 24 hourly candles...")

    csv_path = "TVexport_with_features.csv"
    model_path = "model.pth"

    df = pd.read_csv(csv_path)
    print(f"🧮 CSV Columns: {list(df.columns)}")
    df = df.ffill().bfill()

    # Features used during training
    features = [
    "candle_body", "candle_range", "upper_wick", "lower_wick",
    "close_to_open_ratio", "high_to_low_ratio", "open", "close"
    ]
    X = df[features].tail(24).values.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model = CandleNet(input_size=len(features))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # 🔁 Enable dropout for MC sampling
    def enable_dropout(m):
        if isinstance(m, nn.Dropout):
            m.train()
    model.apply(enable_dropout)

    results = []
    for i, x in enumerate(X_tensor):
        x = x.unsqueeze(0)
        probs = []
        for _ in range(30):  # 30 stochastic forward passes
            with torch.no_grad():
                prob = torch.sigmoid(model(x)).item()
                probs.append(prob)
        mean = np.mean(probs)
        std = np.std(probs)
        confidence = 1.0 - std
        label = "Green" if mean >= 0.5 else "Red"
        results.append((i + 1, label, confidence))

    with open("predictions_hourly.txt", "w") as f:
        for hour, label, confidence in results:
            line = f"Hour {hour}: {label} (Confidence: {confidence:.2f})"
            print(line)
            f.write(line + "\n")

if __name__ == "__main__":
    predict_24_hourly()
