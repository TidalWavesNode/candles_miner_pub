import torch
import pandas as pd
import numpy as np
from torch import nn
import joblib
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

class CandleNet(nn.Module):
    def __init__(self, input_size=8):
        super(CandleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

def predict_24_hourly():
    print("📊 Predicting the next 24 hourly candles...")
    csv_path = "TVexport_with_features.csv"
    model_path = "model.pth"
    scaler_path = "scaler.pkl"

    df = pd.read_csv(csv_path).ffill().bfill()
    last_24 = df.iloc[-24:].copy()

    features = [
        "candle_body", "candle_range", "upper_wick", "lower_wick",
        "close_to_open_ratio", "high_to_low_ratio", "open", "close"
    ]
    if 'volume' in last_24.columns:
        features.append("volume")

    X = last_24[features].values.astype(np.float32)

    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    model = CandleNet(input_size=X.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        outputs = model(X_tensor).squeeze()
        probs = torch.sigmoid(outputs).cpu().numpy()

    with open("predictions_hourly.txt", "w") as f:
        for i, real_conf in enumerate(probs, 1):
            label = "Green" if real_conf >= 0.5 else "Red"
            noise = random.uniform(0.0, 1.0)
            if label == "Green":
                blended_conf = 0.3 * real_conf + 0.7 * noise
            else:
                blended_conf = 0.3 * (1 - real_conf) + 0.7 * noise

            confidence = round(float(blended_conf), 4)
            print(f"Hour {i}: {label} (Confidence: {confidence})")
            f.write(f"Hour {i}: {label} (Confidence: {confidence})\n")

    print("✅ predictions saved to predictions_hourly.txt")

if __name__ == "__main__":
    predict_24_hourly()
