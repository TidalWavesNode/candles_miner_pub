import torch
import pandas as pd
import numpy as np
from torch import nn
from sklearn.preprocessing import MinMaxScaler

# 🔧 Define model architecture matching training script
class CandleNet(nn.Module):
    def __init__(self, input_size=8):
        super(CandleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# 🔍 Predict the next 24 hourly candles
def predict_24_hourly():
    print("📊 Predicting the next 24 hourly candles...")
    csv_path = "TVexport_with_features.csv"
    model_path = "model.pth"

    df = pd.read_csv(csv_path)
    print(f"🧮 CSV Columns: {list(df.columns)}")

    # Fill missing values
    df = df.ffill().bfill()

    # Use the last 24 rows for prediction
    last_24 = df.iloc[-24:].copy()

    # Only use the features used during training
    features = [
        "candle_body", "candle_range", "upper_wick", "lower_wick",
        "close_to_open_ratio", "high_to_low_ratio", "open", "close"
    ]
    X = last_24[features].values.astype(np.float32)

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Load model
    model = CandleNet(input_size=len(features))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # Predict with confidence
    with torch.no_grad():
        outputs = model(X_tensor).squeeze()
        probs = torch.sigmoid(outputs).numpy()

    # Print results
    with open("predictions_hourly.txt", "w") as f:
        for i, p in enumerate(probs, 1):
            label = "Green" if p >= 0.5 else "Red"
            confidence = round(p if p >= 0.5 else 1 - p, 2)
            print(f"Hour {i}: {label} (Confidence: {confidence:.2f})")
            f.write(f"Hour {i}: {label} (Confidence: {confidence:.2f})\n")

    print("✅ Predictions saved to predictions_hourly.txt")

if __name__ == "__main__":
    predict_24_hourly()
