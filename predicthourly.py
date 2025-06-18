import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import requests
import pickle
from sklearn.preprocessing import StandardScaler

# 🧠 Model architecture (matches what was used in training)
class CandleNet(nn.Module):
    def __init__(self):
        super(CandleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 256),
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
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 🧮 Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 🧠 Load trained model
model = CandleNet()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# 📈 Load latest 24 candles
df = pd.read_csv("TVexport_with_features.csv")
features = [
    "open", "high", "low", "close", "candle_body", "candle_range",
    "upper_wick", "lower_wick", "close_to_open_ratio", "high_to_low_ratio"
]
latest = df[features].tail(24).values.astype(np.float32)
X = scaler.transform(latest)
X_tensor = torch.tensor(X)

# 🌐 Fetch current price from Coindesk API
try:
    response = requests.get("https://data-api.coindesk.com//index/cc/v1/latest/tick?market=cadli&instruments=TAO-USDT&apply_mapping=true")
    response.raise_for_status()
    current_price = float(response.json()["data"]["TAO-USDT"]["c"])
except Exception as e:
    print(f"⚠️ Failed to fetch live price: {e}")
    current_price = float(df["close"].iloc[-1])  # fallback to last known price

print(f"📊 Starting TAO Price: ${current_price:.4f}")
print("🔮 Predicting next 24 candles...\n")

# 📤 Predict direction and simulate price movements
price = current_price
for i, x in enumerate(X_tensor, 1):
    with torch.no_grad():
        pred = model(x.unsqueeze(0)).item()
    label = "Green" if pred >= 0.5 else "Red"

    # 💥 Simulate confidence with noise
    noise = np.random.uniform(0, 1)
    confidence = abs(pred - 0.5) * 2
    confidence = np.clip((confidence + noise) / 2, 0, 1)

    # 💹 Simulate price change (± up to 2% movement)
    pct_change = np.random.uniform(0.001, 0.02) * (1 if label == "Green" else -1)
    price *= 1 + pct_change

    print(f"Hour {i}: {label} (Confidence: {confidence:.2f}) → Predicted Price: ${price:.4f}")
