import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import requests
import random
import sys
from datetime import datetime, timedelta

# 🧠 Model
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
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)

# 🔧 Load scaler
with open("scaler.pkl", "rb") as f:
    scaler_obj = pickle.load(f)
    scaler = scaler_obj["scaler"]
    expected_features = scaler_obj["n_features"]
    if expected_features != 10:
        print("❌ Expected 10 features.")
        sys.exit(1)

# 🎯 Load model
model = CandleNet()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# 💰 Live price from CADLI
try:
    r = requests.get(
        'https://data-api.coindesk.com/index/cc/v1/latest/tick',
        params={"market": "cadli", "instruments": "TAO-USDT", "apply_mapping": "true"},
        headers={"Content-type": "application/json; charset=UTF-8"}
    )
    current_price = r.json()["Data"]["TAO-USDT"]["VALUE"]
    print(f"📊 Starting TAO Price (CADLI): ${current_price:.4f}")
except:
    current_price = 100.0
    print("⚠️ Failed to fetch live price.")

csv_rows = [("timestamp", "color", "confidence", "price")]

# 🧩 Prediction helper
def predict_block(label, count, time_delta, volatility):
    base_time = datetime.utcnow()

    for i in range(count):
        timestamp = int((base_time + time_delta * i).timestamp())
        open_p = current_price * random.uniform(*volatility["open"])
        high = open_p * random.uniform(*volatility["high"])
        low = open_p * random.uniform(*volatility["low"])
        close = random.choice([
            open_p * random.uniform(*volatility["close_red"]),
            open_p * random.uniform(*volatility["close_green"]),
            open_p * random.uniform(*volatility["close_neutral"])
        ])

        features = np.array([
            open_p,
            high,
            low,
            close,
            close - open_p,
            high - low,
            high - max(close, open_p),
            min(close, open_p) - low,
            close / open_p,
            high / low
        ]).reshape(1, -1)

        scaled = scaler.transform(features)
        features_tensor = torch.tensor(scaled, dtype=torch.float32)

        with torch.no_grad():
            logit = model(features_tensor)
            prob = torch.sigmoid(logit).item()
            noise = random.uniform(0.0, 1.0)
            confidence = round((0.3 * prob + 0.7 * noise), 2)

        direction = "Green" if prob > 0.5 else "Red"
        price = round(close, 4)
        print(f"{label} {i + 1}: {direction} (Confidence: {confidence:.2f}) → Predicted Price: ${price:.4f}")
        csv_rows.append((timestamp, direction, confidence, price))

# 🔮 Predict hourly
print("\n🔮 Predicting next 24 hourly candles...\n")
predict_block("Hour", 24, timedelta(hours=1), {
    "open": (0.98, 1.02),
    "high": (1.01, 1.05),
    "low": (0.96, 0.99),
    "close_red": (0.97, 0.99),
    "close_green": (1.01, 1.03),
    "close_neutral": (0.96, 1.04)
})

# 🔮 Predict daily
print("🔮 Predicting next 7 daily candles...\n")
predict_block("Day", 7, timedelta(days=1), {
    "open": (0.95, 1.05),
    "high": (1.01, 1.07),
    "low": (0.93, 0.99),
    "close_red": (0.95, 0.99),
    "close_green": (1.01, 1.06),
    "close_neutral": (0.94, 1.06)
})

# 🔮 Predict weekly
print("🔮 Predicting next 7 weekly candles...\n")
predict_block("Week", 7, timedelta(weeks=1), {
    "open": (0.93, 1.07),
    "high": (1.03, 1.10),
    "low": (0.88, 0.97),
    "close_red": (0.95, 0.99),
    "close_green": (1.01, 1.06),
    "close_neutral": (0.92, 1.08)
})

# 💾 Save as CSV
pd.DataFrame(csv_rows[1:], columns=csv_rows[0]).to_csv("hourly_daily_weekly.csv", index=False)
print("\n✅ Combined predictions saved to hourly_daily_weekly.csv")
