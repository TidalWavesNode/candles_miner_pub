import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import requests
import random
import sys
from datetime import datetime, timedelta

# 🧠 Model definition
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

# 🔧 Load scaler and check feature count
with open("scaler.pkl", "rb") as f:
    scaler_obj = pickle.load(f)
    scaler = scaler_obj["scaler"]
    expected_features = scaler_obj["n_features"]

if expected_features != 10:
    print(f"❌ Expected 10 features but got {expected_features}.")
    sys.exit(1)

# 🎯 Load trained model
model = CandleNet()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# 💰 Get current TAO price
try:
    response = requests.get(
        'https://data-api.coindesk.com/index/cc/v1/latest/tick',
        params={"market": "cadli", "instruments": "TAO-USDT", "apply_mapping": "true"},
        headers={"Content-type": "application/json; charset=UTF-8"}
    )
    data = response.json()
    current_price = data["Data"]["TAO-USDT"]["VALUE"]
    print(f"📊 Starting TAO Price: ${current_price:.4f}")
except Exception as e:
    print(f"⚠️ Failed to fetch live price: {e}")
    current_price = 100.0

# 🔮 Define prediction logic
def predict(price, horizon, label, ranges, interval_sec, start_time):
    rows = []
    for i in range(horizon):
        timestamp = int((start_time + timedelta(seconds=i * interval_sec)).timestamp())
        open_p = price
        high = open_p * random.uniform(*ranges["high"])
        low = open_p * random.uniform(*ranges["low"])
        close = open_p * random.uniform(*ranges["close"])

        candle_body = abs(close - open_p)
        candle_range = high - low
        upper_wick = high - max(close, open_p)
        lower_wick = min(close, open_p) - low
        close_to_open = close / open_p
        high_to_low = high / low

        full_vector = np.array([
            open_p, high, low, close,
            candle_body, candle_range, upper_wick, lower_wick, close_to_open, high_to_low
        ]).reshape(1, -1)

        scaled = scaler.transform(full_vector)
        features_tensor = torch.tensor(scaled, dtype=torch.float32)

        with torch.no_grad():
            logit = model(features_tensor)
            prob = torch.sigmoid(logit).item()
            noise = random.uniform(0.0, 1.0)
            confidence = (prob + noise) / 2

        direction = "Green" if prob > 0.5 else "Red"
        delta = open_p * (ranges["delta_base"] + random.uniform(*ranges["delta_range"]))
        price = price + delta if direction == "Green" else price - delta

        print(f"{label} {i + 1}: {direction} (Confidence: {confidence:.2f}) → Predicted Price: ${price:.4f}")
        rows.append((timestamp, direction, round(confidence, 2), round(price, 4)))
    return rows, price

# 📈 Prediction ranges
ranges_hourly = {
    "high": (1.001, 1.01),
    "low": (0.99, 0.999),
    "close": (0.995, 1.005),
    "delta_base": 0.005,
    "delta_range": (0.001, 0.008)
}

ranges_daily = {
    "high": (1.01, 1.05),
    "low": (0.95, 0.99),
    "close": (0.97, 1.03),
    "delta_base": 0.01,
    "delta_range": (0.005, 0.015)
}

ranges_weekly = {
    "high": (1.05, 1.12),
    "low": (0.90, 0.97),
    "close": (0.95, 1.08),
    "delta_base": 0.015,
    "delta_range": (0.01, 0.02)
}

# 🚀 Perform predictions
price = current_price
base_time = datetime.utcnow()

print("\n🔮 Predicting next 24 hourly candles...")
hourly_rows, price = predict(price, 24, "Hour", ranges_hourly, 3600, base_time)

print("\n🔮 Predicting next 7 daily candles...")
daily_rows, price = predict(price, 7, "Day", ranges_daily, 86400, base_time + timedelta(hours=24))

print("\n🔮 Predicting next 4 weekly candles...")
weekly_rows, price = predict(price, 4, "Week", ranges_weekly, 604800, base_time + timedelta(days=7))

# 💾 Save all predictions
combined = hourly_rows + daily_rows + weekly_rows
pd.DataFrame(combined, columns=["timestamp", "color", "confidence", "price"]).to_csv("all_predictions.csv", index=False)
print("✅ Unified predictions saved to all_predictions.csv")
