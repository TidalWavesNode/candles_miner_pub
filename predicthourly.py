import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import random
import requests
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

# 🔃 Load scaler + model
with open("scaler.pkl", "rb") as f:
    scaler_obj = pickle.load(f)
    scaler = scaler_obj["scaler"]
    expected_features = scaler_obj["n_features"]

if expected_features != 10:
    print(f"❌ Expected 10 features but got {expected_features}.")
    exit(1)

model = CandleNet()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# 🌐 Get current TAO/USDT price
try:
    response = requests.get(
        'https://data-api.coindesk.com/index/cc/v1/latest/tick',
        params={"market": "cadli", "instruments": "TAO-USDT", "apply_mapping": "true"},
        headers={"Content-type": "application/json; charset=UTF-8"}
    )
    data = response.json()
    live_price = data["Data"]["TAO-USDT"]["VALUE"]
    print(f"📊 Starting TAO Price (CADLI): ${live_price:.4f}")
except Exception as e:
    print(f"⚠️ Failed to fetch live price: {e}")
    live_price = 100.0

# 🧮 Load last 24 real candles and normalize to live price
df = pd.read_csv("TVexport_with_features.csv").tail(24).copy()
base_close = df.iloc[0]["close"]
adjustment_ratio = live_price / base_close

# Adjust OHLC and derived features proportionally
df[["open", "high", "low", "close"]] *= adjustment_ratio
df["candle_body"] = df["close"] - df["open"]
df["candle_range"] = df["high"] - df["low"]
df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
df["close_to_open_ratio"] = df["close"] / df["open"]
df["high_to_low_ratio"] = df["high"] / df["low"]

features = df[[
    "open", "high", "low", "close",
    "candle_body", "candle_range",
    "upper_wick", "lower_wick",
    "close_to_open_ratio", "high_to_low_ratio"
]]
scaled = scaler.transform(features.values)

# 🔮 Predict
csv_rows = [("timestamp", "color", "confidence", "price")]
base_time = datetime.utcnow()

print("🔮 Predicting next 24 hourly candles...\n")

for i in range(24):
    features_tensor = torch.tensor(scaled[i].reshape(1, -1), dtype=torch.float32)
    with torch.no_grad():
        logit = model(features_tensor)
        prob = torch.sigmoid(logit).item()
        noise = random.uniform(0.3, 0.7)
        confidence = 0.3 * prob + 0.7 * noise

    direction = "Green" if prob > 0.5 else "Red"
    adjusted_price = df.iloc[i]["close"]
    timestamp = int((base_time + timedelta(hours=i)).timestamp())
    print(f"Hour {i+1}: {direction} (Confidence: {confidence:.2f}) → Predicted Price: ${adjusted_price:.4f}")
    csv_rows.append((timestamp, direction, round(confidence, 2), round(adjusted_price, 4)))

# 💾 Save predictions
pd.DataFrame(csv_rows[1:], columns=csv_rows[0]).to_csv("hourly_predictions.csv", index=False)
print("✅ Hourly predictions saved to hourly_predictions.csv")
