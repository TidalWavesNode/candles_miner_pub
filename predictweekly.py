import torch
import torch.nn as nn
import pandas as pd
import numpy as np
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

# 🔧 Load scaler
with open("scaler.pkl", "rb") as f:
    scaler_obj = pickle.load(f)
    scaler = scaler_obj["scaler"]
    expected_features = scaler_obj["n_features"]

if expected_features != 10:
    print(f"❌ Expected 10 features but got {expected_features}.")
    sys.exit(1)

# 🎯 Load model
model = CandleNet()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# 💰 Get live TAO price
try:
    response = requests.get(
        'https://data-api.coindesk.com/index/cc/v1/latest/tick',
        params={"market": "cadli", "instruments": "TAO-USDT", "apply_mapping": "true"},
        headers={"Content-type": "application/json; charset=UTF-8"}
    )
    data = response.json()
    current_price = data["Data"]["TAO-USDT"]["VALUE"]
    print(f"📊 Starting TAO Price (CADLI): ${current_price:.4f}")
except Exception as e:
    print(f"⚠️ Failed to fetch live price: {e}")
    current_price = 100.0

# 🔮 Predict next 7 weekly candles
print("🔮 Predicting next 7 weekly candles...\n")
csv_rows = [("timestamp", "color", "confidence", "price")]
base_time = datetime.utcnow()

for week in range(7):
    timestamp = int((base_time + timedelta(weeks=week)).timestamp())

    open_p = current_price * random.uniform(0.93, 1.07)
    high = open_p * random.uniform(1.03, 1.10)
    low = open_p * random.uniform(0.88, 0.97)
    close = random.choice([
        open_p * random.uniform(0.95, 0.99),  # likely red
        open_p * random.uniform(1.01, 1.06),  # likely green
        open_p * random.uniform(0.92, 1.08),  # chaotic/neutral
    ])

    candle_body = close - open_p
    candle_range = high - low
    upper_wick = high - max(close, open_p)
    lower_wick = min(close, open_p) - low
    close_to_open = close / open_p
    high_to_low = high / low

    features = np.array([
        open_p, high, low, close,
        candle_body, candle_range, upper_wick, lower_wick,
        close_to_open, high_to_low
    ]).reshape(1, -1)

    scaled = scaler.transform(features)
    features_tensor = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        logit = model(features_tensor)
        prob = torch.sigmoid(logit).item()
        noise = random.uniform(0.0, 1.0)
        confidence = round((0.3 * prob + 0.7 * noise), 2)

    direction = "Green" if prob > 0.5 else "Red"
    price = close  # Do not chain prices week-to-week

    print(f"Week {week + 1}: {direction} (Confidence: {confidence:.2f}) → Predicted Price: ${price:.4f}")
    csv_rows.append((timestamp, direction, confidence, round(price, 4)))

# 💾 Save output
pd.DataFrame(csv_rows[1:], columns=csv_rows[0]).to_csv("weekly_predictions.csv", index=False)
print("✅ Weekly predictions saved to weekly_predictions.csv")
