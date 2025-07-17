import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import random
import requests
from datetime import datetime, timedelta
import pandas as pd

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

# 🧮 Simulate 24 candle feature rows
features_list = []
base_price = live_price

for _ in range(24):
    open_price = base_price + random.uniform(-5, 5)
    close_price = open_price + random.uniform(-3, 3)
    high_price = max(open_price, close_price) + random.uniform(0, 2)
    low_price = min(open_price, close_price) - random.uniform(0, 2)
    candle_body = close_price - open_price
    candle_range = high_price - low_price
    upper_wick = high_price - max(open_price, close_price)
    lower_wick = min(open_price, close_price) - low_price
    close_to_open = close_price / open_price if open_price != 0 else 1
    high_to_low = high_price / low_price if low_price != 0 else 1

    features_list.append([
        open_price, high_price, low_price, close_price,
        candle_body, candle_range,
        upper_wick, lower_wick,
        close_to_open, high_to_low
    ])

features_array = np.array(features_list)
scaled = scaler.transform(features_array)

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
    predicted_price = features_list[i][3]  # close price
    timestamp = int((base_time + timedelta(hours=i)).timestamp())
    print(f"Hour {i+1}: {direction} (Confidence: {confidence:.2f}) → Predicted Price: ${predicted_price:.4f}")
    csv_rows.append((timestamp, direction, round(confidence, 2), round(predicted_price, 4)))

# 💾 Save predictions
os.makedirs(os.path.expanduser("~/.candles/data/"), exist_ok=True)
pd.DataFrame(csv_rows[1:], columns=csv_rows[0]).to_csv(
    os.path.expanduser("~/.candles/data/hourly_predictions.csv"), index=False
)
print("✅ Hourly predictions saved to ~/.candles/data/hourly_predictions.csv")
