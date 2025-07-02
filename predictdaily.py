import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import requests
import random
import sys

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

with open("scaler.pkl", "rb") as f:
    scaler_obj = pickle.load(f)
    scaler = scaler_obj["scaler"]
    expected_features = scaler_obj["n_features"]

if expected_features != 10:
    print(f"❌ Expected 10 features but got {expected_features}.")
    sys.exit(1)

model = CandleNet()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

try:
    response = requests.get(
        'https://data-api.coindesk.com/index/cc/v1/latest/tick',
        params={"market": "cadli", "instruments": "TAO-USDT", "apply_mapping": "true"},
        headers={"Content-type": "application/json; charset=UTF-8"}
    )
    data = response.json()
    current_price = data["Data"]["TAO-USDT"]["VALUE"]
except Exception as e:
    print(f"⚠️ Failed to fetch live price: {e}")
    current_price = 100.0

price = current_price
output = [f"📊 Starting TAO Price: ${price:.4f}\n", "🔮 Predicting next 7 daily candles...\n"]

for day in range(1, 8):
    open_p = price
    high = open_p * random.uniform(1.01, 1.05)
    low = open_p * random.uniform(0.95, 0.99)
    close = open_p * random.uniform(0.97, 1.03)

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
    delta = open_p * (0.01 + random.uniform(0.005, 0.015))
    price = price + delta if direction == "Green" else price - delta

    line = f"Day {day}: {direction} (Confidence: {confidence:.2f}) → Predicted Price: ${price:.4f}\n"
    print(line, end="")
    output.append(line)

with open("predictions_daily.txt", "w") as f:
    f.writelines(output)
print("✅ Daily predictions saved to predictions_daily.txt")
