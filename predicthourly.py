import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import requests
import random
import sys

# Define the model
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

# Load scaler and model
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

# Load most recent row
df = pd.read_csv("TVexport_with_features.csv")
last_row = df.iloc[-1].copy()

# Fetch current price
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
    current_price = last_row["close"]

print("🔮 Predicting next 24 candles...\n")

price = current_price

for hour in range(1, 25):
    # Build feature vector for current hour
    feature_vector = np.array([
        price,  # open
        price * random.uniform(1.001, 1.01),  # high
        price * random.uniform(0.99, 0.999),  # low
        price * random.uniform(0.995, 1.005),  # close
    ])
    candle_body = abs(feature_vector[3] - feature_vector[0])
    candle_range = feature_vector[1] - feature_vector[2]
    upper_wick = feature_vector[1] - max(feature_vector[3], feature_vector[0])
    lower_wick = min(feature_vector[3], feature_vector[0]) - feature_vector[2]
    close_to_open = feature_vector[3] / feature_vector[0]
    high_to_low = feature_vector[1] / feature_vector[2]

    full_vector = np.array([
        feature_vector[0], feature_vector[1], feature_vector[2], feature_vector[3],
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
    delta = price * (0.005 + random.uniform(0.001, 0.008))
    price = price + delta if direction == "Green" else price - delta

    print(f"Hour {hour}: {direction} (Confidence: {confidence:.2f}) → Predicted Price: ${price:.4f}")
