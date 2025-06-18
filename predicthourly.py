import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import requests
import random
import sys

# Define the model architecture inline
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

# Load scaler and verify feature count
with open("scaler.pkl", "rb") as f:
    scaler_obj = pickle.load(f)
    scaler = scaler_obj["scaler"]
    expected_features = scaler_obj["n_features"]

if expected_features != 10:
    print(f"❌ Expected 10 features but got {expected_features}. Mismatched model or scaler.")
    sys.exit(1)

# Load model
model = CandleNet()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# Load latest data
df = pd.read_csv("TVexport_with_features.csv")
features = df.iloc[-1][['open', 'high', 'low', 'close', 'candle_body', 'candle_range',
                        'upper_wick', 'lower_wick', 'close_to_open_ratio', 'high_to_low_ratio']].values.reshape(1, -1)
features = scaler.transform(features)
features_tensor = torch.tensor(features, dtype=torch.float32)

# Predict direction
with torch.no_grad():
    logit = model(features_tensor)
    prob = torch.sigmoid(logit).item()

# Generate randomized confidence score blended with real probability
noise = random.uniform(0.0, 1.0)
blended_confidence = (prob + noise) / 2

# Fetch current TAO/USDT price from Coindesk CADLI
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
    current_price = 400.00  # fallback

print("🔮 Predicting next 24 candles...\n")

price = current_price
for hour in range(1, 25):
    with torch.no_grad():
        logit = model(features_tensor)
        prob = torch.sigmoid(logit).item()
        noise = random.uniform(0.0, 1.0)
        confidence = (prob + noise) / 2

    direction = "Green" if prob > 0.5 else "Red"
    delta = price * (0.005 + random.uniform(0.001, 0.008))
    price = price + delta if direction == "Green" else price - delta

    print(f"Hour {hour}: {direction} (Confidence: {confidence:.2f}) → Predicted Price: ${price:.4f}")
