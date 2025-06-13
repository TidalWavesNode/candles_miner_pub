import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ---- Model Definition ----
class CandleNet(nn.Module):
    def __init__(self, input_size=10):
        super(CandleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
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

# ---- Temperature Scaling Wrapper ----
class TemperatureScaledModel(nn.Module):
    def __init__(self, base_model, temperature=2.0):
        super(TemperatureScaledModel, self).__init__()
        self.base_model = base_model
        self.temperature = temperature

    def forward(self, x):
        logits = self.base_model(x)
        return logits / self.temperature

# ---- Prediction Function ----
def predict_24_hourly():
    print("📊 Predicting the next 24 hourly candles...")

    # Load and preprocess CSV
    df = pd.read_csv("TVexport_with_features.csv")
    features = ['candle_body', 'candle_range', 'upper_wick', 'lower_wick',
            'close_to_open_ratio', 'high_to_low_ratio', 'open', 'close']
    print(f"🧮 CSV Columns: {list(df.columns)}")

    X = df[features].values[-24:]  # Last 24 rows
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Load model
    base_model = CandleNet(input_size=10)
    base_model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model = TemperatureScaledModel(base_model, temperature=2.0)
    model.eval()

    # Predict
    with torch.no_grad():
        outputs = model(X_tensor).squeeze()
        probs = torch.sigmoid(outputs)

    # Display predictions
    for i, prob in enumerate(probs):
        prediction = "Green" if prob >= 0.5 else "Red"
        confidence = prob.item() if prediction == "Green" else 1 - prob.item()
        print(f"Hour {i+1}: {prediction} (Confidence: {confidence:.2f})")

    # Optionally save
    with open("predictions_hourly.txt", "w") as f:
        for i, prob in enumerate(probs):
            prediction = "Green" if prob >= 0.5 else "Red"
            f.write(f"Hour {i+1}: {prediction}\n")

# ---- Run ----
if __name__ == "__main__":
    predict_24_hourly()
