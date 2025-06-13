import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

# ✅ Define the model architecture (same as in training)
class CandleNet(nn.Module):
    def __init__(self, input_size=8):
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

def predict_24_hourly():
    print("📊 Predicting the next 24 hourly candles...")

    # ✅ Load data
    df = pd.read_csv("TVexport_with_features.csv")
    print("🧮 CSV Columns:", list(df.columns))

    # ✅ Use only the 8 features used during training
    features = ['candle_body', 'candle_range', 'upper_wick', 'lower_wick',
                'close_to_open_ratio', 'high_to_low_ratio', 'open', 'close']
    data = df[features].copy().tail(24)

    # ✅ Normalize inputs
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # ✅ Convert to tensor
    X = torch.tensor(data_scaled, dtype=torch.float32)

    # ✅ Load model
    model = CandleNet(input_size=8)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()

    # ✅ Make predictions
    with torch.no_grad():
        logits = model(X).squeeze()
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).int()

    # ✅ Show output with confidence
    with open("predictions_hourly.txt", "w") as f:
        for i in range(24):
            label = "Green" if preds[i] == 1 else "Red"
            confidence = float(probs[i])
            output = f"Hour {i+1}: {label} (Confidence: {confidence:.2f})"
            print(output)
            f.write(output + "\n")

if __name__ == "__main__":
    predict_24_hourly()
