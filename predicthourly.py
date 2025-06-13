import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

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

def enable_dropout(model):
    """Enable dropout layers during test-time."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def predict_24_hourly(mc_passes=30):
    print("📊 Predicting the next 24 hourly candles with MC Dropout...")
    df = pd.read_csv("TVexport_with_features.csv")
    print("🧮 CSV Columns:", list(df.columns))

    features = ['candle_body', 'candle_range', 'upper_wick', 'lower_wick',
                'close_to_open_ratio', 'high_to_low_ratio', 'open', 'close']

    data = df[features].copy().tail(24)
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    X = torch.tensor(X, dtype=torch.float32)

    model = CandleNet(input_size=8)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()
    enable_dropout(model)  # ✅ Enable dropout during eval for MC Dropout

    all_probs = []

    with torch.no_grad():
        for _ in range(mc_passes):
            logits = model(X).squeeze()
            probs = torch.sigmoid(logits)
            all_probs.append(probs.unsqueeze(0))  # Shape: [1, 24]

    probs_stack = torch.cat(all_probs, dim=0)  # Shape: [mc_passes, 24]
    mean_probs = probs_stack.mean(dim=0)
    std_probs = probs_stack.std(dim=0)  # Uncertainty

    with open("predictions_hourly.txt", "w") as f:
        for i in range(24):
            label = "Green" if mean_probs[i] >= 0.5 else "Red"
            confidence = mean_probs[i].item()
            uncertainty = std_probs[i].item()
            output = (f"Hour {i+1}: {label} "
                      f"(Confidence: {confidence:.2f}, "
                      f"Uncertainty: ±{uncertainty:.2f})")
            print(output)
            f.write(output + "\n")

if __name__ == "__main__":
    predict_24_hourly()
