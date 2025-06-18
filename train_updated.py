import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import argparse
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, required=True)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.0005)
args = parser.parse_args()

df = pd.read_csv(args.csv)
features = [
    "open", "high", "low", "close", "candle_body", "candle_range",
    "upper_wick", "lower_wick", "close_to_open_ratio", "high_to_low_ratio"
]
X = df[features].values.astype(np.float32)
y = (df["close"] > df["open"]).astype(int).values.astype(np.float32)

# 🧪 Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save both scaler and feature count
with open("scaler.pkl", "wb") as f:
    pickle.dump({"scaler": scaler, "n_features": X.shape[1]}, f)

# 🔧 Datasets
train_data = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

model = CandleNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# 🏋️ Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += ((pred > 0.5) == yb).sum().item()
        total += yb.size(0)
    acc = correct / total
    print(f"📈 Epoch {epoch}/{args.epochs} - Loss: {total_loss:.4f} - Accuracy: {acc:.4f}")

# 💾 Save model
torch.save(model.state_dict(), "model.pth")
print("✅ Model saved to model.pth")
