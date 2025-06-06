import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse

# Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

class CandleDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        print("📄 CSV Columns:", df.columns.tolist())

        # Fill missing values
        df = df.ffill().bfill()

        # Feature engineering (assumed already present if using TVexport_with_features.csv)
        if 'candle_body' not in df.columns:
            df["candle_body"] = df["close"] - df["open"]
            df["candle_range"] = df["high"] - df["low"]
            df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
            df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
            df["close_to_open_ratio"] = df["close"] / df["open"]
            df["high_to_low_ratio"] = df["high"] / df["low"]

        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        self.y = (df["close"] > df["open"]).astype(int).values
        self.X = df[[
            "candle_body", "candle_range", "upper_wick", "lower_wick",
            "close_to_open_ratio", "high_to_low_ratio", "open", "close"
        ]].values

        self.X = StandardScaler().fit_transform(self.X).astype(np.float32)
        self.y = self.y.astype(np.float32)

        print(f"✅ Final dataset: {len(self.X)} samples, {self.X.shape[1]} features")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CandleNet(nn.Module):
    def __init__(self, input_size):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="TVexport_with_features.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    print("📦 Loading data...")
    dataset = CandleDataset(args.csv)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = CandleNet(input_size=dataset.X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct = 0.0, 0
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(X).squeeze()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)

            preds = torch.round(torch.sigmoid(outputs))
            correct += (preds == y).sum().item()

        accuracy = correct / len(dataset)
        avg_loss = total_loss / len(dataset)
        print(f"📈 Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("✅ Model saved to model.pth")
