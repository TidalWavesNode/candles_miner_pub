import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import argparse

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

class CandleDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        print("📄 CSV Columns:", df.columns.tolist())

        # Fill missing values
        df = df.ffill().bfill()

        # Feature engineering
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
            nn.Dropout(0.5),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

def train_model(csv_path, epochs, batch_size, lr):
    dataset = CandleDataset(csv_path)

    # Handle class imbalance
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(dataset.y), y=dataset.y.astype(int))
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])

    # Use weighted sampling to balance each batch
    sample_weights = np.array([class_weights[int(label)] for label in dataset.y])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    model = CandleNet(input_size=dataset.X.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct = 0.0, 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

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
        print(f"📈 Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("✅ Model saved to model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="TVexport_with_features.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train_model(args.csv, args.epochs, args.batch_size, args.lr)
