import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

class CandleDataset(Dataset):
    def __init__(self, csv_path, scaler_path="scaler.pkl", save_scaler=True):
        df = pd.read_csv(csv_path).ffill().bfill()

        features = [
            "candle_body", "candle_range", "upper_wick", "lower_wick",
            "close_to_open_ratio", "high_to_low_ratio", "open", "close"
        ]
        if 'volume' in df.columns:
            features.append('volume')

        self.y = (df["close"] > df["open"]).astype(int).values.astype(np.float32)
        print(f"⚖️ Positive class ratio: {np.mean(self.y):.2f}")

        scaler = StandardScaler()
        self.X = scaler.fit_transform(df[features].values.astype(np.float32))

        if save_scaler:
            joblib.dump(scaler, scaler_path)

        self.X = self.X.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CandleNet(nn.Module):
    def __init__(self, input_size):
        super(CandleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

def train_model(csv_path, epochs, batch_size, lr):
    dataset = CandleDataset(csv_path)
    
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(dataset.y),
        y=dataset.y.astype(int)
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
    sample_weights = np.array([class_weights[int(label)] for label in dataset.y])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    model = CandleNet(input_size=dataset.X.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)  # ✅ Add weight_decay
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

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

        scheduler.step()

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
