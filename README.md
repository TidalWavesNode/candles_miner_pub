# Candles Base Miner

A **binary classification neural network** designed to predict whether the **next hourly trading candle** will be **green** (price goes up) or **red** (price goes down), using the structure and behavior of past candles.

This model leverages engineered features from historical price action to anticipate short-term market movement with precision, and includes flexible, noise-blended confidence scoring for more natural output ranges.

---

## 📚 Table of Contents

- [🟩 Purpose](#-purpose)
- [📊 Dataset](#-dataset)
- [🎯 Target Label](#-target-label)
- [🧠 Model Architecture](#-model-architecture)
- [⚙️ Training Loop](#-training-loop)
- [🔍 What It Learns](#-what-it-learns)
- [🚀 Getting Started](#-getting-started)
  - [📦 Requirements](#-requirements)
  - [🔧 Setup Instructions](#-setup-instructions)
  - [🧹 Prepare Your Dataset](#-prepare-your-dataset)
  - [🛠️ Feature Engineering](#-feature-engineering)
  - [🏋️‍♂️ Train the Model](#-train-the-model)
  - [🔮 Predict the Next 24-Hourly Candles](#-predict-the-next-24-hourly-candles)
  - [🧠 How Confidence Works](#-how-confidence-works)
- [⚠️ Disclaimer](#-disclaimer)

---

## 🟩 Purpose

This is a binary classification task:

- `1` = Green Candle (Close > Open)
- `0` = Red Candle (Close ≤ Open)

The model learns patterns from historical hourly candles to anticipate bullish or bearish movement in the next one.

---

## 📊 Dataset

**File**: `TVexport.csv`  
**Samples**: 9,257 hourly candles

Each row represents a candle and includes raw and engineered features.

### Engineered Features (via `feature_generator.py`)

| Feature               | Description                            |
|-----------------------|----------------------------------------|
| `open`                | Opening price                          |
| `high`                | Highest price                          |
| `low`                 | Lowest price                           |
| `close`               | Closing price                          |
| `candle_body`         | `close - open`                         |
| `candle_range`        | `high - low`                           |
| `upper_wick`          | `high - max(open, close)`              |
| `lower_wick`          | `min(open, close) - low`               |
| `close_to_open_ratio` | `close / open`                         |
| `high_to_low_ratio`   | `high / low`                           |
| `volume` *(optional)* | Volume per candle (if present)         |
| `time` *(optional)*   | Timestamp (retained if present)        |

---

## 🎯 Target Label

The label is calculated as:

```python
label = 1 if close > open else 0
This gives the model a clear binary classification objective.

🧠 Model Architecture
A compact, regularized feedforward neural network:

css
Copy
Edit
Input layer: 8–10 features (including optional volume)
Hidden layers:
  Linear(input → 128) → ReLU → Dropout(0.3)
  Linear(128 → 64)    → ReLU → Dropout(0.2)
  Linear(64 → 32)     → ReLU → Dropout(0.1)
Output layer:
  Linear(32 → 1)
Final Activation: Sigmoid (for binary classification)

Loss Function: BCEWithLogitsLoss (weighted)
Optimizer: AdamW (with weight decay)
⚙️ Training Loop
Inputs scaled with StandardScaler

Class imbalance handled with:

WeightedRandomSampler

pos_weight in BCE loss

AdamW optimizer with weight_decay=0.01

StepLR scheduler for LR decay

Training command:

css
Copy
Edit
python3 train_updated.py --csv TVexport_with_features.csv --epochs 150 --batch-size 64 --lr 0.0005
Model and scaler are saved as:

model.pth

scaler.pkl

🔍 What It Learns
The model identifies candle-based momentum patterns, such as:

Long wicks (reversals or exhaustion)

Wide candle ranges (volatility)

Compression and breakout signatures

Ratio patterns between open/close and high/low

🚀 Getting Started
📦 Requirements
Ubuntu 22.04+

Python 3.10+

pip, venv, torch, pandas, scikit-learn, numpy, joblib

🔧 Setup Instructions
bash
Copy
Edit
git clone https://github.com/TidalWavesNode/candles_miner_pub
cd candles_miner_pub
python3 -m venv env
source env/bin/activate
pip install pandas torch scikit-learn numpy joblib
🧹 Prepare Your Dataset
Save your OHLC data as TVexport.csv:

lua
Copy
Edit
time,open,high,low,close[,volume]
Use data from TradingView or any API (Binance, MEXC, Kraken, etc.).

🛠️ Feature Engineering
Add features to your raw data:

bash
Copy
Edit
python3 feature_generator.py --input TVexport.csv --output TVexport_with_features.csv
Features include body/wick measurements and ratio metrics. Optional volume is retained if present.

🏋️‍♂️ Train the Model
bash
Copy
Edit
python3 train_updated.py --csv TVexport_with_features.csv --epochs 150 --batch-size 64 --lr 0.0005
Produces model.pth and scaler.pkl for use in predictions.

🔮 Predict the Next 24-Hourly Candles
bash
Copy
Edit
python3 predicthourly.py
This:

Loads the latest 24 candles from TVexport_with_features.csv

Predicts Green or Red for each

Outputs results to predictions_hourly.txt

⏱️ Output Format
less
Copy
Edit
Hour 1: Green (Confidence: 0.72)
Hour 2: Red (Confidence: 0.43)
...
Hour 24: Green (Confidence: 0.89)
🧠 How Confidence Works
The direction is predicted using real model output.

The confidence level is a blend of model probability + randomness for natural variability:

python
Copy
Edit
confidence = 0.3 × model_conf + 0.7 × random_noise
Ensures a realistic range (0.00 to 1.00)

Confidence reflects directional bias, but varies for realism

Useful for visualization, weighting, or UX design

⚠️ Disclaimer
This model is educational and experimental. It does not constitute financial advice or trading guidance. Use it responsibly and at your own risk.
