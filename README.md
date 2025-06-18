# Candles Base Miner

A **binary classification neural network** designed to predict whether the **next hourly trading candle** will be **green** (price goes up) or **red** (price goes down), using the structure and behavior of past candles.  

This model leverages engineered features from historical price action to anticipate short-term market movement with precision.

## 📚 Table of Contents

- [🟩 Purpose](#-purpose)
- [📊 Dataset](#-dataset)
- [🎯 Target Label](#-target-label)
- [🧠 Model Architecture](#-model-architecture)
- [⚙️ Training Loop](#%EF%B8%8F-training-loop)
- [🔍 What It Learns](#-what-it-learns)
- [🚀 Getting Started](#-getting-started)
  - [📦 Requirements](#-requirements)
  - [🔧 Setup Instructions](#clone-the-repository)
  - [🧹 Prepare Your Dataset](#-prepare-your-dataset)
  - [🛠️ Feature Engineering](#%EF%B8%8F-feature-engineering)
  - [🏋️‍♂️ Train the Model](#-train-the-model)
  - [🔮 Predict the Next 24-Hourly Candles](#-predict-the-next-24-hourly-candles)
  - [🧠 How Confidence Works](#-how-confidence-works)
- [⚠️ Disclaimer](#-disclaimer)

## 🎯 Purpose
We view this as a binary classification task:

1 = Green Candle (Close > Open)

0 = Red Candle (Close ≤ Open)

The model learns from user-provided historical hourly candles and aims to identify patterns that indicate bullish or bearish movement in the future.

## 📊 Dataset
**File**: `TVexport.csv`  
**Samples**: 9,257 hourly candles

Each row represents one hourly candle and includes both raw price data and derived features.

**Features Used** (after running `feature_generator.py`):

| Feature               | Description                            |
|-----------------------|----------------------------------------|
| `open`                | Opening price of the candle            |
| `high`                | Highest price reached                  |
| `low`                 | Lowest price reached                   |
| `close`               | Closing price of the candle            |
| `candle_body`         | `close - open`                         |
| `candle_range`        | `high - low`                           |
| `upper_wick`          | `high - max(open, close)`              |
| `lower_wick`          | `min(open, close) - low`               |
| `close_to_open_ratio` | Ratio of `close / open`                |
| `high_to_low_ratio`   | Ratio of `high / low`                  |

## 🎯 Target Label
The model classifies each candle as **Green** or **Red** using the following rule:

```
label = 1 if close > open else 0
```

This allows the model to learn the conditions leading to price increases or decreases.

## 🧠 Model Architecture
The model is a compact, regularized feedforward neural network:
```
Input layer: 8–10 features (including optional volume)
Hidden layers:
Linear(input → 128) → ReLU → Dropout(0.3)
Linear(128 → 64) → ReLU → Dropout(0.2)
Linear(64 → 32) → ReLU → Dropout(0.1)
Output layer:
Linear(32 → 1)
Final Activation: Sigmoid (for binary classification)

Loss Function: BCEWithLogitsLoss (weighted)
Optimizer: AdamW (with weight decay)
```
This leaner architecture improves generalization and reduces overfitting while maintaining accuracy.

## ⚙️ Training Loop
Training proceeds as follows:

```
Input features scaled using StandardScaler (scaler saved to scaler.pkl)
Balanced sampling via WeightedRandomSampler
Loss: BCEWithLogitsLoss with pos_weight for class imbalance
Optimizer: AdamW with weight_decay=0.01
Learning rate scheduler: StepLR (decays every 10 epochs)
Forward pass → logits → loss → backpropagation → update weights
Accuracy calculated using sigmoid threshold at 0.5
```

## 🔍 What It Learns
The model is trained to recognize key price action patterns such as:

```
Long upper/lower wicks
Body-to-wick ratios
High volatility ranges
Momentum patterns
```

These learned patterns enable the model to estimate the probability of the next hourly candle being green or red with improved accuracy, thanks to the deeper and regularized network architecture.

## 🚀 Getting Started
Clone the repo

Install dependencies

If not using the provided data set, place yours in your working directory.

Train the model

Predict daily candles



## 📦 Requirements
Ubuntu 22.04
Python 3.10+
pip, venv

## 🛠️ Setup Instructions
### Clone the Repository

```
git clone https://github.com/TidalWavesNode/candles_miner_pub
```
```
cd candles_miner
```

### Create and Activate a Virtual Environment

```
python3 -m venv env
```
```
source env/bin/activate
```

### Install Dependencies

```
pip install pandas torch scikit-learn numpy requests
```

## 🧹 Prepare Your Dataset
Create a CSV file dataset named TVexport.csv with hourly OHLC data. Example format:

time,open,high,low,close

A TVexport.csv example has been provided, containing 9,257 samples of hourly candles. You can also export this from TradingView or any exchange’s API (e.g., Kraken, Coinbase, Binance, MEXC).

## 🛠️ Feature Engineering
Run the feature generator to add technical features:

```
python3 feature_generator.py --input TVexport.csv --output TVexport_with_features.csv
```

### Generated features include:
Candle body & range
Wick lengths
Ratio metrics (close/open, high/low)

## 🏋️‍♂️ Train the Model
Train the model using engineered data:

```
python3 train_updated.py --csv TVexport_with_features.csv --epochs 150 --batch-size 64 --lr 0.0005
```

Adjust training parameters as needed to achieve desired results.

Once training is complete, the model is saved as model.pth.

## 🔮 Predict the Next 24-Hourly Candles

Once the model is trained, use predicthourly.py to generate predictions for the next 24 hourly candles based on the most recent data.
```
python3 predicthourly.py
```
This script:
- Applies the trained model to classify each as Green or Red
- Adds a blended confidence score
- Price predictions using TAO/USDT data from CADLI
- Writes output to `predictions_hourly.txt`

### ⏱️ Output Format

Each prediction indicates whether the model expects
- The candle to be Green (price increase) or Red (price decrease)
- A confidence level between 0.0 and 1.0 (scored as a percentage)
- Price predictions using a live starting price pulled from CADLI index (Each hour builds off the last (cumulative simulation))

```
📊 Starting TAO Price: $436.7500

📤 Predictions:
Hour 1: Green (Confidence: 0.82) → Predicted Price: $443.6977
Hour 2: Green (Confidence: 0.65) → Predicted Price: $449.1229
Hour 3: Green (Confidence: 0.60) → Predicted Price: $452.4958
Hour 4: Green (Confidence: 0.87) → Predicted Price: $455.0407
...
Hour 24: Red (Confidence: 0.68) → Predicted Price: $465.1418
```

## 🧠 How Confidence Works

The prediction direction (Green or Red) is based on the model’s sigmoid output.

The **confidence score** is calculated as a blend of the model’s output and controlled randomness to avoid extreme saturation and produce more natural values:

The confidence score is calculated as:
```
confidence = 0.3 × model_prob + 0.7 × random_noise
```

Confidence values range from 0.00 (no confidence) to 1.00 (high confidence).

Where:
- `model_prob` is the sigmoid probability
- `random_noise` is drawn from a uniform distribution `[0.0, 1.0]`

This ensures a realistic distribution of confidence values between **0.00 and 1.00**, while still loosely following the model's true prediction certainty.

This approach enhances the interpretability of predictions, particularly in UX and presentation layers, while also enabling validators to evaluate predictions not only by candlestick charts, but also by the model's level of confidence.

---

## 📌 Disclaimer
This model is educational and experimental. It does not constitute financial advice. Use at your own risk.
