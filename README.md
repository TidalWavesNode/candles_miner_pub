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
The model is a deep feedforward neural network with a funnel-shaped structure:
```
Input layer: 10 features
Hidden layers:
Linear(10 → 256) → ReLU → Dropout(0.2)
Linear(256 → 256) → ReLU → Dropout(0.2)
Linear(256 → 128) → ReLU → Dropout(0.1)
Linear(128 → 64) → ReLU
Output layer: Linear(64 → 1)
Final Activation: Sigmoid (for binary classification)
Regularization: Dropout layers to reduce overfitting
Loss Function: BCEWithLogitsLoss
Optimizer: Adam (or optionally SGD)
```

## ⚙️ Training Loop
Training proceeds as follows:

```
Inputs are normalized using StandardScaler
Batch size: 64
Epochs: Configurable
Forward pass → raw logits
Loss is computed with BCEWithLogitsLoss
Backpropagation with optimizer step
Accuracy is calculated using a 0.5 threshold on sigmoid output
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
pip install pandas torch scikit-learn numpy
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

```
python3 predicthourly.py
```
Displays hourly predictions (Green/Red)
Saves them to predictions_hourly.txt

✅ Sample Output

```
📤 Predictions:
Hour 1: Green
Hour 2: Green
Hour 3: Green
Hour 4: Green
Hour 5: Red
Hour 6: Green
Hour 7: Red
Hour 8: Red
Hour 9: Red
Hour 10: Red
Hour 11: Red
Hour 12: Green
Hour 13: Red
Hour 14: Red
Hour 15: Green
Hour 16: Green
Hour 17: Red
Hour 18: Green
Hour 19: Green
Hour 20: Green
Hour 21: Red
Hour 22: Red
Hour 23: Green
Hour 24: Red
```

## 📌 Disclaimer
This model is educational and experimental. It does not constitute financial advice. Use at your own risk.
