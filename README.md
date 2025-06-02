## A binary classification neural network designed to predict whether the next hourly trading candle will be green (price goes up) or red (price goes down), based on the behavior of historical trading candles.

Table of Contents:

[Purpose](https://github.com/NeuralNexusII/candles_miner/blob/main/README.md#-problem-statement)

[Dataset](https://github.com/NeuralNexusII/candles_miner/blob/main/README.md#-dataset)

[Target Label](https://github.com/NeuralNexusII/candles_miner/blob/main/README.md#-target-label)

[Model Architecture](https://github.com/NeuralNexusII/candles_miner/blob/main/README.md#-model-architecture)

[Training Loop](https://github.com/NeuralNexusII/candles_miner/blob/main/README.md#%EF%B8%8F-training-loop)

[What it Learns](https://github.com/NeuralNexusII/candles_miner/blob/main/README.md#-what-it-learns)

[Getting Started](https://github.com/NeuralNexusII/candles_miner/blob/main/README.md#-getting-started)

[Requirements](https://github.com/NeuralNexusII/candles_miner/blob/main/README.md#-requirements)

[Setup Instructions](https://github.com/NeuralNexusII/candles_miner/blob/main/README.md#%EF%B8%8F-setup-instructions)

[Prepare Your Dataset](https://github.com/NeuralNexusII/candles_miner/blob/main/README.md#-prepare-your-dataset)

[Feature Engineering](https://github.com/NeuralNexusII/candles_miner/blob/main/README.md#%EF%B8%8F-feature-engineering)

[Train the Model](https://github.com/NeuralNexusII/candles_miner/blob/main/README.md#-train-the-model)

[Predict the next 24-Hourly Candles](https://github.com/NeuralNexusII/candles_miner/blob/main/README.md#-predict-the-next-24-hourly-candles)

[Disclaimer](https://github.com/NeuralNexusII/candles_miner/blob/main/README.md#-disclaimer)

## 📈 Purpose
We view this as a binary classification task:

1 = Green Candle (Close > Open)

0 = Red Candle (Close ≤ Open)

The model learns from user-provided historical hourly candles and aims to identify patterns that indicate bullish or bearish movement in the next future.

## 🔢 Dataset
File: MTexport.csv
Samples: 9,257 hourly candles

Each row represents one hourly candle and includes both raw price data and derived features.

Features Used: (Once feature_generator.py has been executed on the provided dataset)
```
Feature - Description
open - Opening price of the candle
high - Highest price reached
low - Lowest price reached
close - Closing price of the candle
candle_body - close - open
candle_range - high - low
upper_wick - high - max(open, close)
lower_wick - min(open, close) - low
```
## 🎯 Target Label
The label is generated as:

`
label = 1 if close > open else 0
`
This allows the model to learn the conditions leading to price increases or decreases.

## 🧠 Model Architecture
We're using a feedforward neural network with:
```
Input layer: 8 features
Hidden layer: Customizable size (default: 128 neurons)
Output layer: 1 neuron for binary prediction
Activation: ReLU between layers
Loss Function: BCEWithLogitsLoss
Optimizer: Typically Adam or SGD
```

## ⚙️ Training Loop
Training proceeds as follows:

```
Normalize inputs using StandardScaler
Batch size: Typically 64 samples
Forward pass → raw logits
Loss computed with BCEWithLogitsLoss
Backpropagation and optimizer update
Accuracy: Threshold predictions at 0.5
```

## 💡 What It Learns
The model is trained to recognize candle formations, such as:

```
Long upper/lower wicks
Large vs. small body ratios
Candle volatility
```

These insights help it estimate the likelihood of upward or downward movement in the next hour.

## 🚀 Getting Started
Clone the repo

Install dependencies

If not using the provided data set, place yours in your working directory

Train the model

Predict daily candles



## 🧰 Requirements
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

### 📄 Prepare Your Dataset
Create a CSV file dataset named TVexport.csv with hourly OHLC data. Example format:

time,open,high,low,close

A TVexport.csv example has been provided that contains 9257 samples of hourly candles. You can also export this from TradingView or any exchange’s API (e.g., Kraken, Coinbase, Binance, MEXC).

### ⚙️ Feature Engineering
Run the feature generator to add technical features:

```
python3 feature_generator.py --input TVexport.csv --output TVexport_with_features.csv
```

### Generated features include:
Candle body & range

Wick lengths

Ratio metrics (close/open, high/low)

### 🧠 Train the Model
Train the model using engineered data:

```
python3 train_updated.py --csv TVexport_with_features.csv --epochs 150 --batch-size 64 --lr 0.0005
```

Adjust training parameters as needed to achieve desired results.

Once training is complete, the model is saved as model.pth.

### 🔮 Predict the Next 24-Hourly Candles

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
This model is educational and experimental. It does not constitute financial advice. Use it at your own risk.
