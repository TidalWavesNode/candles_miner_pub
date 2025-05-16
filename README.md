# A lightweight machine learning pipeline that trains a model to predict hourly crypto candle directions (Green or Red) using engineered features. 

## 🧰 Requirements
Ubuntu 22.04
Python 3.10+
pip, venv

## 🛠️ Setup Instructions
### Clone the Repository

```
git clone https://github.com/NeuralNexusII/candles_miner.git
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
python3 train_updated.py --csv TVexport_with_features.csv --epochs 150 --batch-size 64 --lr 0.0005 --hidden 128
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

