#A lightweight machine learning pipeline that trains a model to predict hourly crypto candle directions (Green or Red) using engineered features. 

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
Create a CSV file named TVexport.csv with hourly OHLC data. Example format:

time,open,high,low,close
Export this from TradingView or your exchange’s API (e.g., Binance, MEXC).

### ⚙️ Feature Engineering
Run the feature generator to add technical features:

```
python3 feature_generator.py --csv TVexport.csv --output TVexport_with_features.csv
```

### Generated features include:
Candle body & range
Wick lengths
Ratio metrics (close/open, high/low)

### 🧠 Train the Model
Train the model using engineered data:

```
ython3 train_updated.py --csv TVexport_with_features.csv --epochs 150 --batch-size 64 --lr 0.0005 --hidden 128
```

Once complete this saves your model to model.pth.

### 🔮 Predict the Next 24 Hourly Candles

```
python3 predicthourly.py
```
Displays hourly predictions (Green/Red)
Saves them to predictions_hourly.txt

✅ Sample Output

```📤 Predictions:
Hour 1: Green
Hour 2: Red
...
Hour 24: Red```

