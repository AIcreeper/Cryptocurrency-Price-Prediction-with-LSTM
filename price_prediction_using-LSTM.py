import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import talib as ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from binance import Client
import logging
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("BINANCE_API_KEY")
API_SECRET = os.environ.get("BINANCE_API_SECRET")
INTERVAL = Client.KLINE_INTERVAL_5MINUTE
INITIAL_TRAINING_PERIOD_DAYS = 120
RETRAINING_PERIOD_DAYS = 35
PREDICTION_PERIODS = {
    "1_hour": 12,
    "6_hours": 72,
    "12_hours": 144,
    "1_day": 288,
    "3_days": 864,
    "1_week": 2016,
}
WAIT_TIME = 30 * 60
LOOK_BACK = 60
LSTM_UNITS = 100  # Increased units
DROPOUT_RATE = 0.3  # Increased rate
LEARNING_RATE = 0.001
EPOCHS = 150  # Increased epochs
BATCH_SIZE = 64  # Increased Batch Size

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("prediction_bot_lstm.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()
client = Client(API_KEY, API_SECRET)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    logging.info("CUDA is available. Using GPU.")
else:
    logging.info("CUDA not available. Using CPU.")


def get_historical_data(symbol, interval, num_days):
    logging.info(f"Fetching historical data for {symbol}...")
    end_time = int(time.time() * 1000)
    start_time = end_time - (num_days * 24 * 60 * 60 * 1000)
    klines = client.get_historical_klines(symbol, interval, start_time, end_time)

    df = pd.DataFrame(
        klines,
        columns=[
            "Open time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close time",
            "Quote asset volume",
            "Number of trades",
            "Taker buy base asset volume",
            "Taker buy quote asset volume",
            "Ignore",
        ],
    )
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
    df = df.set_index("Open time")
    numerical_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Quote asset volume",
        "Number of trades",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
    ]
    df[numerical_cols] = df[numerical_cols].astype(float)

    return df


def add_features(data):
    data["SMA"] = ta.SMA(data["Close"], timeperiod=28)
    data["RSI"] = ta.RSI(data["Close"], timeperiod=28)
    data["MACD"], data["Signal"], data["Hist"] = ta.MACD(
        data["Close"], fastperiod=24, slowperiod=52, signalperiod=18
    )
    return data


def prepare_data_for_lstm(data, look_back=LOOK_BACK, prediction_period=1):
    data_copy = data.copy()
    data_copy.dropna(subset=["Close", "SMA", "RSI", "MACD", "Signal", "Hist"], inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(
        data_copy[["Close", "SMA", "RSI", "MACD", "Signal", "Hist"]]
    )

    X = []
    y = []

    for i in range(look_back, len(scaled_data) - prediction_period):
        X.append(scaled_data[i - look_back : i, :])
        y.append(scaled_data[i + prediction_period, 0])

    X, y = np.array(X), np.array(y)

    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler.fit(data_copy[["Close"]])

    return X, y, scaler, close_scaler


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True) # Bidirectional LSTM layer
        self.dropout1 = nn.Dropout(DROPOUT_RATE)
        self.lstm2 = nn.LSTM(hidden_size*2, hidden_size, batch_first=True, bidirectional=True) # Added second LSTM layer 
        self.dropout2 = nn.Dropout(DROPOUT_RATE)
        self.fc = nn.Linear(hidden_size*2, output_size) # Adjusted to take output from the second bidirectional LSTM layer

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out[:, -1, :])
        out, _ = self.lstm2(out.unsqueeze(1)) #Feeding dropout out through another LSTM layer
        out = self.dropout2(out[:, -1, :])
        out = self.fc(out)
        return out


def train_lstm_model(X_train, y_train, input_size, symbol):
    logging.info(f"Training LSTM model for {symbol}...")
    model = LSTMModel(input_size=input_size, hidden_size=LSTM_UNITS, output_size=1).to(
        device
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output.squeeze(), y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

    return model


def predict_price_change(model, data, scaler, close_scaler, symbol, prediction_period):
    model.eval()
    last_data = data.tail(LOOK_BACK).copy()
    last_data.dropna(subset=["Close", "SMA", "RSI", "MACD", "Signal", "Hist"], inplace=True)

    if len(last_data) < LOOK_BACK:
        logging.warning(f"Not enough data for prediction for {symbol}.")
        return None

    scaled_last_data = scaler.transform(
        last_data[["Close", "SMA", "RSI", "MACD", "Signal", "Hist"]]
    )
    X_pred = np.array([scaled_last_data])

    X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32).to(device)

    with torch.no_grad():
        predicted_scaled_close = model(X_pred_tensor).cpu().numpy()[0][0]

    predicted_scaled_close = np.array([[predicted_scaled_close]])
    predicted_close_price = close_scaler.inverse_transform(predicted_scaled_close)[0][0]

    last_close_price = last_data["Close"].iloc[-1]
    predicted_change_usd = predicted_close_price - last_close_price

    return predicted_change_usd


def get_current_price(symbol):
    try:
        logging.info(f"Fetching current price for {symbol}...")
        ticker = client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker["price"])
        logging.info(f"Current price of {symbol} in USDT: {current_price:.2f} USDT")
        return current_price
    except Exception as e:
        logging.error(f"Error fetching current price for {symbol}: {str(e)}")
        return None


def main():
    logging.info("Starting prediction bot (LSTM)...")
    symbols = input(
        "Enter the coin pairs to predict (comma-separated, e.g., BTCUSDT, ETHUSDT, DOGEUSDT): "
    ).upper().split(",")
    symbols = [symbol.strip() for symbol in symbols]

    while True:
        for symbol in symbols:
            current_price = get_current_price(symbol)
            if current_price:
                logging.info(f"Current price for {symbol} in USDT: {current_price:.2f} USDT")

            logging.info(f"Training initial model for {symbol}...")
            data = get_historical_data(symbol, INTERVAL, INITIAL_TRAINING_PERIOD_DAYS)
            df_features = add_features(data.copy())
            
            predictions = {}
            
            for time_horizon, period in PREDICTION_PERIODS.items():
                X, y, scaler, close_scaler = prepare_data_for_lstm(df_features, prediction_period=period)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                
                if X_train.size > 0:
                    model = train_lstm_model(
                        X_train, y_train, input_size=X_train.shape[2], symbol=symbol
                    )
                
                    predicted_change = predict_price_change(model, df_features, scaler, close_scaler, symbol, prediction_period=period)
                    predictions[time_horizon] = predicted_change
                    if predicted_change:
                        logging.info(f"Initial predicted price change for {symbol} in {time_horizon}: {predicted_change:.2f} USDT")
                    else:
                      logging.warning(f"Could not get the initial prediction for {symbol} in {time_horizon}")
                else:
                    logging.warning(f"No training data for {symbol} at {time_horizon} ")

            logging.info(
                f"Retraining model for {symbol} with the last {RETRAINING_PERIOD_DAYS} days of data..."
            )
            data = get_historical_data(symbol, INTERVAL, RETRAINING_PERIOD_DAYS)
            df_features = add_features(data.copy())
            
            retrained_predictions = {}
            
            for time_horizon, period in PREDICTION_PERIODS.items():
                X, y, scaler, close_scaler = prepare_data_for_lstm(df_features, prediction_period=period)

                if len(X) > 0:
                    model = train_lstm_model(X, y, input_size=X.shape[2], symbol=symbol)
                    predicted_change_retrain = predict_price_change(model, df_features, scaler, close_scaler, symbol, prediction_period=period)
                    retrained_predictions[time_horizon] = predicted_change_retrain
                    if predicted_change_retrain:
                        logging.info(f"Retrained predicted price change for {symbol} in {time_horizon}: {predicted_change_retrain:.2f} USDT")
                    else:
                        logging.warning(f"Could not get the retrained prediction for {symbol} in {time_horizon}")
                else:
                    logging.warning(f"Insufficient data for retraining {symbol} at {time_horizon}.")

        logging.info("Waiting for 30 minutes before restarting the process for all coins...")
        time.sleep(WAIT_TIME)


if __name__ == "__main__":
    main()