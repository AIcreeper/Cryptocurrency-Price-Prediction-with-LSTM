# Cryptocurrency Price Prediction with LSTM

## Overview
This project utilizes an LSTM (Long Short-Term Memory) neural network to predict cryptocurrency coin prices. The script fetches the last 120 days of price data from Binance for a given coin pair and uses this data to train the LSTM model. The trained model can then predict the price for the next 1 hour, 3 hours, 12 hours, 1 day, 3 days, and 7 days.

## Features
- **Coin Pair Input**: The user inputs the desired coin pair (e.g., BTC/USDT, ETH/BTC).
- **Data Fetching**: Fetches the last 120 days of price data from Binance.
- **Price Prediction**: Predicts the price for the next 1 hour, 3 hours, 12 hours, 1 day, 3 days, and 7 days using LSTM.
- **LSTM Model Training**: The model is trained using the fetched data to forecast future prices.

## Technologies Used
- **Python 3.x**
- **TensorFlow/Keras** for building and training the LSTM model
- **Binance API** for fetching historical price data
- **Pandas** for data manipulation
- **NumPy** for numerical operations
- **Matplotlib** for data visualization

## Installation

### Requirements
- Python 3.8+
- Install the required dependencies by running the following command:
   ```bash
   pip install -r requirements.txt
