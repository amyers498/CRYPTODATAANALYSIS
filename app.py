import streamlit as st
import krakenex
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()
API_KEY = os.getenv('KRAKEN_API_KEY')
PRIVATE_KEY = os.getenv('PRIVATE_KEY')

# Initialize Kraken API with keys
api = krakenex.API(key=API_KEY, secret=PRIVATE_KEY)

# Define fees
MAKER_FEE = 0.0025  # 0.25%
TAKER_FEE = 0.0040  # 0.40%

# Function to fetch OHLC data from Kraken
def fetch_ohlc(pair, interval):
    response = api.query_public('OHLC', {'pair': pair, 'interval': interval})
    if 'error' in response and response['error']:
        st.error(f"Error fetching data: {response['error']}")
        return None
    pair_data = list(response['result'].values())[0]
    df = pd.DataFrame(pair_data, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
    return df

# Function to perform analysis
def analyze_crypto(data):
    data['rsi'] = RSIIndicator(data['close'], window=14).rsi()
    bb = BollingerBands(data['close'], window=20, window_dev=2)
    data['bb_upper'] = bb.bollinger_hband()
    data['bb_lower'] = bb.bollinger_lband()
    data['bb_middle'] = bb.bollinger_mavg()
    data['ema_9'] = EMAIndicator(data['close'], window=9).ema_indicator()
    data['ema_26'] = EMAIndicator(data['close'], window=26).ema_indicator()
    return data

# Function to calculate profitability
def calculate_profitability(investment, buy_price):
    maker_fee = investment * MAKER_FEE
    total_cost = investment + maker_fee
    breakeven_price = (total_cost + (total_cost * TAKER_FEE)) / (investment / buy_price)
    return maker_fee, breakeven_price

# Function to generate actionable insights for Quick Trade Mode
def quick_trade_analysis(data, investment):
    last_close = data['close'].iloc[-1]
    ema_9 = data['ema_9'].iloc[-1]
    ema_26 = data['ema_26'].iloc[-1]
    bb_middle = data['bb_middle'].iloc[-1]
    bb_upper = data['bb_upper'].iloc[-1]
    bb_lower = data['bb_lower'].iloc[-1]
    rsi = data['rsi'].iloc[-1]

    maker_fee, breakeven_price = calculate_profitability(investment, last_close)
    insights = []

    # Entry Recommendation
    if last_close > ema_9 and last_close > bb_middle:
        insights.append("Price is above EMA 9 and Bollinger Band middle line. Consider entering a trade now.")
    elif last_close <= bb_middle and last_close >= ema_26:
        insights.append("Price is near support levels (EMA 26 or middle Bollinger Band). Look for bullish confirmation to enter.")
    else:
        insights.append("Price is below key support levels. Wait for a better entry.")

    # Take-Profit Recommendation
    if last_close < bb_upper:
        take_profit = bb_upper
        insights.append(f"Set a take-profit target near the upper Bollinger Band at ${take_profit:.4f}.")
    else:
        insights.append("Price is already near the upper Bollinger Band. Consider exiting soon.")

    # Stop-Loss Recommendation
    stop_loss = max(bb_lower, ema_26)
    insights.append(f"Set a stop-loss near ${stop_loss:.4f} to manage risk.")

    # Fee & Break-Even Info
    insights.append(f"Break-even price (including fees): ${breakeven_price:.4f}")

    return insights, maker_fee, breakeven_price

# Streamlit UI
st.title("Crypto Analysis Dashboard")
st.sidebar.header("Settings")

# Input for Crypto Pair
pair = st.sidebar.text_input("Enter Crypto Pair (e.g., BTCUSD)", "BTCUSD")
interval = st.sidebar.selectbox("Select Timeframe (minutes)", options=[1, 5, 15, 30, 60, 240], index=4)
investment = st.sidebar.number_input("Enter Investment Amount (in $)", value=100.0, step=1.0)
quick_trade_mode = st.sidebar.checkbox("Enable Quick Trade Mode", value=False)
refresh_rate = st.sidebar.slider("Set Refresh Rate (seconds)", 5, 300, 30)

# Create placeholders for refresh updates
data_placeholder = st.empty()
insights_placeholder = st.empty()
charts_placeholder = st.empty()

while True:
    with st.spinner("Fetching data..."):
        data = fetch_ohlc(pair, interval)
        if data is not None:
            data = analyze_crypto(data)
            last_close = data['close'].iloc[-1]

            # Update Data Section
            with data_placeholder.container():
                st.write("### Latest Data Points")
                st.dataframe(data[['close', 'rsi', 'bb_upper', 'bb_middle', 'bb_lower', 'ema_9', 'ema_26']].tail())

            # Insights Section
            if quick_trade_mode:
                insights, maker_fee, breakeven_price = quick_trade_analysis(data, investment)
            else:
                insights, _, _ = quick_trade_analysis(data, investment)

            with insights_placeholder.container():
                st.write("### Insights")
                for insight in insights:
                    st.write(f"- {insight}")


            # Charts Section
            with charts_placeholder.container():
                st.write("### Bollinger Bands Chart")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(data.index, data['close'], label='Close Price', color='blue')
                ax.plot(data.index, data['bb_upper'], linestyle='--', color='red', label='Upper Band')
                ax.plot(data.index, data['bb_middle'], linestyle='--', color='orange', label='Middle Band')
                ax.plot(data.index, data['bb_lower'], linestyle='--', color='green', label='Lower Band')
                ax.legend()
                st.pyplot(fig)

                st.write("### RSI Chart")
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.plot(data.index, data['rsi'], color='purple', label='RSI')
                ax.axhline(70, linestyle='--', color='red', label='Overbought')
                ax.axhline(30, linestyle='--', color='green', label='Oversold')
                ax.legend()
                st.pyplot(fig)

    time.sleep(refresh_rate)
