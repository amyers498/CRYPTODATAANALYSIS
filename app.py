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
def calculate_profitability(investment, entry_price, take_profit_price, stop_loss_price):
    maker_fee = investment * MAKER_FEE
    effective_investment = investment - maker_fee
    crypto_quantity = effective_investment / entry_price
    total_cost = effective_investment + maker_fee
    profit_take_profit = (crypto_quantity * take_profit_price) - total_cost
    loss_stop_loss = (crypto_quantity * stop_loss_price) - total_cost
    breakeven_price = (total_cost / crypto_quantity) * (1 + TAKER_FEE)
    return maker_fee, total_cost, profit_take_profit, loss_stop_loss, breakeven_price

# Function to generate actionable insights for Quick Trade Mode
def quick_trade_analysis(data, investment, effective_price, custom_take_profit=None, custom_stop_loss=None):
    last_close = data['close'].iloc[-1]
    take_profit = custom_take_profit if custom_take_profit and custom_take_profit > 0 else data['bb_upper'].iloc[-1]
    stop_loss = custom_stop_loss if custom_stop_loss and custom_stop_loss > 0 else max(data['bb_lower'].iloc[-1], data['ema_26'].iloc[-1])
    maker_fee, total_cost, profit_take_profit, loss_stop_loss, breakeven_price = calculate_profitability(
        investment, effective_price, take_profit, stop_loss
    )
    insights = []
    if last_close > data['ema_9'].iloc[-1] and last_close > data['bb_middle'].iloc[-1]:
        insights.append("Price is above EMA 9 and Bollinger Band middle line. Consider entering a trade now.")
    elif last_close <= data['bb_middle'].iloc[-1] and last_close >= data['ema_26'].iloc[-1]:
        insights.append("Price is near support levels (EMA 26 or middle Bollinger Band). Look for bullish confirmation to enter.")
    else:
        insights.append("Price is below key support levels. Wait for a better entry.")
    insights.append(f"Set a take-profit target at ${take_profit:.4f}.")
    insights.append(f"Set a stop-loss at ${stop_loss:.4f} to manage risk.")
    insights.append(f"Break-even price (including fees): ${breakeven_price:.4f}")
    return insights, maker_fee, total_cost, profit_take_profit, loss_stop_loss, take_profit, stop_loss, breakeven_price

# Streamlit UI
st.title("Crypto Analysis Dashboard")
st.sidebar.header("Settings")

# Inputs
pair = st.sidebar.text_input("Enter Crypto Pair (e.g., BTCUSD)", "BTCUSD")
interval = st.sidebar.selectbox("Select Timeframe (minutes)", options=[1, 5, 15, 30, 60, 240], index=4)
investment = st.sidebar.number_input("Enter Investment Amount (in $)", value=100.0, step=1.0)

# Toggle for using actual entry price
use_actual_price = st.sidebar.checkbox("Use My Entry Price", value=False)
actual_entry_price = None
if use_actual_price:
    actual_entry_price = st.sidebar.number_input("Enter Your Entry Price ($)", value=0.0, step=0.000001, format="%.6f")

# Toggle for custom take-profit and stop-loss
use_custom_targets = st.sidebar.checkbox("Use Custom Take-Profit and Stop-Loss", value=False)
custom_take_profit = None
custom_stop_loss = None
if use_custom_targets:
    custom_take_profit = st.sidebar.number_input("Enter Custom Take-Profit Price ($)", value=0.0, step=0.000001, format="%.6f")
    custom_stop_loss = st.sidebar.number_input("Enter Custom Stop-Loss Price ($)", value=0.0, step=0.000001, format="%.6f")

refresh_rate = st.sidebar.slider("Set Refresh Rate (seconds)", 5, 300, 30)

# Placeholders for updates
data_placeholder = st.empty()
insights_placeholder = st.empty()
charts_placeholder = st.empty()

# Main loop
while True:
    with st.spinner("Fetching data..."):
        data = fetch_ohlc(pair, interval)
        if data is not None:
            data = analyze_crypto(data)
            last_close = data['close'].iloc[-1]
            effective_price = actual_entry_price if use_actual_price and actual_entry_price > 0 else last_close
            insights, maker_fee, total_cost, profit_take_profit, loss_stop_loss, take_profit, stop_loss, breakeven_price = quick_trade_analysis(
                data, investment, effective_price, custom_take_profit, custom_stop_loss
            )

            with data_placeholder.container():
                st.write("### Latest Data Points")
                st.dataframe(data[['close', 'rsi', 'bb_upper', 'bb_middle', 'bb_lower', 'ema_9', 'ema_26']].tail())

            with insights_placeholder.container():
                st.write("### Insights")
                for insight in insights:
                    st.write(f"- {insight}")
                st.write("### Profitability Analysis")
                st.write(f"**Investment Amount**: ${investment:.2f}")
                if use_actual_price:
                    st.write(f"**Your Entry Price**: ${actual_entry_price:.6f}")
                st.write(f"**Current Price**: ${last_close:.4f}")
                st.write(f"**Maker Fee**: ${maker_fee:.4f}")
                st.write(f"**Total Cost (after fees)**: ${total_cost:.4f}")
                st.write(f"**Potential Profit at Take Profit ({take_profit:.4f})**: ${profit_take_profit:.4f}")
                st.write(f"**Potential Loss at Stop Loss ({stop_loss:.4f})**: ${loss_stop_loss:.4f}")
                st.write(f"**Break-Even Price (including taker fee)**: ${breakeven_price:.4f}")

            with charts_placeholder.container():
                st.write("### Bollinger Bands Chart")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(data.index, data['close'], label='Close Price', color='blue')
                ax.plot(data.index, data['bb_upper'], linestyle='--', color='red', label='Upper Band')
                ax.plot(data.index, data['bb_middle'], linestyle='--', color='orange', label='Middle Band')
                ax.plot(data.index, data['bb_lower'], linestyle='--', color='green', label='Lower Band')
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)  # Close the figure to free memory

                st.write("### RSI Chart")
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.plot(data.index, data['rsi'], color='purple', label='RSI')
                ax.axhline(70, linestyle='--', color='red', label='Overbought')
                ax.axhline(30, linestyle='--', color='green', label='Oversold')
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)  # Close the figure to free memory

    time.sleep(refresh_rate)
