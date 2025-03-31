import asyncio
import aiohttp
import pandas as pd
import numpy as np
import os
import time
import subprocess # For notifications (optional)
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from aiohttp import BasicAuth

# --- Luno API Configuration ---
# IMPORTANT: Set these environment variables before running the script
# export LUNO_API_KEY_ID='your_key_id'
# export LUNO_API_KEY_SECRET='your_key_secret'
LUNO_API_KEY_ID = os.getenv('LUNO_API_KEY_ID')
LUNO_API_KEY_SECRET = os.getenv('LUNO_API_KEY_SECRET')
API_BASE_URL = "https://api.luno.com/api/exchange/1" # Luno Exchange API base

# --- Agent Configuration ---
PAIR_TO_ANALYZE = "SOLMYR" # Luno pair to trade (e.g., XBTZAR, ETHMYR)
# Luno uses duration in seconds. 86400 = 1 day
TIMEFRAME_SECONDS = 86400
CANDLE_LIMIT = 250 # Number of candles to fetch for analysis

# --- Trend Indicator Settings ---
EMA_SHORT_LENGTH = 21
EMA_LONG_LENGTH = 55
ATR_LENGTH = 10
SUPERTREND_FACTOR = 3.0

# --- Agent State ---
current_signal = "NEUTRAL" # Can be NEUTRAL, BUY, HOLD

# --- Continuous Run Settings ---
CHECK_INTERVAL_SECONDS = 1 * 60 * 60 # Check every 1 hour

# --- Notification Function (Optional) ---
def send_agent_notification(symbol, signal, price):
    title = f"Luno Trend Agent: {symbol}"
    message = f"New Signal: {signal} @ {price}"
    try:
        if subprocess.run(['which', 'notify-send'], capture_output=True, text=True).returncode == 0:
            subprocess.run(['notify-send', title, message], check=False, timeout=10)
            print(f"--- Notification attempted: {signal} for {symbol} ---")
        else:
            print("Warning: 'notify-send' command not found. Skipping notification.")
    except Exception as e: print(f"Warning: Failed to send agent notification: {e}")

# --- Data Fetching (Adapted for Luno Candles) ---
async def get_luno_candles(session, pair, duration_seconds, limit):
    """Fetches candlestick data from Luno Exchange API."""
    if not LUNO_API_KEY_ID or not LUNO_API_KEY_SECRET:
        print("Error: LUNO_API_KEY_ID or LUNO_API_KEY_SECRET not set.")
        return None

    now_ms = int(time.time() * 1000)
    since_ms = now_ms - (limit * duration_seconds * 1000)
    url = f"{API_BASE_URL}/candles"
    params = {'pair': pair, 'since': since_ms, 'duration': duration_seconds}
    auth = BasicAuth(LUNO_API_KEY_ID, LUNO_API_KEY_SECRET)

    try:
        async with session.get(url, params=params, auth=auth) as response:
            if response.status == 401:
                print(f"Error: Luno API authentication failed (401). Check API keys/permissions for Exchange API.")
                return None
            response.raise_for_status()
            data = await response.json()
            candles = data.get('candles', [])

            min_len_needed = max(EMA_LONG_LENGTH, ATR_LENGTH + 1) # Min length for indicators
            if candles and len(candles) >= min_len_needed:
                df = pd.DataFrame(candles)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                return df
            else:
                print(f"Warning: Insufficient data for {pair} {duration_seconds}s. Got {len(candles)}, needed {min_len_needed}.")
                return None
    except aiohttp.ClientResponseError as e:
        print(f"HTTP Error fetching {pair} {duration_seconds}s: {e.status} {e.message}")
        return None
    except Exception as e:
        print(f"Fetch error {pair} {duration_seconds}s: {e}")
        return None

# --- Indicator Calculations ---
def calculate_ema(series, length):
    if len(series) < length: return pd.Series(np.nan, index=series.index)
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def calculate_atr(high, low, close, length=14):
     if len(close) < length + 1: return pd.Series(np.nan, index=close.index)
     high_low = high - low; high_close = np.abs(high - close.shift()); low_close = np.abs(low - close.shift())
     tr_df = pd.concat([high_low, high_close, low_close], axis=1)
     tr = tr_df.max(axis=1, skipna=False)
     atr = tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
     return atr

def calculate_supertrend(high, low, close, atr_length=10, factor=3.0):
    if len(close) < atr_length + 1: return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)
    atr = calculate_atr(high, low, close, length=atr_length)
    if atr.isnull().all(): return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)

    hl2 = (high + low) / 2
    upper_band = hl2 + (factor * atr)
    lower_band = hl2 - (factor * atr)
    supertrend = pd.Series(np.nan, index=close.index)
    trend = pd.Series(1, index=close.index)

    for i in range(1, len(close)):
        prev_close = close.iloc[i-1]; prev_upper = upper_band.iloc[i-1]; prev_lower = lower_band.iloc[i-1]; prev_trend = trend.iloc[i-1]
        if pd.isna(prev_close) or pd.isna(prev_upper) or pd.isna(prev_lower): trend.iloc[i] = prev_trend
        elif prev_close > prev_upper: trend.iloc[i] = 1
        elif prev_close < prev_lower: trend.iloc[i] = -1
        else: trend.iloc[i] = prev_trend
        current_lower = lower_band.iloc[i]; current_upper = upper_band.iloc[i]
        if trend.iloc[i] == 1:
            lower_band.iloc[i] = max(current_lower if pd.notna(current_lower) else -np.inf, prev_lower if pd.notna(prev_lower) else -np.inf)
            supertrend.iloc[i] = lower_band.iloc[i]
        else:
            upper_band.iloc[i] = min(current_upper if pd.notna(current_upper) else np.inf, prev_upper if pd.notna(prev_upper) else np.inf)
            supertrend.iloc[i] = upper_band.iloc[i]

    direction = pd.Series(np.where(close > supertrend, 1, -1), index=close.index)
    direction.ffill(inplace=True)
    return supertrend, direction

# --- Trend Analysis Logic ---
def analyze_trend(df):
    """Analyzes the trend based on EMA and Supertrend."""
    if df is None or df.empty:
        return "NEUTRAL", "No data"

    # Calculate indicators
    df['EMA_Short'] = calculate_ema(df['close'], length=EMA_SHORT_LENGTH)
    df['EMA_Long'] = calculate_ema(df['close'], length=EMA_LONG_LENGTH)
    df['Supertrend'], df['ST_Direction'] = calculate_supertrend(df['high'], df['low'], df['close'], atr_length=ATR_LENGTH, factor=SUPERTREND_FACTOR)

    # Check latest values
    latest = df.iloc[-1]
    if latest.isnull().any(): # Check if any latest indicator is NaN
         return "NEUTRAL", "Indicators not ready"

    # --- Define Trend Conditions ---
    is_bullish_trend = (latest['close'] > latest['EMA_Long']) and \
                       (latest['EMA_Short'] > latest['EMA_Long']) and \
                       (latest['ST_Direction'] == 1)

    is_bearish_trend = (latest['close'] < latest['EMA_Long']) and \
                       (latest['EMA_Short'] < latest['EMA_Long']) and \
                       (latest['ST_Direction'] == -1)

    # --- Generate Signal ---
    new_signal = "NEUTRAL" # Default
    if is_bullish_trend:
        new_signal = "HOLD" # Stay in if already bullish
        if current_signal != "HOLD" and current_signal != "BUY": # Check previous state
             new_signal = "BUY" # Signal entry
    elif is_bearish_trend:
        new_signal = "NEUTRAL" # Exit or stay out
        # Could add a "SELL" signal here if managing positions

    reason = f"Close={latest['close']:.2f}, EMA_S={latest['EMA_Short']:.2f}, EMA_L={latest['EMA_Long']:.2f}, ST_Dir={int(latest['ST_Direction'])}"
    return new_signal, reason

# --- Main Agent Loop ---
async def agent_loop():
    global current_signal
    print(f"Starting Luno Trend Agent for {PAIR_TO_ANALYZE} on {TIMEFRAME_SECONDS}s timeframe.")
    print(f"Check interval: {CHECK_INTERVAL_SECONDS / 60:.1f} minutes.")

    if not LUNO_API_KEY_ID or not LUNO_API_KEY_SECRET:
        print("FATAL ERROR: LUNO_API_KEY_ID and LUNO_API_KEY_SECRET environment variables must be set.")
        return

    while True:
        print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}] Running check...")
        try:
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                df = await get_luno_candles(session, PAIR_TO_ANALYZE, TIMEFRAME_SECONDS, CANDLE_LIMIT)

                if df is not None:
                    new_signal, reason = analyze_trend(df.copy()) # Analyze a copy

                    print(f"Pair: {PAIR_TO_ANALYZE}")
                    print(f"Analysis Result: {reason}")
                    print(f"Previous Signal: {current_signal}, New Signal: {new_signal}")

                    if new_signal != current_signal:
                        print(f"*** Signal Changed: {current_signal} -> {new_signal} ***")
                        current_signal = new_signal
                        # Optional: Send notification only on BUY signal or major changes
                        if new_signal == "BUY":
                             send_agent_notification(PAIR_TO_ANALYZE, new_signal, df.iloc[-1]['close'])
                    else:
                        print("Signal unchanged.")
                else:
                    print("Could not get data for analysis.")

        except aiohttp.ClientConnectorError as e:
            print(f"Network connection error: {e}. Retrying later...")
        except asyncio.TimeoutError:
             print(f"Network operation timed out. Retrying later...")
        except Exception as e:
            print(f"An unexpected error occurred in agent loop: {e}")

        print(f"Waiting for {CHECK_INTERVAL_SECONDS / 60:.1f} minutes until next check...")
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        import uvloop
        uvloop.install()
        print("Using uvloop")
    except ImportError:
        print("uvloop not found, using default asyncio loop")

    try:
        asyncio.run(agent_loop())
    except KeyboardInterrupt:
        print("\nAgent stopped by user.")
    except Exception as e:
         print(f"\nCritical error during agent execution: {e}")
