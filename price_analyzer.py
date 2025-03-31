import asyncio
import aiohttp
import pandas as pd
from io import StringIO
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime
import os
from collections import defaultdict

API_BASE_URL = "https://api.binance.com/api/v3"
FETCH_LIMIT = 30  # Fetch enough candles for context and patterns
PLOT_DIR = "plots" # Directory to save plots
TIMEFRAMES = ['1h', '4h', '1d', '1w', '1M'] # Timeframes to analyze

async def get_kline_data(session, symbol, interval='1d', limit=FETCH_LIMIT):
    """Fetches historical kline/candlestick data for a symbol and interval from Binance."""
    url = f"{API_BASE_URL}/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    try:
        # print(f"Fetching {symbol} {interval}...") # Optional: for debugging
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            # Binance klines format: [open_time, open, high, low, close, volume, close_time, ...]
            if data and len(data) >= 2: # Need at least 2 candles for basic patterns
                # Convert to DataFrame
                df = pd.DataFrame(data, columns=[
                    'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Close Time', 'Quote Asset Volume', 'Number of Trades',
                    'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
                ])
                # Convert timestamp to datetime and set as index (required for mplfinance)
                df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
                df.set_index('Open Time', inplace=True)
                # Convert OHLCV columns to numeric
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col])
                # print(f"Success: {symbol} {interval} - {len(df)} candles") # Optional: for debugging
                return df
            else:
                print(f"Warning: Insufficient data received for {symbol} {interval} (received {len(data)} candles)")
                return None
    except aiohttp.ClientResponseError as e:
         # Handle specific errors like 400 for invalid symbols/intervals gracefully
        if e.status == 400:
             print(f"Warning: Could not fetch data for {symbol} {interval} (status {e.status}). Invalid symbol/interval or insufficient history?")
        else:
             print(f"Error fetching data for {symbol} {interval}: {e}")
        return None
    except aiohttp.ClientError as e:
        print(f"Network error fetching data for {symbol} {interval}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred for {symbol} {interval}: {e}")
        return None

# --- Candlestick Pattern Recognition Functions ---
# (These functions remain unchanged)

def is_bullish_engulfing(df):
    """Checks for a Bullish Engulfing pattern using the last two candles."""
    if df is None or len(df) < 2: return False
    prev_candle = df.iloc[-2]
    curr_candle = df.iloc[-1]
    is_prev_bearish = prev_candle['Close'] < prev_candle['Open']
    is_curr_bullish = curr_candle['Close'] > curr_candle['Open']
    is_engulfing = (curr_candle['Open'] < prev_candle['Close']) and \
                   (curr_candle['Close'] > prev_candle['Open'])
    return is_prev_bearish and is_curr_bullish and is_engulfing

def is_bearish_engulfing(df):
    """Checks for a Bearish Engulfing pattern using the last two candles."""
    if df is None or len(df) < 2: return False
    prev_candle = df.iloc[-2]
    curr_candle = df.iloc[-1]
    is_prev_bullish = prev_candle['Close'] > prev_candle['Open']
    is_curr_bearish = curr_candle['Close'] < curr_candle['Open']
    is_engulfing = (curr_candle['Open'] > prev_candle['Close']) and \
                   (curr_candle['Close'] < prev_candle['Open'])
    return is_prev_bullish and is_curr_bearish and is_engulfing

def is_hammer(df):
    """Checks for a Hammer pattern on the last candle."""
    if df is None or len(df) < 1: return False
    candle = df.iloc[-1]
    body_size = abs(candle['Close'] - candle['Open'])
    lower_shadow = candle['Open'] - candle['Low'] if candle['Close'] >= candle['Open'] else candle['Close'] - candle['Low']
    upper_shadow = candle['High'] - candle['Close'] if candle['Close'] >= candle['Open'] else candle['High'] - candle['Open']
    is_long_lower = lower_shadow > (body_size * 2) if body_size > 0 else lower_shadow > (candle['High'] - candle['Low']) * 0.6
    is_small_upper = upper_shadow < (body_size * 0.5) if body_size > 0 else upper_shadow < (candle['High'] - candle['Low']) * 0.2
    return body_size > 0 and is_long_lower and is_small_upper

def is_shooting_star(df):
    """Checks for a Shooting Star pattern on the last candle."""
    if df is None or len(df) < 1: return False
    candle = df.iloc[-1]
    body_size = abs(candle['Close'] - candle['Open'])
    upper_shadow = candle['High'] - candle['Close'] if candle['Close'] >= candle['Open'] else candle['High'] - candle['Open']
    lower_shadow = candle['Open'] - candle['Low'] if candle['Close'] >= candle['Open'] else candle['Close'] - candle['Low']
    is_long_upper = upper_shadow > (body_size * 2) if body_size > 0 else upper_shadow > (candle['High'] - candle['Low']) * 0.6
    is_small_lower = lower_shadow < (body_size * 0.5) if body_size > 0 else lower_shadow < (candle['High'] - candle['Low']) * 0.2
    return body_size > 0 and is_long_upper and is_small_lower

# --- Plotting Function ---

def plot_candlestick_with_patterns(df, symbol, timeframe, patterns):
    """Plots candlestick chart for a specific timeframe and highlights detected patterns."""
    if df is None or df.empty:
        print(f"Cannot plot {symbol} {timeframe}: No data.")
        return

    pattern_texts = []
    pattern_lines = [] # List to store line sequences for alines
    line_colors = []   # List to store colors for each line sequence

    # Prepare lines for mplfinance alines based on detected patterns for THIS timeframe
    for pattern_name, is_detected in patterns.items():
        if is_detected:
            pattern_texts.append(pattern_name)
            idx_curr = df.index[-1]
            candle_curr = df.iloc[-1]

            if "Engulfing" in pattern_name:
                if len(df) < 2: continue # Need previous candle
                idx_prev = df.index[-2]
                candle_prev = df.iloc[-2]
                color = 'blue' if 'Bullish' in pattern_name else 'purple'
                pattern_lines.append([(idx_prev, candle_prev['Open']), (idx_curr, candle_curr['Close'])])
                line_colors.append(color)
                pattern_lines.append([(idx_prev, candle_prev['Close']), (idx_curr, candle_curr['Open'])])
                line_colors.append(color)
            elif pattern_name == "Hammer":
                pattern_lines.append([(idx_curr, max(candle_curr['Open'], candle_curr['Close'])), (idx_curr, candle_curr['Low'])])
                line_colors.append('green')
            elif pattern_name == "Shooting Star":
                 pattern_lines.append([(idx_curr, min(candle_curr['Open'], candle_curr['Close'])), (idx_curr, candle_curr['High'])])
                 line_colors.append('red')

    # Create plot directory if it doesn't exist
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    # Include timeframe in filename and title
    plot_filename = os.path.join(PLOT_DIR, f"{symbol}_{timeframe}_plot.png")
    plot_title = f"{symbol} ({timeframe}) Candlestick Chart ({df.index[-1].strftime('%Y-%m-%d %H:%M' if timeframe in ['1h','4h'] else '%Y-%m-%d')})\nDetected: {', '.join(pattern_texts) or 'None'}"

    # Prepare arguments for mpf.plot
    plot_kwargs = dict(
        type='candle',
        style='yahoo',
        title=plot_title,
        ylabel='Price',
        volume=True,
        ylabel_lower='Volume',
        figsize=(12, 7),
        savefig=plot_filename
    )

    # Add alines if patterns were detected for this timeframe
    if pattern_lines:
        plot_kwargs['alines'] = dict(alines=pattern_lines, colors=line_colors, linewidths=1.5, alpha=0.7)

    try:
        mpf.plot(df, **plot_kwargs) # Use dictionary unpacking
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error plotting {symbol} {timeframe}: {e}")
    finally:
        plt.close('all') # Close figures to free memory

async def main():
    symbols = ["SOLUSDT", "BTCUSDT", "ETHUSDT", "BNBBTC", "ADAUSDT"] # Example symbols

    # Create tasks for all symbol/timeframe combinations
    tasks = []
    task_info = [] # To map results back to symbol/timeframe
    async with aiohttp.ClientSession() as session:
        for symbol in symbols:
            for tf in TIMEFRAMES:
                tasks.append(get_kline_data(session, symbol, interval=tf))
                task_info.append({'symbol': symbol, 'timeframe': tf})

        print(f"Fetching data for {len(symbols)} symbols across {len(TIMEFRAMES)} timeframes...")
        results = await asyncio.gather(*tasks, return_exceptions=True) # Capture exceptions too
        print("Data fetching complete.")

        # Process results - group by symbol
        data_by_symbol = defaultdict(dict)
        for i, result in enumerate(results):
            info = task_info[i]
            symbol = info['symbol']
            tf = info['timeframe']
            if isinstance(result, pd.DataFrame):
                data_by_symbol[symbol][tf] = result
            elif isinstance(result, Exception):
                 print(f"Task failed for {symbol} {tf}: {result}") # Log exceptions from gather
            # else: result is None due to handled errors in get_kline_data

        print("\n--- Multi-Timeframe Candlestick Pattern Analysis ---")

        for symbol, timeframe_data in data_by_symbol.items():
            print(f"\n--- {symbol} ---")
            symbol_has_pattern = False
            patterns_found_for_plotting = defaultdict(dict)

            for tf, df in sorted(timeframe_data.items(), key=lambda item: TIMEFRAMES.index(item[0])): # Sort by timeframe order
                if df is not None and len(df) >= 2:
                    patterns = {
                        "Bullish Engulfing": is_bullish_engulfing(df),
                        "Bearish Engulfing": is_bearish_engulfing(df),
                        "Hammer": is_hammer(df),
                        "Shooting Star": is_shooting_star(df)
                    }

                    detected_tf = [name for name, detected in patterns.items() if detected]
                    if detected_tf:
                        print(f"  [{tf}]: Detected -> {', '.join(detected_tf)}")
                        symbol_has_pattern = True
                        patterns_found_for_plotting[tf] = {name: det for name, det in patterns.items() if det} # Store only detected patterns for plotting
                    # else:
                    #     print(f"  [{tf}]: No patterns detected.") # Optional: print if nothing found

                elif df is not None:
                     print(f"  [{tf}]: Could not analyze (only {len(df)} candle(s))")
                # else: Data fetch failed or returned None, already printed warning/error

            if not symbol_has_pattern:
                print("  No patterns detected on any analyzed timeframe.")

            # Plotting - only plot timeframes where patterns were detected
            for tf, detected_patterns in patterns_found_for_plotting.items():
                 if tf in timeframe_data: # Ensure data exists for this timeframe
                     plot_candlestick_with_patterns(timeframe_data[tf], symbol, tf, detected_patterns)


if __name__ == "__main__":
    # Optional: uvloop integration (if installed)
    try:
        import uvloop
        uvloop.install()
        print("Using uvloop")
    except ImportError:
        print("uvloop not found, using default asyncio loop")
        pass

    # Ensure plot directory exists before starting async loop
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    asyncio.run(main())
