import asyncio
import aiohttp
import pandas as pd
import numpy as np
import os
import time
import subprocess # For notifications
from collections import defaultdict
from datetime import datetime
import traceback # Import traceback for detailed error logging

API_BASE_URL = "https://api.binance.com/api/v3"
EXCHANGE_RATE_API_URL = "https://api.exchangerate-api.com/v4/latest/USDT" # Public API for exchange rates
FETCH_LIMIT_DEFAULT = 250
FETCH_LIMIT_WEEKLY = 100
SYMBOLS_DCA = ["SOLUSDT", "BTCUSDT", "ETHUSDT", "ADAUSDT"] # Updated list
TIMEFRAMES_DCA = ['1h', '4h', '1d', '1w', '1M']
TF_WEIGHTS = {'1h': 0.5, '4h': 0.75, '1d': 1.0, '1w': 1.25, '1M': 1.5}

# Indicator Settings
RSI_LENGTH = 14
EMA_LENGTH = 200
EMA_LENGTH_WEEKLY = 50
BBANDS_LENGTH = 20
BBANDS_STDDEV = 2.0
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_LENGTH = 10
SUPERTREND_FACTOR = 3.0
OBV_SMOOTHING = 5
PIVOT_LOOKBACK = 2 # Bars before/after for pivot detection
SUPPORT_PROXIMITY_PERCENT = 0.90 # Show supports within 10% below current price

# Scoring Thresholds
RSI_OVERSOLD_STRONG = 30
BUY_RANGE_RAW_SCORE_THRESHOLD = 4
BUY_RANGE_TIMEFRAMES = ['1d', '1w']
OVERALL_SCORE_THRESHOLD = 5.0

# Continuous Run Settings
CHECK_INTERVAL_SECONDS = 4 * 60 * 60 # 4 hours

# --- Notification Function ---
def send_notification(symbol, score, price_myr):
    title = f"DCA Alert: {symbol}"
    message = f"Potential DCA opportunity detected for {symbol}.\nOverall Weighted Score: {score:.2f}\nCurrent Price: ~{price_myr:.2f} MYR"
    try:
        subprocess.run(['notify-send', title, message], check=False, timeout=10)
        print(f"--- Notification attempted for {symbol} ---")
    except FileNotFoundError: print("Warning: 'notify-send' command not found.")
    except subprocess.TimeoutExpired: print("Warning: 'notify-send' command timed out.")
    except Exception as e: print(f"Warning: Failed to send notification for {symbol}: {e}")

# --- Data Fetching ---
async def get_kline_data_dca(session, symbol, interval='1d', limit=FETCH_LIMIT_DEFAULT):
    url = f"{API_BASE_URL}/klines"; params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status(); data = await response.json()
            min_len_needed = EMA_LENGTH if interval not in ['1w', '1M'] else EMA_LENGTH_WEEKLY
            min_len_needed = max(min_len_needed, MACD_SLOW, BBANDS_LENGTH, RSI_LENGTH, ATR_LENGTH, PIVOT_LOOKBACK * 2 + 1)
            if data and len(data) >= min_len_needed:
                df = pd.DataFrame(data, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume','Close Time', 'Quote Asset Volume', 'Number of Trades','Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])
                df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms'); df.set_index('Open Time', inplace=True)
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']: df[col] = pd.to_numeric(df[col])
                df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                return df
            else:
                min_basic_len = max(BBANDS_LENGTH, RSI_LENGTH, MACD_SLOW, PIVOT_LOOKBACK * 2 + 1)
                if data and len(data) >= min_basic_len:
                    df = pd.DataFrame(data, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume','Close Time', 'Quote Asset Volume', 'Number of Trades','Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])
                    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms'); df.set_index('Open Time', inplace=True)
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']: df[col] = pd.to_numeric(df[col])
                    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                    return df
                else: return None
    except aiohttp.ClientResponseError as e: return None
    except Exception as e: print(f"Fetch error {symbol} {interval}: {e}"); return None

async def get_usdt_myr_rate(session):
    """Fetches the current USDT to MYR exchange rate."""
    try:
        async with session.get(EXCHANGE_RATE_API_URL) as response:
            response.raise_for_status()
            data = await response.json()
            rate = data.get('rates', {}).get('MYR')
            if rate:
                print(f"Fetched USDT/MYR rate: {rate:.4f}")
                return float(rate)
            else:
                print("Error: MYR rate not found in API response.")
                return None
    except aiohttp.ClientError as e:
        print(f"Error fetching exchange rate: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error fetching exchange rate: {e}")
        return None

# --- Manual Indicator Calculations (Unchanged) ---
def calculate_rsi(series, length=14):
    delta = series.diff(); gain = (delta.where(delta > 0, 0)).rolling(window=length).mean(); loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss.replace(0, np.nan); rsi = 100 - (100 / (1 + rs)); rsi = rsi.fillna(100)
    return rsi

def calculate_ema(series, length=200):
    if len(series) < length: return pd.Series(np.nan, index=series.index)
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def calculate_bbands(series, length=20, std=2.0):
    if len(series) < length: return pd.Series(np.nan, index=series.index), pd.Series(np.nan, index=series.index), pd.Series(np.nan, index=series.index)
    sma = series.rolling(window=length, min_periods=length).mean(); std_dev = series.rolling(window=length, min_periods=length).std()
    upper_band = sma + (std_dev * std); lower_band = sma - (std_dev * std)
    return upper_band, sma, lower_band

def calculate_macd(series, fast=12, slow=26, signal=9):
    if len(series) < slow: return pd.Series(np.nan, index=series.index), pd.Series(np.nan, index=series.index), pd.Series(np.nan, index=series.index)
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean(); ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow; signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean(); histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(high, low, close, length=14):
    if len(close) < length + 1: return pd.Series(np.nan, index=close.index)
    high_low = high - low; high_close = np.abs(high - close.shift()); low_close = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    return atr

def calculate_supertrend(high, low, close, atr_length=10, factor=3.0):
    if len(close) < atr_length + 1: return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)
    atr = calculate_atr(high, low, close, length=atr_length); hl2 = (high + low) / 2
    upper_band = hl2 + (factor * atr); lower_band = hl2 - (factor * atr)
    supertrend = pd.Series(np.nan, index=close.index); trend = pd.Series(1, index=close.index)
    for i in range(1, len(close)):
        prev_upper = upper_band.iloc[i-1]; prev_lower = lower_band.iloc[i-1]; prev_close = close.iloc[i-1]
        if pd.isna(prev_upper) or pd.isna(prev_lower) or pd.isna(prev_close): trend.iloc[i] = trend.iloc[i-1]
        elif prev_close > prev_upper: trend.iloc[i] = 1
        elif prev_close < prev_lower: trend.iloc[i] = -1
        else: trend.iloc[i] = trend.iloc[i-1]
        current_lower = lower_band.iloc[i]; current_upper = upper_band.iloc[i]
        prev_lower_safe = lower_band.iloc[i-1] if pd.notna(lower_band.iloc[i-1]) else -np.inf
        prev_upper_safe = upper_band.iloc[i-1] if pd.notna(upper_band.iloc[i-1]) else np.inf
        if trend.iloc[i] == 1:
            lower_band.iloc[i] = max(current_lower, prev_lower_safe) if pd.notna(current_lower) else prev_lower_safe
            supertrend.iloc[i] = lower_band.iloc[i]
        else:
            upper_band.iloc[i] = min(current_upper, prev_upper_safe) if pd.notna(current_upper) else prev_upper_safe
            supertrend.iloc[i] = upper_band.iloc[i]
    direction = pd.Series(np.where(close > supertrend, 1, -1), index=close.index); direction.bfill(inplace=True)
    return supertrend, direction

def calculate_obv(close, volume):
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def find_pivot_lows(low_series, n=2):
    if len(low_series) < 2 * n + 1: return pd.Series(False, index=low_series.index)
    is_lower_than_prev = low_series < low_series.shift(1).rolling(window=n, min_periods=n).min()
    is_lower_than_next = low_series < low_series.shift(-n).rolling(window=n, min_periods=n).min()
    is_pivot = is_lower_than_prev & is_lower_than_next
    return is_pivot

def find_pivot_highs(high_series, n=2):
    if len(high_series) < 2 * n + 1: return pd.Series(False, index=high_series.index)
    is_higher_than_prev = high_series > high_series.shift(1).rolling(window=n, min_periods=n).max()
    is_higher_than_next = high_series > high_series.shift(-n).rolling(window=n, min_periods=n).max()
    is_pivot = is_higher_than_prev & is_higher_than_next
    return is_pivot

# --- Scoring Function (Modified for MYR display) ---
def calculate_tf_score(df, timeframe, usdt_myr_rate):
    try: # Add outer try block for the entire function
        min_len_needed = max(BBANDS_LENGTH, RSI_LENGTH, MACD_SLOW)
        if df is None or df.empty or len(df) < min_len_needed: return None, {"Error": f"Insufficient data ({len(df) if df is not None else 0} < {min_len_needed})"}

        # Calculate indicators on original USDT data (INDENTED)
        df['RSI'] = calculate_rsi(df['close'], length=RSI_LENGTH)
        df['EMA'] = calculate_ema(df['close'], length=EMA_LENGTH)
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bbands(df['close'], length=BBANDS_LENGTH, std=BBANDS_STDDEV)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        try: df['Supertrend'], df['ST_Direction'] = calculate_supertrend(df['high'], df['low'], df['close'], atr_length=ATR_LENGTH, factor=SUPERTREND_FACTOR)
        except Exception as e: df['Supertrend'], df['ST_Direction'] = np.nan, np.nan
        df['OBV'] = calculate_obv(df['close'], df['volume'])

        latest = df.iloc[-1]; previous = df.iloc[-2]
        raw_score = 0; reasons = []; indicator_values = {}

        required_score_cols = ['RSI', 'BB_Lower', 'MACD', 'MACD_Signal', 'MACD_Hist']
        if latest[required_score_cols].isnull().any() or previous['MACD_Hist'] is None: return 0, {"Error": f"NaN in scoring indicators"}

        latest_close_usdt = latest['close']
        indicator_values['Close (USDT)'] = f"{latest_close_usdt:.4f}"
        # Add MYR close price
        if usdt_myr_rate:
            latest_close_myr = latest_close_usdt * usdt_myr_rate
            indicator_values['Close (MYR)'] = f"{latest_close_myr:.2f}"
        else:
            indicator_values['Close (MYR)'] = "N/A (Rate Error)"

        rsi_value = latest['RSI']; indicator_values['RSI'] = f"{rsi_value:.2f}"
        if rsi_value < RSI_OVERSOLD_STRONG: raw_score += 3; reasons.append(f"RSI<{RSI_OVERSOLD_STRONG}")

        ema_value = latest['EMA']; ema_len = EMA_LENGTH_WEEKLY if timeframe == '1w' else EMA_LENGTH; ema_col_name = f"EMA{ema_len}"
        if pd.notna(ema_value):
            indicator_values[ema_col_name] = f"{ema_value:.4f}" # Keep EMA in USDT for comparison logic
            if latest_close_usdt <= ema_value: raw_score += 2; reasons.append(f"Price<=EMA{ema_len}")
        else: indicator_values[ema_col_name] = "NaN"

        bbl_value = latest['BB_Lower']; indicator_values['BB Lower'] = f"{bbl_value:.4f}" # Keep BB Lower in USDT
        if latest_close_usdt <= bbl_value: raw_score += 2; reasons.append(f"Price<=LowerBB")

        macd_line = latest['MACD']; signal_line = latest['MACD_Signal']; hist = latest['MACD_Hist']; prev_hist = previous['MACD_Hist']
        indicator_values['MACD Hist'] = f"{hist:.4f}"
        if macd_line > signal_line: raw_score += 1; reasons.append("MACD>Signal")
        if pd.notna(hist) and pd.notna(prev_hist) and hist > prev_hist: raw_score += 1; reasons.append("MACD Hist Incr")

        st_value = latest['Supertrend']; st_dir = latest['ST_Direction']; obv_value = latest['OBV']
        indicator_values['ST Dir'] = 'Up' if st_dir == 1 else 'Down' if st_dir == -1 else 'N/A'
        indicator_values['OBV'] = f"{obv_value:.0f}" if pd.notna(obv_value) else "NaN"
        obv_rising = False
        if len(df) > OBV_SMOOTHING and pd.notna(obv_value) and df['OBV'].iloc[-OBV_SMOOTHING:].notna().all():
             obv_prev_smoothed = df['OBV'].iloc[-OBV_SMOOTHING:].mean()
             if obv_value > obv_prev_smoothed: obv_rising = True
        indicator_values['OBV Rising?'] = 'Yes' if obv_rising else 'No'

        indicator_values['Raw Score'] = raw_score; indicator_values['Reasons'] = ", ".join(reasons) if reasons else "None"
        indicator_values['Buy Range (MYR)'] = "N/A" # Placeholder, calculated later if needed

        return raw_score, indicator_values # Successful return

    except Exception as e:
        # Catch any unexpected error during calculation
        print(f"!!! Exception inside calculate_tf_score for {timeframe}: {e}")
        # Return None score and an error dict to signify failure
        return None, {"Error": f"Exception in score calc: {e}"}

# --- Main Execution Logic (Modified for MYR) ---
async def run_analysis_cycle(session, previous_scores):
    # Fetch exchange rate first
    usdt_myr_rate = await get_usdt_myr_rate(session)
    if usdt_myr_rate is None:
        print("Could not fetch USDT/MYR rate. Skipping analysis cycle.")
        return previous_scores # Return previous scores to avoid notification spam if rate fails temporarily

    tasks = []; task_info = []
    for symbol in SYMBOLS_DCA:
        for tf in TIMEFRAMES_DCA:
            limit = FETCH_LIMIT_WEEKLY if tf == '1w' else FETCH_LIMIT_DEFAULT
            tasks.append(get_kline_data_dca(session, symbol, interval=tf, limit=limit))
            task_info.append({'symbol': symbol, 'timeframe': tf})
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching data...")
    all_results_raw = await asyncio.gather(*tasks, return_exceptions=True)
    print("Data fetching complete.")

    data_by_symbol_tf = defaultdict(dict); weekly_trend = {}; daily_supports_usdt = {}; daily_resistances_usdt = {}
    for i, result in enumerate(all_results_raw):
        info = task_info[i]; symbol = info['symbol']; tf = info['timeframe']
        if isinstance(result, pd.DataFrame): data_by_symbol_tf[symbol][tf] = result
        elif isinstance(result, Exception): data_by_symbol_tf[symbol][tf] = {"Error": f"Data fetch failed: {result}"}
        else: data_by_symbol_tf[symbol][tf] = {"Error": "Insufficient data or fetch error"}

        # Calculate weekly trend (based on USDT)
        if tf == '1w' and isinstance(result, pd.DataFrame) and len(result) >= EMA_LENGTH_WEEKLY:
            weekly_ema = calculate_ema(result['close'], length=EMA_LENGTH_WEEKLY)
            latest_close_usdt = result['close'].iloc[-1]; latest_weekly_ema = weekly_ema.iloc[-1]
            if pd.notna(latest_close_usdt) and pd.notna(latest_weekly_ema): weekly_trend[symbol] = latest_close_usdt > latest_weekly_ema
            else: weekly_trend[symbol] = None
        elif tf == '1w' and symbol not in weekly_trend: weekly_trend[symbol] = None

        # Calculate daily supports and resistances (in USDT)
        if tf == '1d' and isinstance(result, pd.DataFrame):
             daily_df = result
             low_pivots = find_pivot_lows(daily_df['low'], n=PIVOT_LOOKBACK)
             all_supports_usdt = daily_df['low'][low_pivots].iloc[:-1].tail(3).tolist() # Keep raw USDT values
             current_daily_close_usdt = daily_df['close'].iloc[-1]
             nearby_supports_usdt = [s for s in all_supports_usdt if s < current_daily_close_usdt and s >= current_daily_close_usdt * SUPPORT_PROXIMITY_PERCENT]
             daily_supports_usdt[symbol] = nearby_supports_usdt

             high_pivots = find_pivot_highs(daily_df['high'], n=PIVOT_LOOKBACK)
             resistance_levels_usdt = daily_df['high'][high_pivots].iloc[:-1].tail(3).tolist() # Keep raw USDT values
             daily_resistances_usdt[symbol] = resistance_levels_usdt
        elif tf == '1d':
             if symbol not in daily_supports_usdt: daily_supports_usdt[symbol] = []
             if symbol not in daily_resistances_usdt: daily_resistances_usdt[symbol] = []


    analysis_results = defaultdict(lambda: {'overall_score': 0.0, 'timeframes': {}})
    current_scores = {}
    latest_prices_myr = {} # Store latest MYR price for notifications

    for symbol, tf_data in data_by_symbol_tf.items():
        is_long_term_bullish = weekly_trend.get(symbol, False)
        for tf, df_or_error in tf_data.items():
            if isinstance(df_or_error, dict) and "Error" in df_or_error:
                 analysis_results[symbol]['timeframes'][tf] = df_or_error; continue
            df = df_or_error
            if tf in ['1h', '4h', '1d'] and not is_long_term_bullish:
                 analysis_results[symbol]['timeframes'][tf] = {"Info": "Skipped (Below 50w EMA)"}; continue

            current_ema_length = EMA_LENGTH_WEEKLY if tf == '1w' else EMA_LENGTH
            df_copy = df.copy() # Work on a copy to avoid modifying original data

            try:
                # Pass rate to scoring function
                raw_score, indicators = calculate_tf_score(df_copy, tf, usdt_myr_rate)

                if raw_score is not None:
                    timeframe_weight = TF_WEIGHTS.get(tf, 1.0)
                    weighted_score = raw_score * timeframe_weight
                    indicators['Weighted Score'] = weighted_score
                    analysis_results[symbol]['timeframes'][tf] = indicators # Store the results dict
                    analysis_results[symbol]['overall_score'] += weighted_score

                    # Store latest MYR price if available
                    if tf == '1d' and 'Close (MYR)' in indicators and indicators['Close (MYR)'] != "N/A (Rate Error)":
                         try: latest_prices_myr[symbol] = float(indicators['Close (MYR)'])
                         except ValueError: pass

                    # Calculate Buy Range in MYR
                    if raw_score >= BUY_RANGE_RAW_SCORE_THRESHOLD and tf in BUY_RANGE_TIMEFRAMES and usdt_myr_rate:
                        latest_close_usdt_str = indicators.get('Close (USDT)')
                        bbl_usdt = df_copy.iloc[-1]['BB_Lower']
                        ema_usdt = df_copy.iloc[-1]['EMA'] # Use the EMA calculated within calculate_tf_score
                        buy_range_myr_str = "N/A"

                        if latest_close_usdt_str:
                            try:
                                latest_close_usdt_f = float(latest_close_usdt_str)
                                potential_supports_usdt_below = []
                                if pd.notna(bbl_usdt) and bbl_usdt < latest_close_usdt_f: potential_supports_usdt_below.append(bbl_usdt)
                                if pd.notna(ema_usdt) and ema_usdt < latest_close_usdt_f: potential_supports_usdt_below.append(ema_usdt)

                                if potential_supports_usdt_below:
                                    buy_low_usdt = min(potential_supports_usdt_below)
                                    buy_high_usdt = min(buy_low_usdt * 1.01, latest_close_usdt_f * 0.998)

                                    buy_low_myr = buy_low_usdt * usdt_myr_rate
                                    buy_high_myr = buy_high_usdt * usdt_myr_rate

                                    if buy_high_myr > buy_low_myr: buy_range_myr_str = f"{buy_low_myr:.2f} - {buy_high_myr:.2f}"
                                    else: buy_range_myr_str = f"Near {buy_low_myr:.2f}?"
                                else:
                                    latest_close_myr = latest_close_usdt_f * usdt_myr_rate
                                    buy_range_myr_str = f"Below {latest_close_myr:.2f}?"
                            except ValueError: buy_range_myr_str = "Error parsing price"
                        analysis_results[symbol]['timeframes'][tf]['Buy Range (MYR)'] = buy_range_myr_str
                else: # raw_score is None, indicators holds the error dict
                    analysis_results[symbol]['timeframes'][tf] = indicators # Store the error dict

            except Exception as score_error:
                print(f"Error during score calculation for {symbol} {tf}: {score_error}")
                # Assign an error dictionary explicitly if calculation fails unexpectedly
                analysis_results[symbol]['timeframes'][tf] = {"Error": f"Score calculation exception: {score_error}"}
                # Ensure overall score doesn't get NaN added if calculation fails badly
                analysis_results[symbol]['overall_score'] += 0.0 # No score contribution

        # --- This block should be outside the inner TF loop ---
        current_overall_score = analysis_results[symbol]['overall_score']
        current_scores[symbol] = current_overall_score
        last_score = previous_scores.get(symbol, 0)

        # Send notification if threshold crossed
        if current_overall_score >= OVERALL_SCORE_THRESHOLD and last_score < OVERALL_SCORE_THRESHOLD:
            price_for_notif = latest_prices_myr.get(symbol, 0.0) # Get latest MYR price for notification
            send_notification(symbol, current_overall_score, price_for_notif)

    # Print results table
    print("\n--- Multi-Timeframe DCA Analysis (MYR Prices) ---")
    # Adjusted headers and widths for MYR
    header = f"{'Symbol':<12} | {'Overall Score':<13} | {'Supports (MYR)':<30} | {'Resistances (MYR)':<30} | {'TF':<3} | {'Raw':<3} | {'Wght':<4} | {'Close (MYR)':<15} | {'RSI':<8} | {'MACD Hist':<10} | {'Buy Range (MYR)':<25} | Reasons / Status"
    print(header); print("-" * (len(header) + 10))
    sorted_symbols = sorted(analysis_results.items(), key=lambda item: item[1]['overall_score'], reverse=True)

    for symbol, data in sorted_symbols:
        overall_score = data['overall_score']; highlight = "*" if overall_score >= OVERALL_SCORE_THRESHOLD else " "
        trend_status = "Bullish" if weekly_trend.get(symbol) else "Bearish" if weekly_trend.get(symbol) is False else "Unknown"

        # Convert supports/resistances to MYR for display
        supports_myr_str = "N/A (Rate Error)"
        resistances_myr_str = "N/A (Rate Error)"
        if usdt_myr_rate:
            supports_myr = [f"{s * usdt_myr_rate:.2f}" for s in daily_supports_usdt.get(symbol, [])]
            supports_myr_str = ', '.join(supports_myr) or "N/A"
            resistances_myr = [f"{r * usdt_myr_rate:.2f}" for r in daily_resistances_usdt.get(symbol, [])]
            resistances_myr_str = ', '.join(resistances_myr) or "N/A"

        print(f"{symbol:<12} |{highlight}{overall_score:<12.2f} | {supports_myr_str:<30} | {resistances_myr_str:<30} | {'---':<3} | {'---':<3} | {'----':<4} | {'-'*15} | {'-'*8} | {'-'*10} | {'-'*25} | --- Weekly Trend: {trend_status} ---")
        sorted_tfs = sorted(data['timeframes'].keys(), key=lambda t: TIMEFRAMES_DCA.index(t))
        for tf in sorted_tfs:
             indicators = data['timeframes'][tf]
             if "Error" in indicators: print(f"{'':<12} | {' ':<13} | {' ':<30} | {' ':<30} | {tf:<3} | {'N/A':<3} | {'N/A':<4} | {'N/A':<15} | {'N/A':<8} | {'N/A':<10} | {'N/A':<25} | {indicators['Error']}")
             elif "Info" in indicators: print(f"{'':<12} | {' ':<13} | {' ':<30} | {' ':<30} | {tf:<3} | {'---':<3} | {'----':<4} | {'---':<15} | {'---':<8} | {'---':<10} | {'---':<25} | {indicators['Info']}")
             else:
                  raw_s = indicators.get('Raw Score', 'N/A'); weighted_s = indicators.get('Weighted Score', 'N/A')
                  weighted_s_str = f"{weighted_s:.2f}" if isinstance(weighted_s, (float, int)) else "N/A"
                  print(f"{'':<12} | {' ':<13} | {' ':<30} | {' ':<30} | {tf:<3} | {str(raw_s):<3} | {weighted_s_str:<4} | {indicators.get('Close (MYR)', 'N/A'):<15} | {indicators.get('RSI', 'N/A'):<8} | {indicators.get('MACD Hist', 'N/A'):<10} | {indicators.get('Buy Range (MYR)', 'N/A'):<25} | {indicators.get('Reasons', 'N/A')}")
        print("-" * (len(header) + 10))
    print(f"* Overall Weighted Score >= {OVERALL_SCORE_THRESHOLD:.1f} highlighted.")
    return current_scores

async def main_loop():
    """Main loop for continuous analysis."""
    previous_scores = {}
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                 current_scores = await run_analysis_cycle(session, previous_scores)
                 previous_scores = current_scores # Update scores even if rate failed, to use last known good scores
        except aiohttp.ClientConnectorError as e: print(f"Network connection error: {e}. Retrying...")
        except NameError as ne: # Catch NameError specifically
            print(f"!!! NameError in main_loop: {ne}")
            traceback.print_exc() # Print traceback for NameError
        except Exception as e:
            print(f"An unexpected error occurred in main loop: {e}")
            traceback.print_exc() # Print traceback for other errors
        print(f"\nWaiting for {CHECK_INTERVAL_SECONDS // 3600} hours until next check...")
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    try: import uvloop; uvloop.install(); print("Using uvloop")
    except ImportError: print("uvloop not found, using default asyncio loop")
    # Run the continuous loop
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt: print("\nScript stopped by user.")
