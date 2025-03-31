import asyncio
import aiohttp
import pandas as pd
import numpy as np
import os
import time
import subprocess # For notifications
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

# --- Analysis Configuration ---
FETCH_LIMIT_DEFAULT = 500 # Increased default candle fetch limit
FETCH_LIMIT_WEEKLY = 300  # Increased weekly candle fetch limit (covers > 5 years)
# Luno pairs to analyze (e.g., XBTZAR, ETHZAR, LTCZAR) - Adjust as needed
SYMBOLS_DCA = ["SOLMYR", "ETHMYR", "ADAMYR"] # Example Luno pairs
# Luno uses duration in seconds. Map standard TFs to Luno durations.
# Supported Luno durations: 60, 300, 900, 1800, 3600, 10800, 14400, 28800, 86400, 259200, 604800
TIMEFRAME_MAP_LUNO = {
    '1h': 3600,
    '4h': 14400,
    '1d': 86400,
    '1w': 604800,
    # '1M' is not directly supported by Luno's candle durations
}
TIMEFRAMES_DCA = list(TIMEFRAME_MAP_LUNO.keys()) # Use the mapped timeframes
TF_WEIGHTS = {'1h': 0.5, '4h': 0.75, '1d': 1.0, '1w': 1.25} # Adjusted weights

# --- Indicator Settings (Same as dca_analyzer.py) ---
RSI_LENGTH = 14
EMA_LENGTH = 200
EMA_LENGTH_WEEKLY = 50 # Use 50 for weekly EMA
BBANDS_LENGTH = 20
BBANDS_STDDEV = 2.0
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_LENGTH = 10
SUPERTREND_FACTOR_1 = 3.0 # Default Supertrend
SUPERTREND_FACTOR_2 = 2.0 # Second Supertrend (tighter)
OBV_SMOOTHING = 5
VOLUME_SMA_LENGTH = 20 # For volume confirmation
PIVOT_LOOKBACK = 2
# SUPPORT_PROXIMITY_PERCENT = 0.90 # No longer needed for scoring logic
SUPPORT_PROXIMITY_ATR_MULTIPLIER = 0.75 # Score if price is within 0.75 * ATR above the closest support
ADX_LENGTH = 14 # Length for ADX calculation
BUY_RANGE_ATR_MULTIPLIER = 0.5 # Multiplier for ATR to determine buy range width above support


# --- Scoring Thresholds (Same as dca_analyzer.py) ---
RSI_OVERSOLD_STRONG = 30
BUY_RANGE_RAW_SCORE_THRESHOLD = 4
BUY_RANGE_TIMEFRAMES = ['1d', '1w']
OVERALL_SCORE_THRESHOLD = 5.0
ADX_THRESHOLD = 25 # Threshold for considering a trend strong


# --- Continuous Run Settings (Same as dca_analyzer.py) ---
CHECK_INTERVAL_SECONDS = 4 * 60 * 60 # 4 hours

# --- Notification Function (Same as dca_analyzer.py) ---
def send_notification(symbol, score):
    title = f"Luno DCA Alert: {symbol}"
    message = f"Potential DCA opportunity detected for {symbol}.\nOverall Weighted Score: {score:.2f}"
    try:
        # Check if notify-send exists
        if subprocess.run(['which', 'notify-send'], capture_output=True, text=True).returncode == 0:
            subprocess.run(['notify-send', title, message], check=False, timeout=10)
            print(f"--- Notification attempted for {symbol} ---")
        else:
            print("Warning: 'notify-send' command not found. Skipping notification.")
    except FileNotFoundError: print("Warning: 'notify-send' command not found.")
    except subprocess.TimeoutExpired: print("Warning: 'notify-send' command timed out.")
    except Exception as e: print(f"Warning: Failed to send notification for {symbol}: {e}")

# --- Data Fetching (Adapted for Luno) ---
async def get_kline_data_dca(session, pair, duration_seconds, limit):
    """Fetches candlestick data from Luno API."""
    if not LUNO_API_KEY_ID or not LUNO_API_KEY_SECRET:
        print("Error: LUNO_API_KEY_ID or LUNO_API_KEY_SECRET not set.")
        return None

    # Calculate 'since' timestamp (milliseconds)
    now_ms = int(time.time() * 1000)
    since_ms = now_ms - (limit * duration_seconds * 1000)

    url = f"{API_BASE_URL}/candles"
    params = {'pair': pair, 'since': since_ms, 'duration': duration_seconds}
    auth = BasicAuth(LUNO_API_KEY_ID, LUNO_API_KEY_SECRET)

    try:
        async with session.get(url, params=params, auth=auth) as response:
            # Luno returns 401 Unauthorized if keys are wrong/missing
            if response.status == 401:
                print(f"Error: Luno API authentication failed (401). Check API keys.")
                return None
            response.raise_for_status() # Raise exceptions for 4xx/5xx errors
            data = await response.json()

            candles = data.get('candles', [])

            # Determine minimum length needed based on indicators
            min_len_needed = EMA_LENGTH if duration_seconds < 604800 else EMA_LENGTH_WEEKLY
            min_len_needed = max(min_len_needed, MACD_SLOW, BBANDS_LENGTH, RSI_LENGTH, ATR_LENGTH, PIVOT_LOOKBACK * 2 + 1)

            if candles and len(candles) >= min_len_needed:
                # Luno candle format: {"timestamp":ms,"open":"str","high":"str","low":"str","close":"str","volume":"str"}
                df = pd.DataFrame(candles)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                # Convert string columns to numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                # Ensure standard column names (already matching)
                # df.rename(columns={'Open': 'open', ...}, inplace=True) # Not needed if names match
                return df
            else:
                # Fallback for basic indicators if EMA length not met
                min_basic_len = max(BBANDS_LENGTH, RSI_LENGTH, MACD_SLOW, PIVOT_LOOKBACK * 2 + 1)
                if candles and len(candles) >= min_basic_len:
                    df = pd.DataFrame(candles)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col])
                    return df # Return with fewer data points, EMA might be NaN
                else:
                    print(f"Warning: Insufficient data for {pair} {duration_seconds}s. Got {len(candles)}, needed {min_len_needed} (or {min_basic_len}).")
                    return None
    except aiohttp.ClientResponseError as e:
        print(f"HTTP Error fetching {pair} {duration_seconds}s: {e.status} {e.message}")
        return None
    except Exception as e:
        print(f"Fetch error {pair} {duration_seconds}s: {e}")
        return None

# --- Manual Indicator Calculations (Same as dca_analyzer.py) ---
def calculate_rsi(series, length=14):
    delta = series.diff(); gain = (delta.where(delta > 0, 0)).rolling(window=length).mean(); loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss.replace(0, np.nan); rsi = 100 - (100 / (1 + rs)); rsi = rsi.fillna(100) # Fill initial NaNs with 100 (or 50?)
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
    # Ensure alignment and handle potential NaNs from shift
    tr_df = pd.concat([high_low, high_close, low_close], axis=1)
    tr = tr_df.max(axis=1, skipna=False) # Don't skip NaNs initially
    # Use ewm for ATR calculation
    atr = tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    return atr

def calculate_supertrend(high, low, close, atr_length=10, factor=3.0):
    if len(close) < atr_length + 1: return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)
    atr = calculate_atr(high, low, close, length=atr_length)
    if atr.isnull().all(): # Handle case where ATR calculation fails
        return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)

    hl2 = (high + low) / 2
    upper_band = hl2 + (factor * atr)
    lower_band = hl2 - (factor * atr)

    # Initialize Supertrend and Trend direction
    supertrend = pd.Series(np.nan, index=close.index)
    trend = pd.Series(1, index=close.index) # Start with uptrend assumption

    for i in range(1, len(close)):
        prev_close = close.iloc[i-1]
        prev_upper = upper_band.iloc[i-1]
        prev_lower = lower_band.iloc[i-1]
        prev_trend = trend.iloc[i-1]

        # Handle potential NaNs in previous values
        if pd.isna(prev_close) or pd.isna(prev_upper) or pd.isna(prev_lower):
            trend.iloc[i] = prev_trend # Carry forward trend if previous data is missing
        elif prev_close > prev_upper:
            trend.iloc[i] = 1 # Uptrend
        elif prev_close < prev_lower:
            trend.iloc[i] = -1 # Downtrend
        else:
            trend.iloc[i] = prev_trend # No change

        current_lower = lower_band.iloc[i]
        current_upper = upper_band.iloc[i]

        # Adjust bands based on trend
        if trend.iloc[i] == 1: # Uptrend
            # Ensure lower band doesn't decrease
            lower_band.iloc[i] = max(current_lower if pd.notna(current_lower) else -np.inf,
                                     prev_lower if pd.notna(prev_lower) else -np.inf)
            supertrend.iloc[i] = lower_band.iloc[i]
        else: # Downtrend
            # Ensure upper band doesn't increase
            upper_band.iloc[i] = min(current_upper if pd.notna(current_upper) else np.inf,
                                     prev_upper if pd.notna(prev_upper) else np.inf)
            supertrend.iloc[i] = upper_band.iloc[i]

    # Determine final direction based on close vs final supertrend value
    direction = pd.Series(np.where(close > supertrend, 1, -1), index=close.index)
    # Forward fill the direction for initial NaNs if any
    direction.ffill(inplace=True)

    return supertrend, direction


def calculate_obv(close, volume):
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def find_pivot_lows(low_series, n=2):
    if len(low_series) < 2 * n + 1: return pd.Series(False, index=low_series.index)
    # Check if current low is lower than the minimum of the 'n' preceding lows
    is_lower_than_prev = low_series < low_series.shift(1).rolling(window=n, min_periods=1).min()
    # Check if current low is lower than the minimum of the 'n' succeeding lows
    is_lower_than_next = low_series < low_series.shift(-n).rolling(window=n, min_periods=1).min()
    is_pivot = is_lower_than_prev & is_lower_than_next
    return is_pivot

def find_pivot_highs(high_series, n=2):
    if len(high_series) < 2 * n + 1: return pd.Series(False, index=high_series.index)
    # Check if current high is higher than the maximum of the 'n' preceding highs
    is_higher_than_prev = high_series > high_series.shift(1).rolling(window=n, min_periods=1).max()
    # Check if current high is higher than the maximum of the 'n' succeeding highs
    is_higher_than_next = high_series > high_series.shift(-n).rolling(window=n, min_periods=1).max()
    is_pivot = is_higher_than_prev & is_higher_than_next
    return is_pivot

def check_bullish_divergence(price_series, indicator_series, lookback=20):
    """Checks for simple bullish divergence over a lookback period."""
    if len(price_series) < lookback + 1 or indicator_series.isnull().sum() > lookback // 2:
        return False, None, None # Not enough data or too many NaNs in indicator

    # Find the index of the most recent low price in the lookback period (excluding current bar)
    recent_price_low_idx = price_series.iloc[-lookback-1:-1].idxmin()
    # Find the index of the low price before that one, within the lookback
    prev_price_low_idx = price_series.iloc[-lookback-1:price_series.index.get_loc(recent_price_low_idx)].idxmin()

    if pd.isna(recent_price_low_idx) or pd.isna(prev_price_low_idx):
        return False, None, None # Couldn't find two distinct lows

    # Check if price made a lower low
    price_lower_low = price_series.loc[recent_price_low_idx] < price_series.loc[prev_price_low_idx]

    # Check if indicator made a higher low at the corresponding times
    indicator_at_recent_low = indicator_series.loc[recent_price_low_idx]
    indicator_at_prev_low = indicator_series.loc[prev_price_low_idx]

    if pd.isna(indicator_at_recent_low) or pd.isna(indicator_at_prev_low):
         return False, None, None # Indicator value missing at low points

    indicator_higher_low = indicator_at_recent_low > indicator_at_prev_low

    is_divergence = price_lower_low and indicator_higher_low

    # Return divergence status and the indicator values at the lows for potential logging/reasoning
    return is_divergence, indicator_at_prev_low, indicator_at_recent_low

# --- ADX Calculation ---
def calculate_di(high, low, close, length=14):
    """Calculates the +DI and -DI."""
    if len(close) < length + 1:
        return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)

    # Calculate True Range (TR) - reuse ATR calculation logic for consistency
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    tr_df = pd.concat([high_low, high_close, low_close], axis=1)
    tr = tr_df.max(axis=1, skipna=False)
    atr = tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean() # Smoothed TR (ATR)

    # Calculate Directional Movement (+DM, -DM)
    move_up = high.diff()
    move_down = -low.diff()
    plus_dm = pd.Series(np.where((move_up > move_down) & (move_up > 0), move_up, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((move_down > move_up) & (move_down > 0), move_down, 0.0), index=close.index)

    # Smooth +DM and -DM
    smooth_plus_dm = plus_dm.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

    # Calculate +DI and -DI
    plus_di = 100 * (smooth_plus_dm / atr.replace(0, np.nan))
    minus_di = 100 * (smooth_minus_dm / atr.replace(0, np.nan))

    # Fill initial NaNs if necessary (though smoothing should handle most)
    plus_di.fillna(0, inplace=True)
    minus_di.fillna(0, inplace=True)

    return plus_di, minus_di

def calculate_adx(high, low, close, length=14):
    """Calculates the ADX."""
    if len(close) < 2 * length: # Need more data for ADX smoothing
        return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)

    plus_di, minus_di = calculate_di(high, low, close, length=length)

    # Calculate the Directional Index (DX)
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    dx.fillna(0, inplace=True) # Fill NaNs in DX (e.g., if +DI and -DI are both 0)

    # Calculate the Average Directional Index (ADX) by smoothing DX
    adx = dx.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    adx.fillna(0, inplace=True) # Fill initial NaNs

    return adx, plus_di, minus_di

# --- Consecutive Signal Calculation ---
def calculate_consecutive_true(series):
    """Calculates the number of consecutive True values ending at each point."""
    if not isinstance(series, pd.Series) or series.dtype != bool:
        raise TypeError("Input must be a boolean Pandas Series.")

    # Create shifted series to identify changes
    shifted = series.shift(1, fill_value=False)
    # Identify points where the value changes or where a True sequence starts
    change_points = (series != shifted)
    # Create groups based on these change points
    groups = change_points.cumsum()
    # Calculate cumulative sum within each group where the series is True
    consecutive_counts = series.groupby(groups).cumsum()
    # Reset counts to 0 where the original series is False
    return consecutive_counts.where(series, 0)


# --- Heiken Ashi Calculation ---
def calculate_heiken_ashi(df):
    """Calculates Heiken Ashi candles."""
    ha_df = pd.DataFrame(index=df.index)
    ha_df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    # Calculate initial HA_Open
    ha_df['HA_Open'] = ((df['open'].shift(1) + df['close'].shift(1)) / 2).fillna((df['open'].iloc[0] + df['close'].iloc[0]) / 2)

    # Iteratively calculate HA_Open for subsequent bars using .loc
    for i in range(1, len(df)):
        # Use .loc with the index label to assign the value directly
        ha_df.loc[ha_df.index[i], 'HA_Open'] = (ha_df.loc[ha_df.index[i-1], 'HA_Open'] + ha_df.loc[ha_df.index[i-1], 'HA_Close']) / 2

    ha_df['HA_High'] = ha_df[['HA_Open', 'HA_Close']].join(df['high']).max(axis=1)
    ha_df['HA_Low'] = ha_df[['HA_Open', 'HA_Close']].join(df['low']).min(axis=1)

    return ha_df


# --- Forecasting Function ---
def estimate_next_check_time(symbol, weekly_df, weekly_status, daily_support_levels):
    """
    Estimates the next time to check based on weekly trend, combining multiple forecast methods.
    Uses time to EMA50, time for MACD Hist > 0, and time to reach closest daily support.
    Returns a date range (earliest - latest) or a single date if only one method yields a result.
    """
    if weekly_df is None or weekly_df.empty or len(weekly_df) < max(EMA_LENGTH_WEEKLY, MACD_SLOW + MACD_SIGNAL): # Need enough data for EMA50 & MACD
        return "N/A (No/Short Weekly Data)"

    now = datetime.now(timezone.utc)
    forecast_dates = [] # Store potential future check dates

    if weekly_status == "Confirmed Bullish":
        return "Monitor Buy Range" # Already good, check buy range

    try:
        # --- Prepare Data ---
        # Ensure necessary indicators are calculated (recalculate for safety)
        # Avoid modifying the input df directly if it's used elsewhere
        df_copy = weekly_df.copy()
        df_copy['EMA50'] = calculate_ema(df_copy['close'], length=EMA_LENGTH_WEEKLY)
        df_copy['MACD'], df_copy['MACD_Signal'], df_copy['MACD_Hist'] = calculate_macd(df_copy['close'])

        # Check if calculations produced enough non-NaN values at the end
        if len(df_copy) < 2 or df_copy[['close', 'EMA50', 'MACD_Hist']].iloc[-1].isnull().any():
             return "N/A (Weekly Calc Error)"

        latest_close = df_copy['close'].iloc[-1]
        latest_ema50 = df_copy['EMA50'].iloc[-1]
        latest_hist = df_copy['MACD_Hist'].iloc[-1]
        prev_hist = df_copy['MACD_Hist'].iloc[-2] if len(df_copy['MACD_Hist'].dropna()) >= 2 else np.nan
        weekly_range = (df_copy['high'] - df_copy['low']).iloc[-10:].mean() # Avg range last 10 weeks

        # --- Forecast Method 1: Time to cross Weekly EMA50 ---
        if latest_close < latest_ema50: # Price below EMA
            price_diff = latest_ema50 - latest_close
            if pd.notna(weekly_range) and weekly_range > 0:
                weeks_to_ema = price_diff / weekly_range
                buffer = 1.5 if weekly_status == "Confirmed Bearish" else 1.2 # Smaller buffer if MACD is improving
                estimated_weeks = max(1, weeks_to_ema * buffer) # Min 1 week forecast
                forecast_dates.append(now + timedelta(weeks=estimated_weeks))

        # --- Forecast Method 2: Time for Weekly MACD Histogram > 0 ---
        if pd.notna(latest_hist) and latest_hist < 0 and pd.notna(prev_hist) and latest_hist > prev_hist: # Negative and increasing
            recent_hist = df_copy['MACD_Hist'].iloc[-5:].dropna()
            if len(recent_hist) >= 2:
                 hist_change_rate = (recent_hist.iloc[-1] - recent_hist.iloc[0]) / (len(recent_hist) -1) if len(recent_hist) > 1 else np.nan
                 if pd.notna(hist_change_rate) and hist_change_rate > 1e-9: # Ensure rate is positive and non-trivial
                     weeks_to_zero = abs(latest_hist / hist_change_rate)
                     estimated_weeks = max(1, weeks_to_zero * 1.3) # Buffer, min 1 week
                     forecast_dates.append(now + timedelta(weeks=estimated_weeks))

        # --- Forecast Method 3: Time to reach closest Daily Support ---
        # Find the highest daily support level strictly below the current price
        supports_below_close = [s for s in daily_support_levels if s < latest_close]
        if supports_below_close:
            closest_support = max(supports_below_close)
            price_diff_support = latest_close - closest_support
            if pd.notna(weekly_range) and weekly_range > 0:
                 # Estimate weeks to drop to support (use weekly range as proxy for speed)
                 weeks_to_support = price_diff_support / weekly_range
                 # Add a small buffer, maybe less buffer needed for dropping?
                 buffer = 1.1
                 estimated_weeks = max(1, weeks_to_support * buffer) # Min 1 week forecast
                 forecast_dates.append(now + timedelta(weeks=estimated_weeks))


        # --- Combine Forecasts ---
        if forecast_dates:
            earliest_date = min(forecast_dates)
            latest_date = max(forecast_dates)
            if earliest_date == latest_date:
                return f"Est. Check: {earliest_date.strftime('%Y-%m-%d')}"
            else:
                # Ensure latest date is actually later than earliest
                if latest_date > earliest_date:
                     return f"Est. Check: {earliest_date.strftime('%Y-%m-%d')} - {latest_date.strftime('%Y-%m-%d')}"
                else: # Should not happen, but fallback
                     return f"Est. Check: {earliest_date.strftime('%Y-%m-%d')}"
        else:
            # Default check if no specific forecast possible
            if weekly_status == "Confirmed Bearish":
                check_date = now + timedelta(days=7) # Check in 1 week
                return f"Est. Check: {check_date.strftime('%Y-%m-%d')}"
            else: # Mixed status, no clear forecast derived
                check_date = now + timedelta(days=3) # Check sooner for mixed signals
                return f"Est. Check: {check_date.strftime('%Y-%m-%d')}"


    except Exception as e:
        # print(f"Error estimating check time for {symbol}: {e}") # Avoid excessive logging in production
        return "N/A (Forecast Error)"


# --- Scoring Function (Adapted for Luno Timeframes) ---
def calculate_tf_score(df, timeframe_key): # Use timeframe_key ('1h', '1d', etc.)
    duration_seconds = TIMEFRAME_MAP_LUNO[timeframe_key]
    min_len_needed = max(BBANDS_LENGTH, RSI_LENGTH, MACD_SLOW)
    if df is None or df.empty or len(df) < min_len_needed:
        return None, {"Error": f"Insufficient data ({len(df) if df is not None else 0} < {min_len_needed})"}

    # Calculate indicators
    df['RSI'] = calculate_rsi(df['close'], length=RSI_LENGTH)
    # Use correct EMA length based on timeframe
    current_ema_length = EMA_LENGTH_WEEKLY if timeframe_key == '1w' else EMA_LENGTH
    df['EMA'] = calculate_ema(df['close'], length=current_ema_length)
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bbands(df['close'], length=BBANDS_LENGTH, std=BBANDS_STDDEV)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    try:
        # Calculate both Supertrends
        df['Supertrend_1'], df['ST1_Direction'] = calculate_supertrend(df['high'], df['low'], df['close'], atr_length=ATR_LENGTH, factor=SUPERTREND_FACTOR_1)
        df['Supertrend_2'], df['ST2_Direction'] = calculate_supertrend(df['high'], df['low'], df['close'], atr_length=ATR_LENGTH, factor=SUPERTREND_FACTOR_2)
    except Exception as e:
        print(f"Error calculating Supertrends for {timeframe_key}: {e}")
        df['Supertrend_1'], df['ST1_Direction'] = np.nan, np.nan
        df['Supertrend_2'], df['ST2_Direction'] = np.nan, np.nan
    df['OBV'] = calculate_obv(df['close'], df['volume'])
    df['Volume_SMA'] = df['volume'].rolling(window=VOLUME_SMA_LENGTH).mean() # Calculate Volume SMA
    df['ATR'] = calculate_atr(df['high'], df['low'], df['close'], length=ATR_LENGTH) # Ensure ATR is calculated
    # Calculate ADX
    df['ADX'], df['PlusDI'], df['MinusDI'] = calculate_adx(df['high'], df['low'], df['close'], length=ADX_LENGTH)
    # Calculate Heiken Ashi candles
    ha_df = calculate_heiken_ashi(df)
    df = df.join(ha_df) # Join HA candles to the main dataframe

    # Get latest and previous data points
    if len(df) < 2:
        return 0, {"Error": "Need at least 2 data points for comparison"}
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    raw_score = 0
    reasons = []
    indicator_values = {}

    # Check for NaNs in essential indicators
    required_score_cols = ['RSI', 'BB_Lower', 'MACD', 'MACD_Signal', 'MACD_Hist']
    if latest[required_score_cols].isnull().any() or pd.isna(previous['MACD_Hist']):
        nan_cols = latest[required_score_cols][latest[required_score_cols].isnull()].index.tolist()
        if pd.isna(previous['MACD_Hist']): nan_cols.append("Prev MACD_Hist")
        return 0, {"Error": f"NaN in scoring indicators: {', '.join(nan_cols)}"}

    latest_close = latest['close']
    indicator_values['Close'] = f"{latest_close:.4f}" # Adjust precision as needed

    # --- Scoring Logic ---
    latest_close = latest['close']
    indicator_values['Close'] = f"{latest_close:.4f}" # Adjust precision as needed

    # EMA Score (Price relative to long-term EMA)
    ema_value = latest['EMA']
    ema_col_name = f"EMA{current_ema_length}"
    indicator_values[ema_col_name] = f"{ema_value:.4f}" if pd.notna(ema_value) else "NaN"
    if pd.notna(ema_value) and latest_close <= ema_value:
        raw_score += 2 # Score for being below long-term average
        reasons.append(f"Price<=EMA{current_ema_length}")

    # RSI + Bollinger Band Confirmation Score
    rsi_value = latest['RSI']
    bbl_value = latest['BB_Lower']
    volume_value = latest['volume']
    volume_sma_value = latest['Volume_SMA']
    indicator_values['RSI'] = f"{rsi_value:.2f}"
    indicator_values['BB Lower'] = f"{bbl_value:.4f}" if pd.notna(bbl_value) else "NaN"
    indicator_values['Volume'] = f"{volume_value:.2f}" if pd.notna(volume_value) else "NaN"
    indicator_values['Volume SMA'] = f"{volume_sma_value:.2f}" if pd.notna(volume_sma_value) else "NaN"

    rsi_bb_condition = rsi_value < RSI_OVERSOLD_STRONG and pd.notna(bbl_value) and latest_close <= bbl_value
    if rsi_bb_condition:
        raw_score += 3 # Base score for RSI + BB confirmation
        reasons.append(f"RSI<{RSI_OVERSOLD_STRONG}+Price<=BB")
        # Volume Confirmation Bonus
        if pd.notna(volume_value) and pd.notna(volume_sma_value) and volume_value > volume_sma_value:
            raw_score += 1 # Bonus point for volume confirmation
            reasons.append("Vol>SMA")

    # MACD Score (Momentum)
    macd_line = latest['MACD']
    signal_line = latest['MACD_Signal']
    hist = latest['MACD_Hist']
    prev_hist = previous['MACD_Hist']
    indicator_values['MACD Hist'] = f"{hist:.4f}" if pd.notna(hist) else "NaN"

    # MACD Line vs Signal Line (Bullish Crossover / Above Signal)
    if pd.notna(macd_line) and pd.notna(signal_line) and macd_line > signal_line:
        raw_score += 1
        reasons.append("MACD>Signal")

    # MACD Histogram Increasing (Improving Momentum)
    if pd.notna(hist) and pd.notna(prev_hist) and hist > prev_hist:
        raw_score += 1
        reasons.append("MACD Hist Incr")

    # Weekly MACD Confirmation (Applied only within weekly score calculation)
    if timeframe_key == '1w' and pd.notna(hist) and pd.notna(prev_hist) and hist > 0 and hist > prev_hist:
         raw_score += 1 # Extra point for weekly MACD turning positive and increasing
         reasons.append("Wkly MACD Hist>0 Incr")

    # Double Supertrend Confirmation Score
    st1_dir = latest['ST1_Direction']
    st2_dir = latest['ST2_Direction']
    indicator_values['ST1 Dir'] = 'Up' if st1_dir == 1 else 'Down' if st1_dir == -1 else 'N/A'
    indicator_values['ST2 Dir'] = 'Up' if st2_dir == 1 else 'Down' if st2_dir == -1 else 'N/A'
    if st1_dir == 1 and st2_dir == 1:
        raw_score += 2 # Score points if BOTH Supertrends are bullish
        reasons.append("ST1+ST2 Up")

    # OBV Score (Volume Momentum) - Kept separate
    obv_value = latest['OBV']
    indicator_values['OBV'] = f"{obv_value:.0f}" if pd.notna(obv_value) else "NaN"
    obv_rising = False
    if len(df) > OBV_SMOOTHING and pd.notna(obv_value) and df['OBV'].iloc[-OBV_SMOOTHING:].notna().all():
        # Compare current OBV to the average of the previous OBV_SMOOTHING periods
        obv_prev_smoothed = df['OBV'].iloc[-OBV_SMOOTHING-1:-1].mean()
        if pd.notna(obv_prev_smoothed) and obv_value > obv_prev_smoothed:
            obv_rising = True
    indicator_values['OBV Rising?'] = 'Yes' if obv_rising else 'No'
    if obv_rising:
        raw_score += 1 # Score point if OBV is rising
        reasons.append("OBV Rising")

    # Heiken Ashi Momentum Score
    ha_open = latest['HA_Open']
    ha_close = latest['HA_Close']
    ha_low = latest['HA_Low']
    ha_high = latest['HA_High']
    ha_bullish = pd.notna(ha_close) and pd.notna(ha_open) and ha_close > ha_open
    ha_strong_bullish = ha_bullish and ha_open == ha_low # Bullish candle with no lower wick
    indicator_values['HA Candle'] = 'Bull' if ha_bullish else 'Bear' if pd.notna(ha_close) and pd.notna(ha_open) and ha_close < ha_open else 'Doji'
    indicator_values['HA Strength'] = 'Strong' if ha_strong_bullish else 'Normal' if ha_bullish else 'N/A'

    if ha_bullish:
        raw_score += 1 # Basic point for bullish HA candle
        reasons.append("HA Bull")
        if ha_strong_bullish:
            raw_score += 1 # Extra point for strong bullish HA (no lower wick)
            reasons.append("HA Strong")

    # ADX Trend Strength Score (Bonus if trend is strong and bullish)
    adx_value = latest['ADX']
    plus_di_value = latest['PlusDI']
    minus_di_value = latest['MinusDI']
    indicator_values['ADX'] = f"{adx_value:.2f}" if pd.notna(adx_value) else "NaN"
    indicator_values['+DI'] = f"{plus_di_value:.2f}" if pd.notna(plus_di_value) else "NaN"
    indicator_values['-DI'] = f"{minus_di_value:.2f}" if pd.notna(minus_di_value) else "NaN"

    is_trending = pd.notna(adx_value) and adx_value > ADX_THRESHOLD
    is_bullish_trend = pd.notna(plus_di_value) and pd.notna(minus_di_value) and plus_di_value > minus_di_value
    if is_trending and is_bullish_trend:
        raw_score += 1 # Bonus point for strong bullish trend
        reasons.append(f"ADX>{ADX_THRESHOLD} Bull")


    # Proximity to Daily Support Score (Only applies when calculating '1d' score)
    if timeframe_key == '1d':
        # Access the globally stored daily supports for the current symbol
        # Note: This requires daily_supports to be populated before scoring begins.
        # We might need to adjust the main loop slightly if this causes issues.
        symbol = df.name # Assuming the DataFrame is named after the symbol during processing
        support_levels = daily_supports.get(symbol, [])
        daily_atr = latest['ATR'] # Get the latest ATR value for the daily timeframe
        if support_levels and pd.notna(daily_atr) and daily_atr > 0:
            closest_support = max(support_levels) # Highest support level below current price
            # Define proximity threshold based on ATR
            proximity_upper_bound = closest_support + (daily_atr * SUPPORT_PROXIMITY_ATR_MULTIPLIER)
            # Check if close is between support and the ATR-based upper bound
            if latest_close >= closest_support and latest_close <= proximity_upper_bound:
                 raw_score += 2 # Add points for being close to support (ATR adjusted)
                 reasons.append(f"Near D Supp({closest_support:.4f}, ATR)")

    # Bullish Divergence Check (Daily timeframe primarily)
    if timeframe_key == '1d':
        rsi_divergence, _, _ = check_bullish_divergence(df['low'], df['RSI'])
        macd_hist_divergence, _, _ = check_bullish_divergence(df['low'], df['MACD_Hist'])

        if rsi_divergence:
            raw_score += 3 # Significant points for divergence
            reasons.append("RSI Bull Div")
            indicator_values['RSI Divergence'] = 'Yes'
        if macd_hist_divergence:
            raw_score += 3 # Significant points for divergence
            reasons.append("MACD Bull Div")
            indicator_values['MACD Divergence'] = 'Yes'


    indicator_values['Raw Score'] = raw_score
    indicator_values['Reasons'] = ", ".join(reasons) if reasons else "None"
    indicator_values['Buy Range'] = "N/A" # Will be calculated later if conditions met

    # --- Calculate Consecutive Signals ---
    try:
        # HA Bullish Consecutive
        df['HA_Bullish_Signal'] = df['HA_Close'] > df['HA_Open']
        df['HA_Bull_Consec'] = calculate_consecutive_true(df['HA_Bullish_Signal'])
        indicator_values['HA Bull Cons'] = int(df['HA_Bull_Consec'].iloc[-1]) if pd.notna(df['HA_Bull_Consec'].iloc[-1]) else 0

        # RSI Oversold Consecutive
        df['RSI_Oversold_Signal'] = df['RSI'] < RSI_OVERSOLD_STRONG
        df['RSI_OS_Consec'] = calculate_consecutive_true(df['RSI_Oversold_Signal'])
        indicator_values['RSI OS Cons'] = int(df['RSI_OS_Consec'].iloc[-1]) if pd.notna(df['RSI_OS_Consec'].iloc[-1]) else 0

        # MACD Hist Increasing Consecutive
        df['MACD_Hist_Incr_Signal'] = df['MACD_Hist'] > df['MACD_Hist'].shift(1)
        df['MACD_Hist_Incr_Consec'] = calculate_consecutive_true(df['MACD_Hist_Incr_Signal'])
        indicator_values['MACD Hist Incr Cons'] = int(df['MACD_Hist_Incr_Consec'].iloc[-1]) if pd.notna(df['MACD_Hist_Incr_Consec'].iloc[-1]) else 0

        # Price <= BB Lower Consecutive
        df['Price_BB_Lower_Signal'] = df['close'] <= df['BB_Lower']
        df['Price_BB_Lower_Consec'] = calculate_consecutive_true(df['Price_BB_Lower_Signal'])
        indicator_values['Price BB Low Cons'] = int(df['Price_BB_Lower_Consec'].iloc[-1]) if pd.notna(df['Price_BB_Lower_Consec'].iloc[-1]) else 0

    except Exception as e:
        print(f"Error calculating consecutive signals for {timeframe_key}: {e}")
        indicator_values['HA Bull Cons'] = 'Err'
        indicator_values['RSI OS Cons'] = 'Err'
        indicator_values['MACD Hist Incr Cons'] = 'Err'
        indicator_values['Price BB Low Cons'] = 'Err'


    return raw_score, indicator_values

# --- Main Execution Logic (Adapted for Luno) ---
# Global dictionaries to store support/resistance - needed for scoring function access
daily_supports = {}
daily_resistances = {}

async def run_analysis_cycle(session, previous_scores):
    global daily_supports, daily_resistances # Declare usage of global vars
    tasks = []
    task_info = [] # To map results back to symbol/timeframe

    # Create tasks for fetching data for each symbol and timeframe
    for symbol in SYMBOLS_DCA:
        for tf_key, duration_sec in TIMEFRAME_MAP_LUNO.items():
            limit = FETCH_LIMIT_WEEKLY if tf_key == '1w' else FETCH_LIMIT_DEFAULT
            tasks.append(get_kline_data_dca(session, symbol, duration_sec, limit))
            task_info.append({'symbol': symbol, 'timeframe': tf_key, 'duration': duration_sec}) # Store key and duration

    print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}] Fetching Luno data...")
    all_results_raw = await asyncio.gather(*tasks, return_exceptions=True)
    print("Luno data fetching complete.")

    data_by_symbol_tf = defaultdict(dict)
    weekly_trend_price_ema = {} # Store weekly trend (price vs EMA50)
    weekly_trend_macd = {} # Store weekly trend (MACD vs Signal)
    weekly_dataframes = {} # Store weekly dataframes for forecasting
    # Reset global dicts for the new cycle
    daily_supports = {}
    daily_resistances = {}


    # Process fetched data - First pass to calculate supports/resistances
    for i, result in enumerate(all_results_raw):
        info = task_info[i]
        symbol = info['symbol']
        tf_key = info['timeframe'] # Use '1h', '1d', etc.
        duration = info['duration']

        if isinstance(result, pd.DataFrame):
            data_by_symbol_tf[symbol][tf_key] = result

            # Calculate weekly trend (using 1w data) and store weekly df
            if tf_key == '1w':
                weekly_dataframes[symbol] = result # Store the weekly dataframe
                if len(result) >= max(EMA_LENGTH_WEEKLY, MACD_SLOW + MACD_SIGNAL): # Ensure enough data for EMA and MACD
                    weekly_ema = calculate_ema(result['close'], length=EMA_LENGTH_WEEKLY)
                    latest_close = result['close'].iloc[-1]
                    latest_weekly_ema = weekly_ema.iloc[-1]
                    # Price vs EMA Trend
                    if pd.notna(latest_close) and pd.notna(latest_weekly_ema):
                        weekly_trend_price_ema[symbol] = latest_close > latest_weekly_ema
                    else:
                        weekly_trend_price_ema[symbol] = None # Not enough data or NaN

                    # MACD Trend
                    macd_line, signal_line, _ = calculate_macd(result['close']) # Use default MACD settings
                    if pd.notna(macd_line.iloc[-1]) and pd.notna(signal_line.iloc[-1]):
                         weekly_trend_macd[symbol] = macd_line.iloc[-1] > signal_line.iloc[-1]
                    else:
                         weekly_trend_macd[symbol] = None
                else:
                     weekly_trend_price_ema[symbol] = None # Not enough data
                     weekly_trend_macd[symbol] = None
            # Ensure weekly trend is initialized if 1w fetch failed
            elif symbol not in weekly_trend_price_ema:
                 weekly_trend_price_ema[symbol] = None
                 weekly_trend_macd[symbol] = None


            # Calculate daily supports and resistances (using 1d data)
            if tf_key == '1d':
                daily_df = result
                symbol = info['symbol'] # Get symbol associated with this daily data
                # Supports (Pivot Lows) - Look back further if needed
                low_pivots = find_pivot_lows(daily_df['low'], n=PIVOT_LOOKBACK)
                current_daily_close = daily_df['close'].iloc[-1]

                # Get all confirmed pivot lows strictly below the current price
                confirmed_pivots_below_close = daily_df['low'][low_pivots].iloc[:-1] # Exclude the latest potential pivot
                supports_below_close = confirmed_pivots_below_close[confirmed_pivots_below_close < current_daily_close]

                # Get the 3 most recent supports below the current price
                recent_supports = supports_below_close.tail(3).round(4).tolist()
                daily_supports[symbol] = sorted(recent_supports) # Sort supports in ascending order

                # Resistances (Pivot Highs) - Keep existing logic
                high_pivots = find_pivot_highs(daily_df['high'], n=PIVOT_LOOKBACK)
                resistance_levels = daily_df['high'][high_pivots].iloc[:-1].tail(3).round(4).tolist()
                daily_resistances[symbol] = resistance_levels
            # Ensure daily levels are initialized if 1d fetch failed
            elif symbol not in daily_supports: daily_supports[symbol] = []
            elif symbol not in daily_resistances: daily_resistances[symbol] = []


        elif isinstance(result, Exception):
            data_by_symbol_tf[symbol][tf_key] = {"Error": f"Data fetch failed: {result}"}
            # Initialize levels if fetch failed for key timeframes
            if tf_key == '1w' and symbol not in weekly_trend: weekly_trend[symbol] = None
            if tf_key == '1d':
                if symbol not in daily_supports: daily_supports[symbol] = []
                if symbol not in daily_resistances: daily_resistances[symbol] = []
        else: # Handle None result (insufficient data or other fetch error)
            data_by_symbol_tf[symbol][tf_key] = {"Error": "Insufficient data or fetch error"}
            if tf_key == '1w' and symbol not in weekly_trend: weekly_trend[symbol] = None
            if tf_key == '1d':
                if symbol not in daily_supports: daily_supports[symbol] = []
                if symbol not in daily_resistances: daily_resistances[symbol] = []


    # --- Perform Analysis (Second Pass - Scoring) ---
    analysis_results = defaultdict(lambda: {'overall_score': 0.0, 'timeframes': {}})
    current_scores = {} # Store overall scores for notification check

    for symbol, tf_data in data_by_symbol_tf.items():
        # Get weekly trend status for filtering lower timeframes
        # Stricter Check: Require BOTH price > EMA50 AND MACD > Signal on weekly
        is_price_above_ema = weekly_trend_price_ema.get(symbol)
        is_macd_bullish = weekly_trend_macd.get(symbol)
        is_strong_weekly_bullish = is_price_above_ema is True and is_macd_bullish is True

        # Determine status string for printing
        if is_price_above_ema is None or is_macd_bullish is None:
            weekly_status_str = "Unknown"
        elif is_strong_weekly_bullish:
            weekly_status_str = "Confirmed Bullish"
        elif is_price_above_ema is True and is_macd_bullish is False:
             weekly_status_str = "Price>EMA, MACD Bear"
        elif is_price_above_ema is False and is_macd_bullish is True:
             weekly_status_str = "Price<EMA, MACD Bull"
        else: # Both False
             weekly_status_str = "Confirmed Bearish"


        for tf_key, df_or_error in tf_data.items():
            # Skip analysis if data fetch failed
            if isinstance(df_or_error, dict) and "Error" in df_or_error:
                analysis_results[symbol]['timeframes'][tf_key] = df_or_error
                continue

            df = df_or_error

            # --- Stricter Weekly Trend Filter ---
            # Skip 1h, 4h, 1d if weekly trend is NOT Confirmed Bullish
            if tf_key in ['1h', '4h', '1d'] and not is_strong_weekly_bullish:
                analysis_results[symbol]['timeframes'][tf_key] = {"Info": f"Skipped (Weekly: {weekly_status_str})"}
                continue
            # --- End Stricter Weekly Trend Filter ---

            # Calculate score for the timeframe
            # Pass a copy to avoid modifying the original df used by other TFs if needed later
            df_copy = df.copy()
            df_copy.name = symbol # Assign symbol name to DataFrame for access in scoring function
            raw_score, indicators = calculate_tf_score(df_copy, tf_key)

            if raw_score is not None:
                timeframe_weight = TF_WEIGHTS.get(tf_key, 1.0)
                weighted_score = raw_score * timeframe_weight
                indicators['Weighted Score'] = weighted_score
                analysis_results[symbol]['timeframes'][tf_key] = indicators
                analysis_results[symbol]['overall_score'] += weighted_score # Accumulate weighted score

                # Calculate Buy Range for specific timeframes if score threshold met
                if raw_score >= BUY_RANGE_RAW_SCORE_THRESHOLD and tf_key in BUY_RANGE_TIMEFRAMES:
                    latest_close_str = indicators.get('Close')
                    bbl_value_str = indicators.get('BB Lower') # Already calculated in score func
                    ema_value_str = indicators.get(f"EMA{EMA_LENGTH_WEEKLY if tf_key == '1w' else EMA_LENGTH}")
                    atr_value_str = indicators.get('ATR') # Get ATR value

                    buy_range_str = "N/A"
                    try:
                        latest_close_f = float(latest_close_str) if latest_close_str and latest_close_str != "N/A" else np.nan
                        bbl_value = float(bbl_value_str) if bbl_value_str and bbl_value_str != "NaN" else np.nan
                        ema_value = float(ema_value_str) if ema_value_str and ema_value_str != "NaN" else np.nan

                        potential_supports_below = []
                        if pd.notna(bbl_value) and bbl_value < latest_close_f:
                            potential_supports_below.append(bbl_value)
                        if pd.notna(ema_value) and ema_value < latest_close_f:
                            potential_supports_below.append(ema_value)

                        # Also consider nearby daily pivot supports
                        daily_support_levels = daily_supports.get(symbol, [])
                        for supp in daily_support_levels:
                             if supp < latest_close_f: # Only consider supports below current price
                                 potential_supports_below.append(supp)

                        if potential_supports_below:
                             # Find the highest support level below the current price
                             buy_low = max(potential_supports_below)
                             # Set buy high based on ATR multiplier above the support, but capped below current price
                             buy_high = latest_close_f # Default cap
                             try:
                                 atr_value = float(atr_value_str) if atr_value_str and atr_value_str != "NaN" else np.nan
                                 if pd.notna(atr_value) and atr_value > 0:
                                     # Calculate upper bound based on ATR
                                     atr_based_high = buy_low + (atr_value * BUY_RANGE_ATR_MULTIPLIER)
                                     # Cap the buy_high at the ATR-based level OR slightly below current close, whichever is lower
                                     buy_high = min(atr_based_high, latest_close_f * 0.998)
                                 else:
                                     # Fallback to percentage if ATR is invalid
                                     buy_high = min(buy_low * 1.01, latest_close_f * 0.998)
                             except (ValueError, TypeError):
                                  # Fallback to percentage on error
                                  buy_high = min(buy_low * 1.01, latest_close_f * 0.998)

                             if buy_high > buy_low: # Ensure range is valid
                                 buy_range_str = f"{buy_low:.4f} - {buy_high:.4f} (ATR)"
                             else: # If high is not > low, just indicate near support
                                 buy_range_str = f"Near {buy_low:.4f}?"
                        else: # No clear support found below
                            buy_range_str = f"Below {latest_close_f:.4f}?"

                    except (ValueError, TypeError) as e:
                        print(f"Error calculating buy range for {symbol} {tf_key}: {e}")
                        buy_range_str = "Error"

                    analysis_results[symbol]['timeframes'][tf_key]['Buy Range'] = buy_range_str

            else: # Handle case where score calculation failed
                analysis_results[symbol]['timeframes'][tf_key] = indicators if indicators else {"Error": "Score calculation failed"}

        # Store current overall score for notification check
        current_overall_score = analysis_results[symbol]['overall_score']
        current_scores[symbol] = current_overall_score

        # Check if notification threshold is crossed
        last_score = previous_scores.get(symbol, 0) # Get previous score, default to 0
        if current_overall_score >= OVERALL_SCORE_THRESHOLD and last_score < OVERALL_SCORE_THRESHOLD:
            print(f"--- Threshold crossed for {symbol} ({last_score:.2f} -> {current_overall_score:.2f}) ---")
            send_notification(symbol, current_overall_score)
        elif current_overall_score < OVERALL_SCORE_THRESHOLD and last_score >= OVERALL_SCORE_THRESHOLD:
             print(f"--- {symbol} dropped below threshold ({last_score:.2f} -> {current_overall_score:.2f}) ---")


    # --- Print Results (Telegram Friendly) ---
    print("\n--- Luno Multi-Timeframe DCA Analysis ---")
    sorted_symbols = sorted(analysis_results.items(), key=lambda item: item[1]['overall_score'], reverse=True)
    output_lines = []

    for symbol, data in sorted_symbols:
        overall_score = data['overall_score']
        highlight = "*" if overall_score >= OVERALL_SCORE_THRESHOLD else ""

        # Determine weekly status string
        is_price_above_ema_print = weekly_trend_price_ema.get(symbol)
        is_macd_bullish_print = weekly_trend_macd.get(symbol)
        is_strong_weekly_bullish_print = is_price_above_ema_print is True and is_macd_bullish_print is True
        if is_price_above_ema_print is None or is_macd_bullish_print is None:
            trend_status = "Unknown"
        elif is_strong_weekly_bullish_print:
            trend_status = "Confirmed Bullish"
        elif is_price_above_ema_print is True and is_macd_bullish_print is False:
             trend_status = "Price>EMA, MACD Bear"
        elif is_price_above_ema_print is False and is_macd_bullish_print is True:
             trend_status = "Price<EMA, MACD Bull"
        else: # Both False
             trend_status = "Confirmed Bearish"

        supports_str = ', '.join(f"{s:.4f}" for s in daily_supports.get(symbol, [])) or "N/A"
        resistances_str = ', '.join(f"{r:.4f}" for r in daily_resistances.get(symbol, [])) or "N/A"

        # Get forecast using the specific symbol's daily supports
        weekly_df = weekly_dataframes.get(symbol)
        symbol_supports = daily_supports.get(symbol, []) # Get supports for this symbol
        forecast_str = estimate_next_check_time(symbol, weekly_df, trend_status, symbol_supports)

        # Symbol Header
        output_lines.append(f"\n*{symbol}* ({highlight}Score: {overall_score:.2f})")
        output_lines.append(f"Weekly Trend: {trend_status}")
        output_lines.append(f"Forecast: {forecast_str}") # Add forecast here
        output_lines.append(f"Supports (1d): {supports_str}")
        output_lines.append(f"Resistances (1d): {resistances_str}")
        output_lines.append("--- Timeframes ---")

        # Timeframe Details
        sorted_tfs = sorted(data['timeframes'].keys(), key=lambda t: TIMEFRAMES_DCA.index(t))
        for tf_key in sorted_tfs:
            indicators = data['timeframes'][tf_key]
            tf_line = f"[{tf_key}] "
            if "Error" in indicators:
                tf_line += f"Error: {indicators['Error']}"
            elif "Info" in indicators:
                tf_line += indicators['Info']
            else:
                raw_s = indicators.get('Raw Score', 'N/A')
                weighted_s = indicators.get('Weighted Score', 'N/A')
                weighted_s_str = f"{weighted_s:.2f}" if isinstance(weighted_s, (float, int)) else "N/A"
                close_str = indicators.get('Close', 'N/A')
                buy_range_str = indicators.get('Buy Range', 'N/A')
                reasons_str = indicators.get('Reasons', 'N/A')
                # Add consecutive counts to output string
                ha_cons = indicators.get('HA Bull Cons', 0)
                rsi_cons = indicators.get('RSI OS Cons', 0)
                macd_cons = indicators.get('MACD Hist Incr Cons', 0)
                bb_cons = indicators.get('Price BB Low Cons', 0)
                cons_str = f"Cons(HA:{ha_cons},RSI_OS:{rsi_cons},MACD+:{macd_cons},BB_Low:{bb_cons})"

                tf_line += f"Raw: {str(raw_s)} | Wght: {weighted_s_str} | Close: {close_str} | BuyRng: {buy_range_str} | {cons_str} | Reasons: {reasons_str}"
            output_lines.append(tf_line)
        output_lines.append("-" * 20) # Separator

    # Print the combined output
    print("\n".join(output_lines))
    print(f"\n* Overall Weighted Score >= {OVERALL_SCORE_THRESHOLD:.1f} highlighted.")
    return current_scores # Return current scores for the next cycle

async def main_loop():
    """Main loop for continuous analysis."""
    if not LUNO_API_KEY_ID or not LUNO_API_KEY_SECRET:
        print("FATAL ERROR: LUNO_API_KEY_ID and LUNO_API_KEY_SECRET environment variables must be set.")
        return # Exit if keys are not set

    print("Luno DCA Analyzer started. Ensure API keys are set in environment variables.")
    print(f"Analyzing pairs: {', '.join(SYMBOLS_DCA)}")
    print(f"Using timeframes: {', '.join(TIMEFRAMES_DCA)}")
    print(f"Notification threshold: {OVERALL_SCORE_THRESHOLD:.1f}")
    print(f"Check interval: {CHECK_INTERVAL_SECONDS / 3600:.1f} hours")

    previous_scores = {}
    while True:
        try:
            # Use a timeout for the session connection
            timeout = aiohttp.ClientTimeout(total=60) # 60 seconds total timeout for all operations
            async with aiohttp.ClientSession(timeout=timeout) as session:
                 current_scores = await run_analysis_cycle(session, previous_scores)
                 previous_scores = current_scores
        except aiohttp.ClientConnectorError as e:
            print(f"Network connection error: {e}. Retrying after delay...")
            await asyncio.sleep(60) # Wait a minute before retrying connection errors
        except asyncio.TimeoutError:
             print(f"Network operation timed out. Retrying after delay...")
             await asyncio.sleep(60)
        except Exception as e:
            print(f"An unexpected error occurred in main loop: {e}")
            # Add a longer delay for unexpected errors to avoid spamming
            await asyncio.sleep(300) # Wait 5 minutes

        print(f"\nWaiting for {CHECK_INTERVAL_SECONDS / 3600:.1f} hours until next check...")
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        import uvloop
        uvloop.install()
        print("Using uvloop")
    except ImportError:
        print("uvloop not found, using default asyncio loop")

    # Run the continuous loop
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\nScript stopped by user.")
    except Exception as e:
         print(f"\nCritical error during script execution: {e}")
