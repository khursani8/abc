import asyncio
import aiohttp
import pandas as pd
import numpy as np
import os
import time
import subprocess # Keep for potential future use, but won't be called for email
import json # Added for config loading
import smtplib
import ssl
from email.message import EmailMessage # For constructing email
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from aiohttp import BasicAuth

# --- Load Configuration ---
CONFIG_FILE = 'config.json'
try:
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    print(f"Configuration loaded from {CONFIG_FILE}")
except FileNotFoundError:
    print(f"FATAL ERROR: Configuration file '{CONFIG_FILE}' not found.")
    exit(1)
except json.JSONDecodeError as e:
    print(f"FATAL ERROR: Could not decode JSON from '{CONFIG_FILE}': {e}")
    exit(1)
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred loading config: {e}")
    exit(1)

# --- Extract Config Values ---
# API
API_BASE_URL = config.get('api', {}).get('base_url', "https://api.luno.com/api/exchange/1")
LUNO_API_KEY_ID_ENV = config.get('api', {}).get('key_id_env_var', 'LUNO_API_KEY_ID')
LUNO_API_KEY_SECRET_ENV = config.get('api', {}).get('key_secret_env_var', 'LUNO_API_KEY_SECRET')
LUNO_API_KEY_ID = os.getenv(LUNO_API_KEY_ID_ENV)
LUNO_API_KEY_SECRET = os.getenv(LUNO_API_KEY_SECRET_ENV)

# Analysis
SYMBOLS_DCA = config.get('analysis', {}).get('symbols', [])
FETCH_LIMIT_DEFAULT = config.get('analysis', {}).get('fetch_limit_default', 500)
FETCH_LIMIT_WEEKLY = config.get('analysis', {}).get('fetch_limit_weekly', 300)
TIMEFRAME_MAP_LUNO = config.get('analysis', {}).get('timeframes', {})
TIMEFRAMES_DCA = list(TIMEFRAME_MAP_LUNO.keys())
TF_WEIGHTS = config.get('analysis', {}).get('tf_weights', {})
CHECK_INTERVAL_SECONDS = config.get('analysis', {}).get('check_interval_seconds', 14400) # Default 4 hours

# Indicators
indicators_cfg = config.get('indicators', {})
RSI_LENGTH = indicators_cfg.get('rsi_length', 14)
EMA_LENGTH = indicators_cfg.get('ema_length', 200)
EMA_LENGTH_WEEKLY = indicators_cfg.get('ema_length_weekly', 50)
BBANDS_LENGTH = indicators_cfg.get('bbands_length', 20)
BBANDS_STDDEV = indicators_cfg.get('bbands_stddev', 2.0)
MACD_FAST = indicators_cfg.get('macd_fast', 12)
MACD_SLOW = indicators_cfg.get('macd_slow', 26)
MACD_SIGNAL = indicators_cfg.get('macd_signal', 9)
ATR_LENGTH = indicators_cfg.get('atr_length', 14)
SUPERTREND_FACTOR_1 = indicators_cfg.get('supertrend_factor_1', 3.0)
SUPERTREND_FACTOR_2 = indicators_cfg.get('supertrend_factor_2', 2.0)
OBV_SMOOTHING = indicators_cfg.get('obv_smoothing', 5)
VOLUME_SMA_LENGTH = indicators_cfg.get('volume_sma_length', 20)
PIVOT_LOOKBACK = indicators_cfg.get('pivot_lookback', 2)
SUPPORT_PROXIMITY_ATR_MULTIPLIER = indicators_cfg.get('support_proximity_atr_multiplier', 0.75)
ADX_LENGTH = indicators_cfg.get('adx_length', 14)
BUY_RANGE_ATR_MULTIPLIER = indicators_cfg.get('buy_range_atr_multiplier', 0.5)

# Scoring
scoring_cfg = config.get('scoring', {})
RSI_OVERSOLD_STRONG = scoring_cfg.get('rsi_oversold_strong', 30)
BUY_RANGE_RAW_SCORE_THRESHOLD = scoring_cfg.get('buy_range_raw_score_threshold', 4)
BUY_RANGE_TIMEFRAMES = scoring_cfg.get('buy_range_timeframes', ['1d', '1w'])
OVERALL_SCORE_THRESHOLD = scoring_cfg.get('overall_score_threshold', 7.0) # Using updated default
ADX_THRESHOLD = scoring_cfg.get('adx_threshold', 25)
CONSECUTIVE_SIGNAL_THRESHOLD = scoring_cfg.get('consecutive_signal_threshold', 2)
POINTS = scoring_cfg.get('points', {}) # Load points dictionary

# Weekly Trend Confirmation Rules
weekly_confirm_cfg = config.get('weekly_trend_confirm', {})
WT_REQUIRE_ADX = weekly_confirm_cfg.get('require_adx', True)
WT_ADX_THRESHOLD = weekly_confirm_cfg.get('adx_threshold', 20)
WT_REQUIRE_HA_BULL = weekly_confirm_cfg.get('require_ha_bull', True)
WT_HA_CONSECUTIVE = weekly_confirm_cfg.get('ha_consecutive_periods', 1)


# --- Email Notification Function ---
def send_email_notification(symbol, score):
    """Sends an email notification using Gmail SMTP."""
    sender_email = os.getenv('EMAIL_SENDER')
    receiver_email = os.getenv('EMAIL_RECIPIENT')
    app_password = os.getenv('EMAIL_APP_PASSWORD')

    if not sender_email or not receiver_email or not app_password:
        print("Warning: Email credentials (EMAIL_SENDER, EMAIL_RECIPIENT, EMAIL_APP_PASSWORD) not found in environment variables. Skipping email notification.")
        return

    subject = f"Luno DCA Alert: {symbol} Threshold Crossed!"
    body = f"""
Potential DCA opportunity detected for {symbol}.

Overall Weighted Score: {score:.2f}
(Threshold: {OVERALL_SCORE_THRESHOLD:.1f})

Check the GitHub Actions logs for detailed analysis.
Timestamp (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}
"""

    em = EmailMessage()
    em['From'] = sender_email
    em['To'] = receiver_email
    em['Subject'] = subject
    em.set_content(body)

    # Add SSL context
    context = ssl.create_default_context()

    try:
        print(f"Attempting to send email notification for {symbol} to {receiver_email}...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(sender_email, app_password)
            smtp.sendmail(sender_email, receiver_email, em.as_string())
        print(f"--- Email notification successfully sent for {symbol} ---")
    except smtplib.SMTPAuthenticationError:
        print("ERROR: Gmail SMTP Authentication failed. Check sender email and App Password.")
    except smtplib.SMTPConnectError:
         print("ERROR: Could not connect to Gmail SMTP server. Check network/firewall.")
    except Exception as e:
        print(f"ERROR: Failed to send email notification for {symbol}: {e}")

# --- (Original send_notification function commented out or removed) ---
# def send_notification(symbol, score): ...

# --- Data Fetching (Adapted for Luno - uses config) ---
async def get_kline_data_dca(session, pair, duration_seconds, limit):
    """Fetches candlestick data from Luno API."""
    if not LUNO_API_KEY_ID or not LUNO_API_KEY_SECRET:
        print(f"Error: {LUNO_API_KEY_ID_ENV} or {LUNO_API_KEY_SECRET_ENV} not set.")
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

            # Determine minimum length needed based on indicators (using config values)
            min_len_needed = EMA_LENGTH if duration_seconds < 604800 else EMA_LENGTH_WEEKLY
            min_len_needed = max(min_len_needed, MACD_SLOW, BBANDS_LENGTH, RSI_LENGTH, ATR_LENGTH, ADX_LENGTH * 2, PIVOT_LOOKBACK * 2 + 1) # Added ADX length requirement

            if candles and len(candles) >= min_len_needed:
                df = pd.DataFrame(candles)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                return df
            else:
                # Fallback for basic indicators if EMA length not met
                min_basic_len = max(BBANDS_LENGTH, RSI_LENGTH, MACD_SLOW, PIVOT_LOOKBACK * 2 + 1, ADX_LENGTH * 2)
                if candles and len(candles) >= min_basic_len:
                    df = pd.DataFrame(candles)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col])
                    return df # Return with fewer data points, EMA/ADX might be NaN
                else:
                    print(f"Warning: Insufficient data for {pair} {duration_seconds}s. Got {len(candles)}, needed {min_len_needed} (or {min_basic_len}).")
                    return None
    except aiohttp.ClientResponseError as e:
        print(f"HTTP Error fetching {pair} {duration_seconds}s: {e.status} {e.message}")
        return None
    except Exception as e:
        print(f"Fetch error {pair} {duration_seconds}s: {e}")
        return None

# --- Manual Indicator Calculations (Using config values) ---
def calculate_rsi(series, length=RSI_LENGTH): # Use config default
    delta = series.diff(); gain = (delta.where(delta > 0, 0)).rolling(window=length).mean(); loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss.replace(0, np.nan); rsi = 100 - (100 / (1 + rs)); rsi = rsi.fillna(100)
    return rsi

def calculate_ema(series, length): # Length passed explicitly
    if len(series) < length: return pd.Series(np.nan, index=series.index)
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def calculate_bbands(series, length=BBANDS_LENGTH, std=BBANDS_STDDEV): # Use config defaults
    if len(series) < length: return pd.Series(np.nan, index=series.index), pd.Series(np.nan, index=series.index), pd.Series(np.nan, index=series.index)
    sma = series.rolling(window=length, min_periods=length).mean(); std_dev = series.rolling(window=length, min_periods=length).std()
    upper_band = sma + (std_dev * std); lower_band = sma - (std_dev * std)
    return upper_band, sma, lower_band

def calculate_macd(series, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL): # Use config defaults
    if len(series) < slow: return pd.Series(np.nan, index=series.index), pd.Series(np.nan, index=series.index), pd.Series(np.nan, index=series.index)
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean(); ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow; signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean(); histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(high, low, close, length=ATR_LENGTH): # Use config default
    if len(close) < length + 1: return pd.Series(np.nan, index=close.index)
    high_low = high - low; high_close = np.abs(high - close.shift()); low_close = np.abs(low - close.shift())
    tr_df = pd.concat([high_low, high_close, low_close], axis=1)
    tr = tr_df.max(axis=1, skipna=False)
    atr = tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    return atr

def calculate_supertrend(high, low, close, atr_length=ATR_LENGTH, factor=3.0): # Use config default for length
    # Factor passed explicitly as it might vary (ST1 vs ST2)
    if len(close) < atr_length + 1: return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)
    atr = calculate_atr(high, low, close, length=atr_length)
    if atr.isnull().all():
        return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)

    hl2 = (high + low) / 2
    upper_band = hl2 + (factor * atr)
    lower_band = hl2 - (factor * atr)
    supertrend = pd.Series(np.nan, index=close.index)
    trend = pd.Series(1, index=close.index)

    for i in range(1, len(close)):
        prev_close = close.iloc[i-1]
        prev_upper = upper_band.iloc[i-1]
        prev_lower = lower_band.iloc[i-1]
        prev_trend = trend.iloc[i-1]
        if pd.isna(prev_close) or pd.isna(prev_upper) or pd.isna(prev_lower):
            trend.iloc[i] = prev_trend
        elif prev_close > prev_upper: trend.iloc[i] = 1
        elif prev_close < prev_lower: trend.iloc[i] = -1
        else: trend.iloc[i] = prev_trend

        current_lower = lower_band.iloc[i]
        current_upper = upper_band.iloc[i]
        if trend.iloc[i] == 1:
            lower_band.iloc[i] = max(current_lower if pd.notna(current_lower) else -np.inf, prev_lower if pd.notna(prev_lower) else -np.inf)
            supertrend.iloc[i] = lower_band.iloc[i]
        else:
            upper_band.iloc[i] = min(current_upper if pd.notna(current_upper) else np.inf, prev_upper if pd.notna(prev_upper) else np.inf)
            supertrend.iloc[i] = upper_band.iloc[i]

    direction = pd.Series(np.where(close > supertrend, 1, -1), index=close.index)
    direction.ffill(inplace=True)
    return supertrend, direction

def calculate_obv(close, volume):
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def find_pivot_lows(low_series, n=PIVOT_LOOKBACK): # Use config default
    if len(low_series) < 2 * n + 1: return pd.Series(False, index=low_series.index)
    is_lower_than_prev = low_series < low_series.shift(1).rolling(window=n, min_periods=1).min()
    is_lower_than_next = low_series < low_series.shift(-n).rolling(window=n, min_periods=1).min()
    is_pivot = is_lower_than_prev & is_lower_than_next
    return is_pivot

def find_pivot_highs(high_series, n=PIVOT_LOOKBACK): # Use config default
    if len(high_series) < 2 * n + 1: return pd.Series(False, index=high_series.index)
    is_higher_than_prev = high_series > high_series.shift(1).rolling(window=n, min_periods=1).max()
    is_higher_than_next = high_series > high_series.shift(-n).rolling(window=n, min_periods=1).max()
    is_pivot = is_higher_than_prev & is_higher_than_next
    return is_pivot

def check_bullish_divergence(price_series, indicator_series, lookback=20): # Lookback kept hardcoded for now, could be config
    if len(price_series) < lookback + 1 or indicator_series.isnull().sum() > lookback // 2:
        return False, None, None
    recent_price_low_idx = price_series.iloc[-lookback-1:-1].idxmin()
    prev_price_low_idx = price_series.iloc[-lookback-1:price_series.index.get_loc(recent_price_low_idx)].idxmin()
    if pd.isna(recent_price_low_idx) or pd.isna(prev_price_low_idx): return False, None, None
    price_lower_low = price_series.loc[recent_price_low_idx] < price_series.loc[prev_price_low_idx]
    indicator_at_recent_low = indicator_series.loc[recent_price_low_idx]
    indicator_at_prev_low = indicator_series.loc[prev_price_low_idx]
    if pd.isna(indicator_at_recent_low) or pd.isna(indicator_at_prev_low): return False, None, None
    indicator_higher_low = indicator_at_recent_low > indicator_at_prev_low
    is_divergence = price_lower_low and indicator_higher_low
    return is_divergence, indicator_at_prev_low, indicator_at_recent_low

# --- ADX Calculation (Using config default) ---
def calculate_di(high, low, close, length=ADX_LENGTH):
    if len(close) < length + 1: return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)
    high_low = high - low; high_close = np.abs(high - close.shift()); low_close = np.abs(low - close.shift())
    tr_df = pd.concat([high_low, high_close, low_close], axis=1); tr = tr_df.max(axis=1, skipna=False)
    atr = tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    move_up = high.diff(); move_down = -low.diff()
    plus_dm = pd.Series(np.where((move_up > move_down) & (move_up > 0), move_up, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((move_down > move_up) & (move_down > 0), move_down, 0.0), index=close.index)
    smooth_plus_dm = plus_dm.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    plus_di = 100 * (smooth_plus_dm / atr.replace(0, np.nan)); minus_di = 100 * (smooth_minus_dm / atr.replace(0, np.nan))
    plus_di.fillna(0, inplace=True); minus_di.fillna(0, inplace=True)
    return plus_di, minus_di

def calculate_adx(high, low, close, length=ADX_LENGTH):
    if len(close) < 2 * length: return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)
    plus_di, minus_di = calculate_di(high, low, close, length=length)
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    dx.fillna(0, inplace=True)
    adx = dx.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    adx.fillna(0, inplace=True)
    return adx, plus_di, minus_di

# --- Consecutive Signal Calculation ---
def calculate_consecutive_true(series):
    if not isinstance(series, pd.Series) or series.dtype != bool: raise TypeError("Input must be a boolean Pandas Series.")
    shifted = series.shift(1, fill_value=False); change_points = (series != shifted)
    groups = change_points.cumsum(); consecutive_counts = series.groupby(groups).cumsum()
    return consecutive_counts.where(series, 0)

# --- Heiken Ashi Calculation ---
def calculate_heiken_ashi(df):
    ha_df = pd.DataFrame(index=df.index)
    ha_df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_df['HA_Open'] = ((df['open'].shift(1) + df['close'].shift(1)) / 2).fillna((df['open'].iloc[0] + df['close'].iloc[0]) / 2)
    for i in range(1, len(df)):
        ha_df.loc[ha_df.index[i], 'HA_Open'] = (ha_df.loc[ha_df.index[i-1], 'HA_Open'] + ha_df.loc[ha_df.index[i-1], 'HA_Close']) / 2
    ha_df['HA_High'] = ha_df[['HA_Open', 'HA_Close']].join(df['high']).max(axis=1)
    ha_df['HA_Low'] = ha_df[['HA_Open', 'HA_Close']].join(df['low']).min(axis=1)
    return ha_df

# --- Forecasting Function (Uses config values) ---
def estimate_next_check_time(symbol, weekly_df, weekly_status, daily_support_levels):
    if weekly_df is None or weekly_df.empty or len(weekly_df) < max(EMA_LENGTH_WEEKLY, MACD_SLOW + MACD_SIGNAL):
        return "N/A (No/Short Weekly Data)"
    now = datetime.now(timezone.utc); forecast_dates = []
    if weekly_status == "Confirmed Bullish": return "Monitor Buy Range"
    try:
        df_copy = weekly_df.copy()
        df_copy['EMA50'] = calculate_ema(df_copy['close'], length=EMA_LENGTH_WEEKLY)
        df_copy['MACD'], df_copy['MACD_Signal'], df_copy['MACD_Hist'] = calculate_macd(df_copy['close'])
        if len(df_copy) < 2 or df_copy[['close', 'EMA50', 'MACD_Hist']].iloc[-1].isnull().any(): return "N/A (Weekly Calc Error)"
        latest_close = df_copy['close'].iloc[-1]; latest_ema50 = df_copy['EMA50'].iloc[-1]
        latest_hist = df_copy['MACD_Hist'].iloc[-1]; prev_hist = df_copy['MACD_Hist'].iloc[-2] if len(df_copy['MACD_Hist'].dropna()) >= 2 else np.nan
        weekly_range = (df_copy['high'] - df_copy['low']).iloc[-10:].mean()
        # EMA50 Forecast
        if latest_close < latest_ema50:
            price_diff = latest_ema50 - latest_close
            if pd.notna(weekly_range) and weekly_range > 0:
                weeks_to_ema = price_diff / weekly_range; buffer = 1.5 if weekly_status == "Confirmed Bearish" else 1.2
                estimated_weeks = max(1, weeks_to_ema * buffer); forecast_dates.append(now + timedelta(weeks=estimated_weeks))
        # MACD Forecast
        if pd.notna(latest_hist) and latest_hist < 0 and pd.notna(prev_hist) and latest_hist > prev_hist:
            recent_hist = df_copy['MACD_Hist'].iloc[-5:].dropna()
            if len(recent_hist) >= 2:
                 hist_change_rate = (recent_hist.iloc[-1] - recent_hist.iloc[0]) / (len(recent_hist) -1) if len(recent_hist) > 1 else np.nan
                 if pd.notna(hist_change_rate) and hist_change_rate > 1e-9:
                     weeks_to_zero = abs(latest_hist / hist_change_rate); estimated_weeks = max(1, weeks_to_zero * 1.3)
                     forecast_dates.append(now + timedelta(weeks=estimated_weeks))
        # Support Forecast
        supports_below_close = [s for s in daily_support_levels if s < latest_close]
        if supports_below_close:
            closest_support = max(supports_below_close); price_diff_support = latest_close - closest_support
            if pd.notna(weekly_range) and weekly_range > 0:
                 weeks_to_support = price_diff_support / weekly_range; buffer = 1.1
                 estimated_weeks = max(1, weeks_to_support * buffer); forecast_dates.append(now + timedelta(weeks=estimated_weeks))
        # Combine
        if forecast_dates:
            earliest_date = min(forecast_dates); latest_date = max(forecast_dates)
            if earliest_date == latest_date: return f"Est. Check: {earliest_date.strftime('%Y-%m-%d')}"
            elif latest_date > earliest_date: return f"Est. Check: {earliest_date.strftime('%Y-%m-%d')} - {latest_date.strftime('%Y-%m-%d')}"
            else: return f"Est. Check: {earliest_date.strftime('%Y-%m-%d')}" # Fallback
        else:
            check_date = now + timedelta(days=(7 if weekly_status == "Confirmed Bearish" else 3))
            return f"Est. Check: {check_date.strftime('%Y-%m-%d')}"
    except Exception as e: return "N/A (Forecast Error)"

# --- Scoring Function (Uses config values) ---
def calculate_tf_score(df, timeframe_key):
    duration_seconds = TIMEFRAME_MAP_LUNO[timeframe_key]
    min_len_needed = max(BBANDS_LENGTH, RSI_LENGTH, MACD_SLOW, ADX_LENGTH * 2) # Ensure enough for ADX
    if df is None or df.empty or len(df) < min_len_needed:
        return None, {"Error": f"Insufficient data ({len(df) if df is not None else 0} < {min_len_needed})"}

    # Calculate indicators
    df['RSI'] = calculate_rsi(df['close'])
    current_ema_length = EMA_LENGTH_WEEKLY if timeframe_key == '1w' else EMA_LENGTH
    df['EMA'] = calculate_ema(df['close'], length=current_ema_length)
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bbands(df['close'])
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['close'])
    try:
        df['Supertrend_1'], df['ST1_Direction'] = calculate_supertrend(df['high'], df['low'], df['close'], factor=SUPERTREND_FACTOR_1)
        df['Supertrend_2'], df['ST2_Direction'] = calculate_supertrend(df['high'], df['low'], df['close'], factor=SUPERTREND_FACTOR_2)
    except Exception as e:
        print(f"Error calculating Supertrends for {timeframe_key}: {e}")
        df['Supertrend_1'], df['ST1_Direction'] = np.nan, np.nan; df['Supertrend_2'], df['ST2_Direction'] = np.nan, np.nan
    df['OBV'] = calculate_obv(df['close'], df['volume'])
    df['Volume_SMA'] = df['volume'].rolling(window=VOLUME_SMA_LENGTH).mean()
    df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])
    df['ADX'], df['PlusDI'], df['MinusDI'] = calculate_adx(df['high'], df['low'], df['close'])
    ha_df = calculate_heiken_ashi(df); df = df.join(ha_df)

    if len(df) < 2: return 0, {"Error": "Need at least 2 data points for comparison"}
    latest = df.iloc[-1]; previous = df.iloc[-2]
    raw_score = 0; reasons = []; indicator_values = {}

    required_score_cols = ['RSI', 'BB_Lower', 'MACD', 'MACD_Signal', 'MACD_Hist', 'ADX', 'PlusDI', 'MinusDI', 'HA_Open', 'HA_Close', 'HA_Low']
    if latest[required_score_cols].isnull().any() or pd.isna(previous['MACD_Hist']):
        nan_cols = latest[required_score_cols][latest[required_score_cols].isnull()].index.tolist()
        if pd.isna(previous['MACD_Hist']): nan_cols.append("Prev MACD_Hist")
        return 0, {"Error": f"NaN in scoring indicators: {', '.join(nan_cols)}"}

    latest_close = latest['close']; indicator_values['Close'] = f"{latest_close:.4f}"

    # --- Scoring Logic (Using POINTS from config) ---
    # EMA
    ema_value = latest['EMA']; ema_col_name = f"EMA{current_ema_length}"
    indicator_values[ema_col_name] = f"{ema_value:.4f}" if pd.notna(ema_value) else "NaN"
    if pd.notna(ema_value) and latest_close <= ema_value:
        raw_score += POINTS.get('price_vs_ema', 0); reasons.append(f"Price<=EMA{current_ema_length}")

    # RSI + BB
    rsi_value = latest['RSI']; bbl_value = latest['BB_Lower']
    volume_value = latest['volume']; volume_sma_value = latest['Volume_SMA']
    indicator_values['RSI'] = f"{rsi_value:.2f}"; indicator_values['BB Lower'] = f"{bbl_value:.4f}" if pd.notna(bbl_value) else "NaN"
    indicator_values['Volume'] = f"{volume_value:.2f}" if pd.notna(volume_value) else "NaN"; indicator_values['Volume SMA'] = f"{volume_sma_value:.2f}" if pd.notna(volume_sma_value) else "NaN"
    rsi_bb_condition = rsi_value < RSI_OVERSOLD_STRONG and pd.notna(bbl_value) and latest_close <= bbl_value
    if rsi_bb_condition:
        raw_score += POINTS.get('rsi_bb_confirm', 0); reasons.append(f"RSI<{RSI_OVERSOLD_STRONG}+Price<=BB")
        if pd.notna(volume_value) and pd.notna(volume_sma_value) and volume_value > volume_sma_value:
            raw_score += POINTS.get('rsi_bb_vol_bonus', 0); reasons.append("Vol>SMA")

    # MACD
    macd_line = latest['MACD']; signal_line = latest['MACD_Signal']; hist = latest['MACD_Hist']; prev_hist = previous['MACD_Hist']
    indicator_values['MACD Hist'] = f"{hist:.4f}" if pd.notna(hist) else "NaN"
    if pd.notna(macd_line) and pd.notna(signal_line) and macd_line > signal_line:
        raw_score += POINTS.get('macd_vs_signal', 0); reasons.append("MACD>Signal")
    if pd.notna(hist) and pd.notna(prev_hist) and hist > prev_hist:
        raw_score += POINTS.get('macd_hist_incr', 0); reasons.append("MACD Hist Incr")
    if timeframe_key == '1w' and pd.notna(hist) and pd.notna(prev_hist) and hist > 0 and hist > prev_hist:
         raw_score += POINTS.get('macd_hist_wkly_confirm', 0); reasons.append("Wkly MACD Hist>0 Incr")

    # Supertrend
    st1_dir = latest['ST1_Direction']; st2_dir = latest['ST2_Direction']
    indicator_values['ST1 Dir'] = 'Up' if st1_dir == 1 else 'Down' if st1_dir == -1 else 'N/A'; indicator_values['ST2 Dir'] = 'Up' if st2_dir == 1 else 'Down' if st2_dir == -1 else 'N/A'
    if st1_dir == 1 and st2_dir == 1:
        raw_score += POINTS.get('supertrend_confirm', 0); reasons.append("ST1+ST2 Up")

    # OBV
    obv_value = latest['OBV']; indicator_values['OBV'] = f"{obv_value:.0f}" if pd.notna(obv_value) else "NaN"
    obv_rising = False
    if len(df) > OBV_SMOOTHING and pd.notna(obv_value) and df['OBV'].iloc[-OBV_SMOOTHING:].notna().all():
        obv_prev_smoothed = df['OBV'].iloc[-OBV_SMOOTHING-1:-1].mean()
        if pd.notna(obv_prev_smoothed) and obv_value > obv_prev_smoothed: obv_rising = True
    indicator_values['OBV Rising?'] = 'Yes' if obv_rising else 'No'
    if obv_rising: raw_score += POINTS.get('obv_rising', 0); reasons.append("OBV Rising")

    # Heiken Ashi
    ha_open = latest['HA_Open']; ha_close = latest['HA_Close']; ha_low = latest['HA_Low']
    ha_bullish = pd.notna(ha_close) and pd.notna(ha_open) and ha_close > ha_open
    ha_strong_bullish = ha_bullish and ha_open == ha_low
    indicator_values['HA Candle'] = 'Bull' if ha_bullish else 'Bear' if pd.notna(ha_close) and pd.notna(ha_open) and ha_close < ha_open else 'Doji'
    indicator_values['HA Strength'] = 'Strong' if ha_strong_bullish else 'Normal' if ha_bullish else 'N/A'
    if ha_bullish:
        raw_score += POINTS.get('ha_bull', 0); reasons.append("HA Bull")
        if ha_strong_bullish:
            raw_score += POINTS.get('ha_strong', 0); reasons.append("HA Strong")

    # ADX
    adx_value = latest['ADX']; plus_di_value = latest['PlusDI']; minus_di_value = latest['MinusDI']
    indicator_values['ADX'] = f"{adx_value:.2f}" if pd.notna(adx_value) else "NaN"
    indicator_values['+DI'] = f"{plus_di_value:.2f}" if pd.notna(plus_di_value) else "NaN"; indicator_values['-DI'] = f"{minus_di_value:.2f}" if pd.notna(minus_di_value) else "NaN"
    is_trending = pd.notna(adx_value) and adx_value > ADX_THRESHOLD
    is_bullish_trend = pd.notna(plus_di_value) and pd.notna(minus_di_value) and plus_di_value > minus_di_value
    if is_trending and is_bullish_trend:
        raw_score += POINTS.get('adx_strong_bull', 0); reasons.append(f"ADX>{ADX_THRESHOLD} Bull")

    # Daily Support Proximity
    if timeframe_key == '1d':
        symbol = df.name; support_levels = daily_supports.get(symbol, []); daily_atr = latest['ATR']
        if support_levels and pd.notna(daily_atr) and daily_atr > 0:
            closest_support = max(support_levels)
            proximity_upper_bound = closest_support + (daily_atr * SUPPORT_PROXIMITY_ATR_MULTIPLIER)
            if latest_close >= closest_support and latest_close <= proximity_upper_bound:
                 raw_score += POINTS.get('near_daily_support', 0); reasons.append(f"Near D Supp({closest_support:.4f}, ATR)")

    # Divergence (Daily only)
    if timeframe_key == '1d':
        rsi_divergence, _, _ = check_bullish_divergence(df['low'], df['RSI'])
        macd_hist_divergence, _, _ = check_bullish_divergence(df['low'], df['MACD_Hist'])
        if rsi_divergence: raw_score += POINTS.get('rsi_divergence', 0); reasons.append("RSI Bull Div"); indicator_values['RSI Divergence'] = 'Yes'
        if macd_hist_divergence: raw_score += POINTS.get('macd_divergence', 0); reasons.append("MACD Bull Div"); indicator_values['MACD Divergence'] = 'Yes'

    indicator_values['Raw Score'] = raw_score

    # --- Calculate & Score Consecutive Signals ---
    try:
        # HA Bullish Consecutive
        df['HA_Bullish_Signal'] = df['HA_Close'] > df['HA_Open']
        df['HA_Bull_Consec'] = calculate_consecutive_true(df['HA_Bullish_Signal'])
        ha_bull_cons_count = int(df['HA_Bull_Consec'].iloc[-1]) if pd.notna(df['HA_Bull_Consec'].iloc[-1]) else 0
        indicator_values['HA Bull Cons'] = ha_bull_cons_count
        if ha_bull_cons_count >= CONSECUTIVE_SIGNAL_THRESHOLD:
            raw_score += POINTS.get('consecutive_ha_bull_bonus', 0); reasons.append(f"HA Bull Cons>={CONSECUTIVE_SIGNAL_THRESHOLD}")

        # RSI Oversold Consecutive
        df['RSI_Oversold_Signal'] = df['RSI'] < RSI_OVERSOLD_STRONG
        df['RSI_OS_Consec'] = calculate_consecutive_true(df['RSI_Oversold_Signal'])
        indicator_values['RSI OS Cons'] = int(df['RSI_OS_Consec'].iloc[-1]) if pd.notna(df['RSI_OS_Consec'].iloc[-1]) else 0
        # No scoring bonus for consecutive oversold usually

        # MACD Hist Increasing Consecutive
        df['MACD_Hist_Incr_Signal'] = df['MACD_Hist'] > df['MACD_Hist'].shift(1)
        df['MACD_Hist_Incr_Consec'] = calculate_consecutive_true(df['MACD_Hist_Incr_Signal'])
        macd_hist_incr_cons_count = int(df['MACD_Hist_Incr_Consec'].iloc[-1]) if pd.notna(df['MACD_Hist_Incr_Consec'].iloc[-1]) else 0
        indicator_values['MACD Hist Incr Cons'] = macd_hist_incr_cons_count
        if macd_hist_incr_cons_count >= CONSECUTIVE_SIGNAL_THRESHOLD:
             raw_score += POINTS.get('consecutive_macd_incr_bonus', 0); reasons.append(f"MACD+ Cons>={CONSECUTIVE_SIGNAL_THRESHOLD}")

        # Price <= BB Lower Consecutive
        df['Price_BB_Lower_Signal'] = df['close'] <= df['BB_Lower']
        df['Price_BB_Lower_Consec'] = calculate_consecutive_true(df['Price_BB_Lower_Signal'])
        indicator_values['Price BB Low Cons'] = int(df['Price_BB_Lower_Consec'].iloc[-1]) if pd.notna(df['Price_BB_Lower_Consec'].iloc[-1]) else 0
        # No scoring bonus for consecutive price below BB

    except Exception as e:
        print(f"Error calculating consecutive signals for {timeframe_key}: {e}")
        indicator_values['HA Bull Cons'] = 'Err'; indicator_values['RSI OS Cons'] = 'Err'
        indicator_values['MACD Hist Incr Cons'] = 'Err'; indicator_values['Price BB Low Cons'] = 'Err'

    indicator_values['Reasons'] = ", ".join(reasons) if reasons else "None"
    indicator_values['Buy Range'] = "N/A" # Will be calculated later if conditions met

    return raw_score, indicator_values

# --- Main Execution Logic (Uses config values) ---
daily_supports = {}; daily_resistances = {} # Global dicts

async def run_analysis_cycle(session, previous_scores):
    global daily_supports, daily_resistances
    tasks = []; task_info = []

    for symbol in SYMBOLS_DCA:
        for tf_key, duration_sec in TIMEFRAME_MAP_LUNO.items():
            limit = FETCH_LIMIT_WEEKLY if tf_key == '1w' else FETCH_LIMIT_DEFAULT
            tasks.append(get_kline_data_dca(session, symbol, duration_sec, limit))
            task_info.append({'symbol': symbol, 'timeframe': tf_key, 'duration': duration_sec})

    print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}] Fetching Luno data...")
    all_results_raw = await asyncio.gather(*tasks, return_exceptions=True)
    print("Luno data fetching complete.")

    data_by_symbol_tf = defaultdict(dict)
    weekly_dataframes = {} # Store weekly dataframes for forecasting & trend check
    daily_supports = {}; daily_resistances = {} # Reset global dicts

    # Process fetched data - First pass (store data, calc daily S/R)
    for i, result in enumerate(all_results_raw):
        info = task_info[i]; symbol = info['symbol']; tf_key = info['timeframe']
        if isinstance(result, pd.DataFrame):
            data_by_symbol_tf[symbol][tf_key] = result
            if tf_key == '1w': weekly_dataframes[symbol] = result # Store weekly df
            if tf_key == '1d': # Calculate Daily S/R
                daily_df = result
                low_pivots = find_pivot_lows(daily_df['low']); current_daily_close = daily_df['close'].iloc[-1]
                confirmed_pivots_below_close = daily_df['low'][low_pivots].iloc[:-1]
                supports_below_close = confirmed_pivots_below_close[confirmed_pivots_below_close < current_daily_close]
                daily_supports[symbol] = sorted(supports_below_close.tail(3).round(4).tolist())
                high_pivots = find_pivot_highs(daily_df['high'])
                daily_resistances[symbol] = daily_df['high'][high_pivots].iloc[:-1].tail(3).round(4).tolist()
        elif isinstance(result, Exception): data_by_symbol_tf[symbol][tf_key] = {"Error": f"Data fetch failed: {result}"}
        else: data_by_symbol_tf[symbol][tf_key] = {"Error": "Insufficient data or fetch error"}
        # Ensure S/R lists exist even if 1d fetch failed
        if symbol not in daily_supports: daily_supports[symbol] = []
        if symbol not in daily_resistances: daily_resistances[symbol] = []

    # --- Perform Analysis (Second Pass - Scoring & Weekly Trend Check) ---
    analysis_results = defaultdict(lambda: {'overall_score': 0.0, 'timeframes': {}, 'weekly_status': 'Unknown'})
    current_scores = {}

    for symbol, tf_data in data_by_symbol_tf.items():
        weekly_df = weekly_dataframes.get(symbol)
        weekly_status_str = "Unknown"
        is_strong_weekly_bullish = False

        # --- Refined Weekly Trend Check ---
        if weekly_df is not None and not weekly_df.empty and len(weekly_df) >= max(EMA_LENGTH_WEEKLY, MACD_SLOW + MACD_SIGNAL, ADX_LENGTH * 2):
            # Calculate necessary weekly indicators
            weekly_df['EMA50'] = calculate_ema(weekly_df['close'], length=EMA_LENGTH_WEEKLY)
            weekly_df['MACD'], weekly_df['MACD_Signal'], _ = calculate_macd(weekly_df['close'])
            weekly_df['ADX'], weekly_df['PlusDI'], weekly_df['MinusDI'] = calculate_adx(weekly_df['high'], weekly_df['low'], weekly_df['close'])
            ha_df_weekly = calculate_heiken_ashi(weekly_df); weekly_df = weekly_df.join(ha_df_weekly)
            weekly_df['HA_Bullish_Signal'] = weekly_df['HA_Close'] > weekly_df['HA_Open']
            weekly_df['HA_Bull_Consec'] = calculate_consecutive_true(weekly_df['HA_Bullish_Signal'])

            latest_w = weekly_df.iloc[-1]

            # Check base conditions
            price_ok = pd.notna(latest_w['close']) and pd.notna(latest_w['EMA50']) and latest_w['close'] > latest_w['EMA50']
            macd_ok = pd.notna(latest_w['MACD']) and pd.notna(latest_w['MACD_Signal']) and latest_w['MACD'] > latest_w['MACD_Signal']

            # Check optional conditions based on config
            adx_ok = (not WT_REQUIRE_ADX) or (pd.notna(latest_w['ADX']) and latest_w['ADX'] > WT_ADX_THRESHOLD and pd.notna(latest_w['PlusDI']) and pd.notna(latest_w['MinusDI']) and latest_w['PlusDI'] > latest_w['MinusDI'])
            ha_ok = (not WT_REQUIRE_HA_BULL) or (pd.notna(latest_w['HA_Bull_Consec']) and latest_w['HA_Bull_Consec'] >= WT_HA_CONSECUTIVE)

            # Determine final status
            if price_ok and macd_ok and adx_ok and ha_ok:
                is_strong_weekly_bullish = True
                weekly_status_str = "Confirmed Bullish"
            elif price_ok and macd_ok: # Base bullish met, but not extras
                 weekly_status_str = "Weak Bullish" # Or some other indicator
            elif latest_w['close'] <= latest_w['EMA50'] and latest_w['MACD'] <= latest_w['MACD_Signal']:
                 weekly_status_str = "Confirmed Bearish"
            else: # Mixed signals
                 weekly_status_str = "Mixed"
        analysis_results[symbol]['weekly_status'] = weekly_status_str
        # --- End Refined Weekly Trend Check ---


        for tf_key, df_or_error in tf_data.items():
            if isinstance(df_or_error, dict) and "Error" in df_or_error:
                analysis_results[symbol]['timeframes'][tf_key] = df_or_error; continue
            df = df_or_error

            # Skip lower TFs if weekly trend isn't bullish
            if tf_key in ['1h', '4h', '1d'] and not is_strong_weekly_bullish:
                analysis_results[symbol]['timeframes'][tf_key] = {"Info": f"Skipped (Weekly: {weekly_status_str})"}; continue

            df_copy = df.copy(); df_copy.name = symbol
            raw_score, indicators = calculate_tf_score(df_copy, tf_key)

            if raw_score is not None:
                timeframe_weight = TF_WEIGHTS.get(tf_key, 1.0)
                weighted_score = raw_score * timeframe_weight
                indicators['Weighted Score'] = weighted_score
                analysis_results[symbol]['timeframes'][tf_key] = indicators
                analysis_results[symbol]['overall_score'] += weighted_score

                # Calculate Buy Range (using config multiplier)
                if raw_score >= BUY_RANGE_RAW_SCORE_THRESHOLD and tf_key in BUY_RANGE_TIMEFRAMES:
                    latest_close_str = indicators.get('Close'); bbl_value_str = indicators.get('BB Lower')
                    ema_value_str = indicators.get(f"EMA{EMA_LENGTH_WEEKLY if tf_key == '1w' else EMA_LENGTH}")
                    atr_value_str = indicators.get('ATR')
                    buy_range_str = "N/A"
                    try:
                        latest_close_f = float(latest_close_str) if latest_close_str and latest_close_str != "N/A" else np.nan
                        bbl_value = float(bbl_value_str) if bbl_value_str and bbl_value_str != "NaN" else np.nan
                        ema_value = float(ema_value_str) if ema_value_str and ema_value_str != "NaN" else np.nan
                        potential_supports_below = []
                        if pd.notna(bbl_value) and bbl_value < latest_close_f: potential_supports_below.append(bbl_value)
                        if pd.notna(ema_value) and ema_value < latest_close_f: potential_supports_below.append(ema_value)
                        daily_support_levels = daily_supports.get(symbol, [])
                        for supp in daily_support_levels:
                             if supp < latest_close_f: potential_supports_below.append(supp)
                        if potential_supports_below:
                             buy_low = max(potential_supports_below); buy_high = latest_close_f # Default cap
                             try:
                                 atr_value = float(atr_value_str) if atr_value_str and atr_value_str != "NaN" else np.nan
                                 if pd.notna(atr_value) and atr_value > 0:
                                     atr_based_high = buy_low + (atr_value * BUY_RANGE_ATR_MULTIPLIER)
                                     buy_high = min(atr_based_high, latest_close_f * 0.998) # Cap below close
                                 else: buy_high = min(buy_low * 1.01, latest_close_f * 0.998) # Fallback %
                             except (ValueError, TypeError): buy_high = min(buy_low * 1.01, latest_close_f * 0.998) # Fallback %
                             if buy_high > buy_low: buy_range_str = f"{buy_low:.4f} - {buy_high:.4f} (ATR)"
                             else: buy_range_str = f"Near {buy_low:.4f}?"
                        else: buy_range_str = f"Below {latest_close_f:.4f}?"
                    except (ValueError, TypeError) as e: print(f"Error calc buy range {symbol} {tf_key}: {e}"); buy_range_str = "Error"
                    analysis_results[symbol]['timeframes'][tf_key]['Buy Range'] = buy_range_str
            else: analysis_results[symbol]['timeframes'][tf_key] = indicators if indicators else {"Error": "Score calculation failed"}

        current_overall_score = analysis_results[symbol]['overall_score']
        current_scores[symbol] = current_overall_score
        last_score = previous_scores.get(symbol, 0)
        if current_overall_score >= OVERALL_SCORE_THRESHOLD and last_score < OVERALL_SCORE_THRESHOLD:
            print(f"--- Threshold crossed for {symbol} ({last_score:.2f} -> {current_overall_score:.2f}) ---")
            send_email_notification(symbol, current_overall_score) # Call the new email function
        elif current_overall_score < OVERALL_SCORE_THRESHOLD and last_score >= OVERALL_SCORE_THRESHOLD:
             print(f"--- {symbol} dropped below threshold ({last_score:.2f} -> {current_overall_score:.2f}) ---")

    # --- Print Results (Enhanced Output) ---
    print("\n--- Luno Multi-Timeframe DCA Analysis ---")
    sorted_symbols = sorted(analysis_results.items(), key=lambda item: item[1]['overall_score'], reverse=True)
    output_lines = []
    for symbol, data in sorted_symbols:
        overall_score = data['overall_score']; highlight = "*" if overall_score >= OVERALL_SCORE_THRESHOLD else ""
        trend_status = data['weekly_status'] # Get calculated weekly status
        supports_str = ', '.join(f"{s:.4f}" for s in daily_supports.get(symbol, [])) or "N/A"
        resistances_str = ', '.join(f"{r:.4f}" for r in daily_resistances.get(symbol, [])) or "N/A"
        weekly_df = weekly_dataframes.get(symbol); symbol_supports = daily_supports.get(symbol, [])
        forecast_str = estimate_next_check_time(symbol, weekly_df, trend_status, symbol_supports)

        output_lines.append(f"\n*{symbol}* ({highlight}Score: {overall_score:.2f})")
        output_lines.append(f"Weekly Trend: {trend_status}")
        output_lines.append(f"Forecast: {forecast_str}")
        output_lines.append(f"Supports (1d): {supports_str}")
        output_lines.append(f"Resistances (1d): {resistances_str}")
        output_lines.append("--- Timeframes ---")

        sorted_tfs = sorted(data['timeframes'].keys(), key=lambda t: TIMEFRAMES_DCA.index(t))
        for tf_key in sorted_tfs:
            indicators = data['timeframes'][tf_key]; tf_line = f"[{tf_key}] "
            if "Error" in indicators: tf_line += f"Error: {indicators['Error']}"
            elif "Info" in indicators: tf_line += indicators['Info']
            else:
                raw_s = indicators.get('Raw Score', 'N/A'); weighted_s = indicators.get('Weighted Score', 'N/A')
                weighted_s_str = f"{weighted_s:.2f}" if isinstance(weighted_s, (float, int)) else "N/A"
                close_str = indicators.get('Close', 'N/A'); buy_range_str = indicators.get('Buy Range', 'N/A')
                reasons_str = indicators.get('Reasons', 'N/A')
                # Enhanced Output Values
                adx_str = indicators.get('ADX', 'N/A'); plus_di_str = indicators.get('+DI', 'N/A'); minus_di_str = indicators.get('-DI', 'N/A')
                ha_candle = indicators.get('HA Candle', 'N/A'); ha_strength = indicators.get('HA Strength', 'N/A')
                ha_cons = indicators.get('HA Bull Cons', 0); rsi_cons = indicators.get('RSI OS Cons', 0)
                macd_cons = indicators.get('MACD Hist Incr Cons', 0); bb_cons = indicators.get('Price BB Low Cons', 0)
                cons_str = f"Cons(HA:{ha_cons},RSI_OS:{rsi_cons},MACD+:{macd_cons},BB_Low:{bb_cons})"
                detail_str = f"ADX:{adx_str}(+{plus_di_str}|-{minus_di_str}) HA:{ha_candle}({ha_strength})"

                tf_line += f"Raw:{str(raw_s)}|Wght:{weighted_s_str}|Close:{close_str}|BuyRng:{buy_range_str}|{detail_str}|{cons_str}|Reasons:{reasons_str}"
            output_lines.append(tf_line)
        output_lines.append("-" * 20)

    print("\n".join(output_lines))
    print(f"\n* Overall Weighted Score >= {OVERALL_SCORE_THRESHOLD:.1f} highlighted.")
    return current_scores

async def main_loop():
    """Main loop for continuous analysis."""
    if not LUNO_API_KEY_ID or not LUNO_API_KEY_SECRET:
        print(f"FATAL ERROR: {LUNO_API_KEY_ID_ENV} and {LUNO_API_KEY_SECRET_ENV} environment variables must be set.")
        return

    print("Luno DCA Analyzer started. Ensure API keys are set in environment variables.")
    print(f"Using config file: {CONFIG_FILE}")
    print(f"Analyzing pairs: {', '.join(SYMBOLS_DCA)}")
    print(f"Using timeframes: {', '.join(TIMEFRAMES_DCA)}")
    print(f"Notification threshold: {OVERALL_SCORE_THRESHOLD:.1f}")
    print(f"Check interval: {CHECK_INTERVAL_SECONDS / 3600:.1f} hours")

    previous_scores = {}
    while True:
        try:
            timeout = aiohttp.ClientTimeout(total=120) # Increased timeout slightly
            async with aiohttp.ClientSession(timeout=timeout) as session:
                 current_scores = await run_analysis_cycle(session, previous_scores)
                 previous_scores = current_scores
        except aiohttp.ClientConnectorError as e:
            print(f"Network connection error: {e}. Retrying after delay...")
            await asyncio.sleep(60)
        except asyncio.TimeoutError:
             print(f"Network operation timed out. Retrying after delay...")
             await asyncio.sleep(60)
        except Exception as e:
            print(f"An unexpected error occurred in main loop: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            await asyncio.sleep(300)

        print(f"\nWaiting for {CHECK_INTERVAL_SECONDS / 3600:.1f} hours until next check...")
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        import uvloop
        uvloop.install()
        print("Using uvloop")
    except ImportError:
        print("uvloop not found, using default asyncio loop")

    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\nScript stopped by user.")
    except Exception as e:
         print(f"\nCritical error during script execution: {e}")
         import traceback
         traceback.print_exc() # Print full traceback for debugging
