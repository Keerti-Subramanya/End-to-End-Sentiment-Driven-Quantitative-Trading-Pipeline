
from SmartApi import SmartConnect
import pyotp
import pandas as pd
import numpy as np
import ta
import requests
import time
from datetime import datetime, timedelta

# --- 1. USER CONFIGURATION ---
TRADING_API_KEY = "your angel one api key"       # Keep this safe for your order execution script later
HISTORICAL_API_KEY = "your angel one api key" # This is the one we will use for this script
CLIENT_ID = "your angel one client id"
PASSWORD = "password"
TOTP_KEY = "your totp key"

# YOUR ALIASES (Priority List)
ALIAS_MAP = {
    'TATAMOTORS.NS': ['TATAMOTORS', 'TMCV', 'TMPV', 'TATAMOTORS-EQ'],
    'TCS.NS': ['TCS', 'TCS-EQ'],
    'TATASTEEL.NS': ['TATASTEEL', 'TATASTEEL-EQ'],
    'RELIANCE.NS': ['RELIANCE', 'RELIANCE-EQ'],
    'SBIN.NS': ['SBIN', 'SBIN-EQ'],
    'HDFCBANK.NS': ['HDFCBANK', 'HDFCBANK-EQ'],
    'ITC.NS': ['ITC', 'ITC-EQ'],
    'SUNPHARMA.NS': ['SUNPHARMA', 'SUNPHARMA-EQ'],
    'ADANIPORTS.NS': ['ADANIPORTS', 'ADANIPORTS-EQ'],
    'LT.NS': ['LT', 'LT-EQ']
}

# --- 2. DYNAMIC TOKEN MAPPER (Fixed Logic) ---
def get_angel_tokens():
    print(" Downloading latest Angel One instrument list...")
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    try:
        data = requests.get(url).json()

        # Create a lookup dictionary: Symbol -> Token
        # We store 'RELIANCE-EQ' as key, and also 'RELIANCE' (stripped) as key
        full_map = {}
        for item in data:
            if item['exch_seg'] == 'NSE':
                symbol = item['symbol']
                token = item['token']
                full_map[symbol] = token
                if symbol.endswith('-EQ'):
                    full_map[symbol.replace('-EQ', '')] = token

        final_token_map = {}

        # Resolve Tokens using Alias Map
        for target_ticker, aliases in ALIAS_MAP.items():
            found = False
            for alias in aliases:
                # We check if the alias exists in our downloaded map
                # (The map now contains both 'RELIANCE' and 'RELIANCE-EQ')
                if alias in full_map:
                    final_token_map[target_ticker] = full_map[alias]
                    found = True
                    break

            if not found:
                print(f" Could not find token for {target_ticker} (Tried: {aliases})")

        print(f" Successfully mapped {len(final_token_map)} tickers.")
        return final_token_map

    except Exception as e:
        print(f" Error fetching token map: {e}")
        return {}

# --- 3. LOGIN FUNCTION ---
def login():
    try:
        # 🛠️ EXACT CHANGE: Pass the Historical API Key here
        obj = SmartConnect(api_key=HISTORICAL_API_KEY)
        totp = pyotp.TOTP(TOTP_KEY).now()
        data = obj.generateSession(CLIENT_ID, PASSWORD, totp)
        if data['status']:
            print(" Login Successful")
            return obj
        else:
            print(" Login Failed:", data['message'])
            return None
    except Exception as e:
        print(f" Connection Error: {e}")
        return None

# --- 4. FEATURE ENGINEERING ---
def calculate_indicators(df, ticker):
    """Calculates the exact features your model expects."""
    df = df.copy()

    # 1. RSI (14)
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    # 2. SMA_50
    df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()

    # 3. Price vs SMA
    df['Price_vs_SMA'] = (df['Close'] - df['SMA_50']) / df['SMA_50']

    # 4. MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()

    # 5. BB Width
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

    # 6. Volume Shock (Vol / SMA_Vol_20)
    vol_sma = df['Volume'].rolling(window=20).mean()
    df['Volume_Shock'] = (df['Volume'] - vol_sma) / vol_sma

    # 7. Distance From 52-W High (Approx using 50 period max for hourly)
    high_max = df['High'].rolling(window=50).max()
    df['Dist_From_High'] = (high_max - df['Close']) / high_max

    # 8. Day of Week
    df['Day_of_Week'] = df['Date'].dt.dayofweek

    # 9. VWAP (Approx)
    df['VWAP'] = (df['High'] + df['Low'] + df['Close']) / 3

    # 10. Log Return (Input Feature)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))

    # Add Ticker Name
    df['Ticker'] = ticker

    # Drop NaNs created by indicators
    return df.dropna().tail(15)

# --- 5. MAIN EXECUTION ---
token_map = get_angel_tokens()
api = login()
final_data = []

if api and token_map:
    print("\n Fetching & Processing Price Data...")

    for ticker_name, token in token_map.items():
        print(f"   {ticker_name} (Token: {token})...", end="")

        try:
            # --- ADD THIS LINE HERE ---
            time.sleep(1.5) # A 1.5-second pause to be extra safe with Rate Limits
            # Fetch 15 days of history
            to_date = datetime.now()
            from_date = to_date - timedelta(days=15) # # Changed to 15 Days to bypass weekends for SMA_50

            params = {
                "exchange": "NSE",
                "symboltoken": token,
                "interval": "ONE_HOUR", # Hourly Data
                "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
                "todate": to_date.strftime("%Y-%m-%d %H:%M")
            }

            hist_data = api.getCandleData(params)

            if hist_data['status'] and hist_data['data']:
                df = pd.DataFrame(hist_data['data'], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.astype({'Open':'float', 'High':'float', 'Low':'float', 'Close':'float', 'Volume':'float'})

                # Engineer Features
                processed_df = calculate_indicators(df, ticker_name)

                # Select only required columns
                cols = ['Date', 'Ticker', 'Close', 'Volume', 'Log_Ret', 'RSI', 'SMA_50',
                        'Price_vs_SMA', 'MACD', 'BB_Width', 'Volume_Shock',
                        'Dist_From_High', 'Day_of_Week', 'VWAP']

                final_data.append(processed_df[cols])
                print("  Done")
            else:
                # Specific error handling
                if hist_data.get('errorCode') == 'AG8004':
                    print("  Invalid API Key (Activation Pending)")
                else:
                    print(f"  No Data (Msg: {hist_data.get('message')})")

        except Exception as e:
            print(f"  Error: {e}")

    # Save Combined File
    if final_data:
        full_df = pd.concat(final_data)
        full_df.to_csv('live_price_data_hourly.csv', index=False)
        print(f"\n Price Data Saved: live_price_data_hourly.csv ({len(full_df)} rows)")
    else:
        print("\n No data fetched.")
