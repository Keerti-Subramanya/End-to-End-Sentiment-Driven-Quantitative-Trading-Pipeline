#The primary business objective of this code is Feature Engineering and Dataset Unification for Algorithmic Trading.
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import timedelta
import os
import shutil
import time

# --- Configuration ---
NEWS_FILE = 'news_factors_90d_deep_latest.csv'
OUTPUT_FILE = 'final_model_dataset.csv'
DECAY_FACTOR = 0.85  # News impact fades by 15% every hour

'''
Initializes a critical business constant. It dictates that the market impact of a news article decreases by 15% every passing hour.
the decay hour allows Quantitative Analysts (Quants) to easily tweak business rules without digging through complex Python loops.
'''

# --- 1. Load & Preprocess News ---
def load_and_preprocess_news(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: {file_path} not found.")
    
    print("📰 Loading News Data...")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Weekend Roller: Move Sat/Sun news to Monday
    def weekend_roller(dt):
        if dt.weekday() == 5: return dt + timedelta(days=2)
        if dt.weekday() == 6: return dt + timedelta(days=1)
        return dt

    df['Date'] = df['Date'].apply(weekend_roller)
    
    # Aggregate (Mean score for the day)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_agg = df.groupby(['Date', 'Ticker'])[numeric_cols].mean().reset_index()
    
    return df_agg

# --- 2. Fetch Hourly Stock Data (With 2026 Ticker Support) ---
def fetch_hourly_stock_data(tickers, start_date):
    # Clear Cache
    if os.path.exists('yfinance_cache'):
        shutil.rmtree('yfinance_cache', ignore_errors=True)

    # Fetch extra buffer for indicators
    fetch_start = start_date - timedelta(days=60)
    print(f"\n📥 Fetching Hourly Data (1h) starting from {fetch_start.date()}...")
    
    # UPDATED ALIASES FOR 2026
    aliases = {
        'TATAMOTORS.NS': ['TMCV.NS', 'TMPV.NS', 'TATAMOTORS.BO'], # New demerged tickers
        'TCS.NS': ['TCS.NS', 'TCS.BO'],
        'TATASTEEL.NS': ['TATASTEEL.NS', 'TATASTEEL.BO'],
        'RELIANCE.NS': ['RELIANCE.NS', 'RELIANCE.BO'],
        'SBIN.NS': ['SBIN.NS', 'SBIN.BO'],
        'HDFCBANK.NS': ['HDFCBANK.NS', 'HDFCBANK.BO'],
        'ITC.NS': ['ITC.NS', 'ITC.BO']
    }

    all_history = []
    
    for ticker in tickers:
        # Get list of candidates (Original ticker + Aliases)
        candidates = aliases.get(ticker, [ticker])
        # Add .BO fallback if not explicitly listed
        if ticker.endswith('.NS') and ticker not in aliases:
            candidates.append(ticker.replace('.NS', '.BO'))

        success = False
        
        for candidate in candidates:
            print(f"   Attempting {candidate} (for {ticker})...")
            
            for attempt in range(3): # 3 Retries
                try:
                    df = yf.download(candidate, start=fetch_start, interval='1h', progress=False)
                    
                    if not df.empty:
                        # Clean Index/Columns
                        df = df.reset_index()
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = [c[0] if c[0] in ['Datetime', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'] else c[0] for c in df.columns]
                        
                        if 'Datetime' in df.columns:
                            df.rename(columns={'Datetime': 'Date'}, inplace=True)
                        
                        # Remove Timezone because Timezones cause fatal errors during Pandas merge operations.
                        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                        df['Ticker'] = ticker # Save as ORIGINAL ticker to match news file
                        
                        all_history.append(df)
                        print(f"      ✅ Success with {candidate}")
                        success = True
                        break 
                    
                except Exception as e:
                    print(f"      ⚠️ Attempt {attempt+1} failed: {e}")
                    time.sleep(1)
            
            if success:
                break 
        
        if not success:
            print(f"   ❌ FAILED to download {ticker} after all attempts.")

    if not all_history:
        return pd.DataFrame()

    return pd.concat(all_history, ignore_index=True)

# --- 3. Merge with Decay Logic ---
def merge_with_decay(price_df, news_df):
    print("\n🧠 Processing News Decay Logic...")
    
    price_df['Join_Date'] = price_df['Date'].dt.normalize()
    
    # Left Join
    merged_df = pd.merge(price_df, news_df, left_on=['Join_Date', 'Ticker'], right_on=['Date', 'Ticker'], how='left', suffixes=('', '_News'))
    
    score_cols = [c for c in news_df.columns if 'Score' in c]
    merged_df[score_cols] = merged_df[score_cols].fillna(0)
    
    # Decay Calculation
    merged_df['Hour_Rank'] = merged_df.groupby(['Ticker', 'Join_Date']).cumcount()
    
    print("   Applying exponential decay to news scores...")
    for col in score_cols:
        merged_df[col] = merged_df[col] * (DECAY_FACTOR ** merged_df['Hour_Rank'])
    
    merged_df.drop(columns=['Join_Date', 'Date_News', 'Hour_Rank'], errors='ignore', inplace=True)
    return merged_df

# --- 4. Apply Indicators (STABLE PANDAS VERSION) ---
'''In an MLOps environment (like Docker or Airflow), 
installing ta-lib often requires compiling C++ binaries which regularly causes build failures'''
def apply_hourly_indicators(df):
    print("\n📈 Calculating Indicators (Stable Mode)...")
    df = df.sort_values(['Ticker', 'Date'])
    
    final_dfs = []
    
    for ticker, group in df.groupby('Ticker'):
        group = group.copy()
        
        # 1. SMA 50
        group['SMA_50'] = group['Close'].rolling(window=50).mean()
        group['Price_vs_SMA'] = group['Close'] - group['SMA_50']
        
        # 2. RSI (14) - Pure Pandas
        delta = group['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        group['RSI'] = 100 - (100 / (1 + rs))

        '''RSI quantifies market velocity.
          It tells the trading algorithm if an asset is mathematically "Overbought" (typically > 70) or "Oversold" (typically < 30), 
          acting as a primary trigger for mean-reversion trading strategies'''
        
        # 3. MACD (12, 26, 9)
        exp12 = group['Close'].ewm(span=12, adjust=False).mean()
        exp26 = group['Close'].ewm(span=26, adjust=False).mean()
        group['MACD'] = exp12 - exp26

        '''MACD is the industry standard for identifying accelerating or decelerating momentum. 
        If the fast 12-period line aggressively crosses above the slow 26-period line, 
        it signals to the algorithm that immediate short-term buying pressure is heavily overwhelming long-term trends.'''

        # 4. Bollinger Bands (20, 2)
        bb_mean = group['Close'].rolling(window=20).mean()
        bb_std = group['Close'].rolling(window=20).std()
        
        # Only keep BB_Width (Relative)
        bb_upper = bb_mean + (2 * bb_std)
        bb_lower = bb_mean - (2 * bb_std)
        group['BB_Width'] = (bb_upper - bb_lower) / bb_mean

        # 5. Volume Shock
        vol_sma = group['Volume'].rolling(window=20).mean()
        group['Volume_Shock'] = group['Volume'] / vol_sma

        # 6. Dist_From_High (Rolling 350 hours ~ 2 months)
        high_rolling = group['Close'].rolling(window=350, min_periods=1).max()
        group['Dist_From_High'] = (high_rolling - group['Close']) / high_rolling

        # 7. Day of Week
        group['Day_of_Week'] = group['Date'].dt.dayofweek
        
        # 8. VWAP (Intraday)
        group['PV'] = group['Close'] * group['Volume']
        group['Cum_Vol'] = group.groupby(group['Date'].dt.date)['Volume'].cumsum()
        group['Cum_PV'] = group.groupby(group['Date'].dt.date)['PV'].cumsum()
        group['VWAP'] = group['Cum_PV'] / group['Cum_Vol']

        '''The primary business objective of this code is 
          Institutional Benchmark Generation and Intraday Trend Identification.
          If the current market price is trading above this calculated VWAP, 
          the algorithm recognizes that the buyers are aggressively paying premium prices on high volume,
          indicating strong intraday bullish momentum.'''


        # Fill NaNs
        group = group.bfill().fillna(0)
        
        # --- Targets ---
        group['Target_Next_Close'] = group['Close'].shift(-1)
        group['Target_Direction'] = (group['Target_Next_Close'] > group['Close']).astype(int)
        
        # Drop last row
        group = group.dropna(subset=['Target_Next_Close'])
        
        # Clean temp cols
        group.drop(columns=['PV', 'Cum_Vol', 'Cum_PV'], errors='ignore', inplace=True)
        
        final_dfs.append(group)

    return pd.concat(final_dfs, ignore_index=True)

# --- Main Execution ---
def main():
    try:
        # 1. Load News
        df_news = load_and_preprocess_news(NEWS_FILE)
        tickers = df_news['Ticker'].unique().tolist()
        min_date = df_news['Date'].min()
        
        # 2. Fetch Hourly Price (Robust Mode)
        df_price = fetch_hourly_stock_data(tickers, min_date)
        if df_price.empty:
            print("❌ Error: No price data found.")
            return

        # 3. Merge with Decay
        df_merged = merge_with_decay(df_price, df_news)
        
        # 4. Indicators
        df_final = apply_hourly_indicators(df_merged)
        
        # 5. Filter Columns (Keep Necessary + News)
        necessary_cols = [
            'Date', 'Ticker', 'Close', 'Volume', # Core
            'RSI', 'SMA_50', 'Price_vs_SMA', 'MACD', 'BB_Width', 'Volume_Shock', 'Dist_From_High', 'Day_of_Week', 'VWAP', # Indicators
            'Score_Macro', 'Score_Policy', 'Score_Corporate', 'Score_Financials', 'Score_Sentiment', # News
            'Target_Next_Close', 'Target_Direction' # Targets
        ]
        
        # Ensure only existing columns are selected
        final_cols = [c for c in necessary_cols if c in df_final.columns]
        df_final = df_final[final_cols]
        
        # 6. Save
        df_final.to_csv(OUTPUT_FILE, index=False)
        
        print(f"\n✅ Success! Dataset Created: {OUTPUT_FILE}")
        print(f"Total Rows: {len(df_final)}")
        print("\nSample Data:")
        print(df_final.head())

    except Exception as e:
        print(f"❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
