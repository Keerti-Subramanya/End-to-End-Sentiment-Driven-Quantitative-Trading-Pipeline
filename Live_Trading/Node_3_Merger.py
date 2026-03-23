import pandas as pd
import numpy as np
import ta
import os
import glob

# --- CONFIGURATION ---
PRICE_FILE = 'live_price_data_hourly.csv'
NEWS_FILE = r'Downloads\live_news_factors*.csv'
OUTPUT_FILE = 'ready_for_prediction.csv'
DECAY_RATE = 0.85


# THESE MUST MATCH YOUR TRAINING LIST EXACTLY
FEATURE_COLS = ['Log_Ret','RSI', 'Volume','MACD', 'SMA_50', 'Score_Macro', 
                'Score_Policy', 'Score_Corporate', 'Score_Financials', 'Score_Sentiment']

def apply_decay(df):
    if 'Day' not in df.columns:
        df['Day'] = pd.to_datetime(df['Date']).dt.date.astype(str)
    
    df['Hour_Rank'] = df.groupby(['Ticker', 'Day']).cumcount()
    score_cols = ['Score_Macro', 'Score_Policy', 'Score_Corporate', 'Score_Financials', 'Score_Sentiment']

    for col in score_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)
        df[col] = df[col] * (DECAY_RATE ** df['Hour_Rank'])
    return df

def main():
    print(" Preparing Final Prediction Dataset...")
    # Find all files matching the name
    list_of_files = glob.glob(NEWS_FILE)

    if not list_of_files:
        raise FileNotFoundError(f" CRITICAL ERROR: Could not find any news files in your Downloads folder!")

    # 🛠️ THE FIX: Use a new variable name 'actual_news_file' so Python doesn't crash
    actual_news_file = max(list_of_files, key=os.path.getmtime)
    print(f" Automatically using the newest news file: {actual_news_file}")

    if not os.path.exists(PRICE_FILE):
        print(f" Error: {PRICE_FILE} missing.")
        return
    
    df_price = pd.read_csv(PRICE_FILE)
    df_price['Date'] = pd.to_datetime(df_price['Date']).dt.tz_localize(None)
    
    if os.path.exists(actual_news_file):
        df_news = pd.read_csv(actual_news_file)
        df_news['Date'] = pd.to_datetime(df_news['Date']).dt.date.astype(str)
    else:
        raise FileNotFoundError(f" CRITICAL ERROR: Could not find news file at {NEWS_FILE}. Please check your Google Drive sync!")
        #df_news = pd.DataFrame(columns=['Date', 'Ticker', 'Headline', 'Key_Insight'] + FEATURE_COLS[5:])

    final_dfs = []
    for ticker in df_price['Ticker'].unique():
        t_df = df_price[df_price['Ticker'] == ticker].sort_values('Date').copy()
        t_df['Day'] = t_df['Date'].dt.date.astype(str)

        # Merge News
        t_df = pd.merge(t_df, df_news[df_news['Ticker'] == ticker],
                        left_on='Day', right_on='Date', how='left', suffixes=('', '_news'))
        
        # 🛠️ THE FIX: Copy today's fetched news upwards to cover the older hours
        if 'Headline' in t_df.columns:
            t_df['Headline'] = t_df['Headline'].bfill().ffill()
        if 'Key_Insight' in t_df.columns:
            t_df['Key_Insight'] = t_df['Key_Insight'].bfill().ffill()
        
        # Apply Decay
        t_df = apply_decay(t_df)

        # Keep last 15 rows (Window of 10 + buffer)
        final_dfs.append(t_df.tail(15))

    if final_dfs:
        full_df = pd.concat(final_dfs)
        
        # We keep Date, Ticker, and Close for the Dashboard display
        # and the FEATURE_COLS for the Model
        display_cols = ['Date', 'Ticker', 'Close', 'Headline', 'Key_Insight']
        save_cols = display_cols + [c for c in FEATURE_COLS if c not in display_cols]
        
        full_df = full_df[save_cols]
        full_df.to_csv(OUTPUT_FILE, index=False)
        print(f" Success! ready_for_prediction.csv is ready with {len(full_df)} rows.")
    else:
        print(" No data processed.")

if __name__ == "__main__":
    main()
