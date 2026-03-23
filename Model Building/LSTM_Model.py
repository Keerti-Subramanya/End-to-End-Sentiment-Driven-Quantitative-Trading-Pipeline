import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# --- Configuration ---
DATA_FILE = 'final_model_dataset.csv'
WINDOW_SIZE = 10   # Look back 10 hours
EPOCHS = 50        # Training loops
BATCH_SIZE = 16

def load_and_process_data(file_path, ticker_symbol):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Error: {file_path} not found.")

    print(f"Loading data for {ticker_symbol}...")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # --- 1. Filter First ---
    df = df[df['Ticker'] == ticker_symbol].copy()
    if df.empty:
        raise ValueError(f"❌ Error: No data found for ticker {ticker_symbol}")
    df = df.sort_values('Date')

    '''
    Why Log Returns instead of Raw Price? Raw stock prices are "non-stationary" (they drift to infinity over time). 
    LSTMs fail spectacularly on non-stationary data. By converting price to Log Returns, the data becomes stationary (it mathematically oscillates around zero). 
    This allows the LSTM to learn actual momentum patterns rather than just memorizing absolute price points.'''
    # --- 2. Feature Selection & Stationary Engineering ---
    # A. Calculate Input Log Returns (The "Price History") <-- VITAL ADDITION
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    # Stationary Target: Next Day Log Return
    df['Target_Log_Ret'] = np.log(df['Target_Next_Close'] / df['Close'])
    
    # Exogenous Features requested:
    
    feature_cols = ['Log_Ret','RSI', 'Volume','MACD', 'SMA_50', 'Score_Macro', 'Score_Policy', 
                    'Score_Corporate', 'Score_Financials', 'Score_Sentiment']
    
    # Clean up any NaNs from the target calculation or features
    df = df.dropna(subset=feature_cols + ['Target_Log_Ret']).reset_index(drop=True)

    print(f"Features used ({len(feature_cols)}): {feature_cols}")
    '''Neural networks are highly sensitive to data scale. If Volume is in the millions and Sentiment is between -1 and 1, 
    the network will incorrectly assume Volume is millions of times more important. Scaling normalizes the playing field.'''
    # --- 3. Scaling (-1 to 1) ---
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaled_features = scaler_x.fit_transform(df[feature_cols])

    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    scaled_target = scaler_y.fit_transform(df[['Target_Log_Ret']])

    # Save ticker-specific scalers
    joblib.dump(scaler_x, f'scaler_x_{ticker_symbol}.pkl')
    joblib.dump(scaler_y, f'scaler_y_{ticker_symbol}.pkl')
    print(f"💾 Scalers saved for {ticker_symbol}.")

    '''Saving the model weights is useless without saving the preprocessing state. 
     I exported the ticker-specific MinMaxScalers as .pkl artifacts to guarantee Inference Consistency in production.
      When the live pipeline fetches fresh data tomorrow,
      it must load these exact historical scalers to transform the new inputs into the precise mathematical boundaries the neural network was trained on. 
      Furthermore, I need the Y-scaler saved in order to inverse-transform the AI's abstract numerical output back into real-world currency for the trading logic.'''
    return df, scaled_features, scaled_target, scaler_y, feature_cols

def create_sequences(df, scaled_features, scaled_target, window_size):
    X, y, actual_prices_prev = [], [], []
    
    # Input is already filtered for one ticker
    close_prices = df['Close'].values 

    for i in range(len(df) - window_size):
        X.append(scaled_features[i : i + window_size])
        y.append(scaled_target[i + window_size])
        # Reference: The Close price at the end of the window to reconstruct price
        actual_prices_prev.append(close_prices[i + window_size - 1])

    return np.array(X), np.array(y), np.array(actual_prices_prev)

def build_regression_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model

def evaluate_performance(y_true_scaled, y_pred_scaled, prev_prices, scaler_y, dataset_name="Test"):
    print(f"\n--- {dataset_name} Performance Evaluation ---")
    
    # 1. Inverse Transform Log Returns
    y_true_log_ret = scaler_y.inverse_transform(y_true_scaled).flatten()
    y_pred_log_ret = scaler_y.inverse_transform(y_pred_scaled).flatten()

    # 2. Reconstruct Prices: Predicted_Price = Last_Close * exp(Log_Ret)
    y_true_price = prev_prices * np.exp(y_true_log_ret)
    y_pred_price = prev_prices * np.exp(y_pred_log_ret)

    # 3. Regression Metrics on Reconstructed Prices (Rupees)
    rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    mae = mean_absolute_error(y_true_price, y_pred_price)
    r2 = r2_score(y_true_price, y_pred_price)
    
    print(f"📉 Price Error (RMSE): ₹{rmse:.2f}")
    print(f"📉 Mean Abs Error (MAE): ₹{mae:.2f}")
    print(f"📊 R² Score: {r2:.4f}")

    # 4. Directional Accuracy
    pred_direction = (y_pred_price > prev_prices).astype(int)
    true_direction = (y_true_price > prev_prices).astype(int)
    
    acc = accuracy_score(true_direction, pred_direction)
    print(f"✅ Directional Accuracy: {acc:.2%}")
    
    return y_true_price, y_pred_price, r2, rmse

def main():
    print("🚀 Starting Automated Specialist Price Prediction Loop...")

    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: {DATA_FILE} not found.")
        return

    # Identify all unique tickers
    all_data = pd.read_csv(DATA_FILE)
    unique_tickers = all_data['Ticker'].unique()
    print(f"Found {len(unique_tickers)} tickers: {list(unique_tickers)}")

    results_summary = []

    for ticker in unique_tickers:
        print("\n" + "="*50)
        print(f"🏗️  TRAINING SPECIALIST MODEL FOR: {ticker}")
        print("="*50)

        # Clear session to prevent memory leaks
        tf.keras.backend.clear_session()

        try:
            # 1. Load Data (Isolated & Sentiment-Integrated)
            df, scaled_X, scaled_y, scaler_y, feature_cols = load_and_process_data(DATA_FILE, ticker_symbol=ticker)
            
            # 2. Create Sequences
            X, y, prev_prices = create_sequences(df, scaled_X, scaled_y, WINDOW_SIZE)
            
            if len(X) < 30:
                print(f"⚠️ Skipping {ticker}: Not enough data (Sequences: {len(X)})")
                continue

            # 3. Train / Test Split
            split_idx = int(len(X) * 0.80)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            prev_prices_train, prev_prices_test = prev_prices[:split_idx], prev_prices[split_idx:]

            # 4. Build & Train Model
            model = build_regression_model((WINDOW_SIZE, len(feature_cols)))
            
            model.fit(
                X_train, y_train, 
                validation_data=(X_test, y_test),
                epochs=EPOCHS, 
                batch_size=BATCH_SIZE, 
                verbose=0 
            )
            
            # Save Ticker-Specific Model
            model_filename = f'model_{ticker}.keras'
            model.save(model_filename)
            print(f"✅ Specialist Model Saved: {model_filename}")

            # 5. Evaluate & Reconstruct Price
            train_preds = model.predict(X_train, verbose=0)
            test_preds = model.predict(X_test, verbose=0)

            _, _, r2_train, rmse_train = evaluate_performance(y_train, train_preds, prev_prices_train, scaler_y, f"TRAIN [{ticker}]")
            y_true_test, y_pred_test, r2_test, rmse_test = evaluate_performance(y_test, test_preds, prev_prices_test, scaler_y, f"TEST [{ticker}]")

            # 6. Generate Latest Trading Signal & Prediction
            last_known_close = prev_prices_test[-1]
            last_pred_price = y_pred_test[-1]
            expected_move = last_pred_price - last_known_close
            
            if expected_move > rmse_test:
                signal = "BUY 🟢"
            elif expected_move < -rmse_test:
                signal = "SELL 🔴"
            else:
                signal = "WAIT ⚪"

            # Store Results
            results_summary.append({
                'Ticker': ticker,
                'Test RMSE (₹)': round(rmse_test, 2),
                'Test R²': round(r2_test, 4),
                'Current Price': round(last_known_close, 2),
                'Predicted Price': round(last_pred_price, 2),
                'Signal': signal
            })

        except Exception as e:
            print(f"❌ Error training {ticker}: {e}")
            continue

    # Final Summary Report
    if results_summary:
        print("\n" + "📊" + " "*15 + "SPECIALIST PREDICTION SUMMARY" + " "*15 + "📊")
        summary_df = pd.DataFrame(results_summary)
        # Reorder columns as requested
        summary_df = summary_df[['Ticker', 'Test RMSE (₹)', 'Test R²', 'Current Price', 'Predicted Price', 'Signal']]
        print(summary_df.to_string(index=False))
        
        summary_df.to_csv('specialist_prediction_summary.csv', index=False)
        print("\n📂 Summary saved to 'specialist_prediction_summary.csv'")
    else:
        print("\n❌ No models were successfully trained.")

if __name__ == "__main__":
    main()

    '''That is a deliberate outcome of the strict risk-management engine I built into the final pipeline. 
    The algorithm is programmed to act like an ultra-conservative institutional fund. 
    It calculates the expected profit of the next hour, but it also knows its own historical margin of error (the RMSE). 
    Right now, the model is seeing that the background noise (Volatility) of the hourly market is larger than the expected trend (Drift).
      Because the math doesn't offer a safe risk-reward ratio, 
    the AI is intelligently choosing to preserve capital by staying in Cash (WAIT).'''
