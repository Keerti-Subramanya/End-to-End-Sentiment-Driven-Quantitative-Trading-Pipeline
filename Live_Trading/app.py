
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import os
import google.generativeai as genai

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Trade Specialist", layout="wide", page_icon="⚡")

# --- CONFIGURATION ---
INPUT_FILE = 'ready_for_prediction.csv'
JOURNAL_FILE = 'trading_journey.csv'  # <--- NEW: Journal File Name
WINDOW_SIZE = 10
FEATURE_COLS = ['Log_Ret', 'RSI', 'Volume', 'MACD', 'SMA_50', 
                'Score_Macro', 'Score_Policy', 'Score_Corporate', 
                'Score_Financials', 'Score_Sentiment']

# --- GEMINI SETUP ---
# Replace 'YOUR_GEMINI_API_KEY' with your actual key from Google AI Studio
genai.configure(api_key="your gemini api key")
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

def get_gemini_response(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

# --- LOAD DATA ---
@st.cache_data
def load_data():
    if not os.path.exists(INPUT_FILE):
        return pd.DataFrame()
    return pd.read_csv(INPUT_FILE)

# --- JOURNALING FUNCTION (NEW) ---
def log_to_journal(date, ticker, close, pred, signal, sentiment, headline):
    """
    Saves the prediction to a CSV file.
    Checks for duplicates so we don't save the same hour twice.
    """
    new_entry = {
        'Date': date,
        'Ticker': ticker,
        'Close_Price': close,
        'AI_Predicted_Price': pred,
        'Signal': signal,
        'Sentiment_Score': sentiment,
        'News_Headline': headline
    }
    
    if os.path.exists(JOURNAL_FILE):
        df_log = pd.read_csv(JOURNAL_FILE)
        # Check if this specific Ticker + Date combination already exists
        # We convert dates to string to ensure matching works
        is_exist = df_log[(df_log['Date'] == str(date)) & (df_log['Ticker'] == ticker)]
        
        if is_exist.empty:
            pd.DataFrame([new_entry]).to_csv(JOURNAL_FILE, mode='a', header=False, index=False)
            return True # Saved successfully
        else:
            return False # Already existed
    else:
        pd.DataFrame([new_entry]).to_csv(JOURNAL_FILE, index=False)
        return True # Created new file

# --- PREDICTION FUNCTION ---
def make_prediction(ticker, df_ticker):
    """
    Loads model/scaler for the ticker and predicts the next price.
    """
    try:
        # 1. Load Assets
        # Assuming files are in the root directory
        scaler_x = joblib.load(f'scaler_x_{ticker}.pkl')
        scaler_y = joblib.load(f'scaler_y_{ticker}.pkl')
        model = tf.keras.models.load_model(f'model_{ticker}.keras')
        
        # 2. Prepare Sequence (Last 10 rows)
        if len(df_ticker) < WINDOW_SIZE:
            return None, "Not enough data"
            
        input_data = df_ticker.tail(WINDOW_SIZE)[FEATURE_COLS]
        
        # 3. Scale & Predict
        scaled_seq = scaler_x.transform(input_data)
        X_input = scaled_seq.reshape(1, WINDOW_SIZE, len(FEATURE_COLS))
        
        pred_scaled = model.predict(X_input, verbose=0)
        pred_log_ret = scaler_y.inverse_transform(pred_scaled)[0][0]
        
        # 4. Convert to Price
        last_close = df_ticker.iloc[-1]['Close']
        predicted_price = last_close * np.exp(pred_log_ret)
        
        return predicted_price, "Success"
        
    except Exception as e:
        return None, str(e)

# --- MAIN APP ---
st.title("⚡ AI Algorithmic Trading Dashboard")
st.markdown(f"### 🔴 Live Prediction & RAG Analysis")
st.divider()

df = load_data()

if df.empty:
    st.error(f"❌ File '{INPUT_FILE}' not found. Please run the 'Data Merger' script first.")
else:
    # --- SIDEBAR SELECTOR ---
    tickers = df['Ticker'].unique()
    selected_ticker = st.sidebar.selectbox("Select Asset", tickers)
    
    # Get Data for Ticker
    df_t = df[df['Ticker'] == selected_ticker].sort_values('Date')
    current_row = df_t.iloc[-1]
    
    # --- RUN PREDICTION ---
    with st.spinner(f"🔮 AI Predicting {selected_ticker}..."):
        pred_price, status = make_prediction(selected_ticker, df_t)
    
    if pred_price is None:
        st.error(f"Prediction Failed: {status}")
        st.warning("Ensure 'scaler_x_....pkl' and 'model_....keras' files are in the folder.")
    else:
        # Calculate Signal
        last_price = current_row['Close']
        move_pct = (pred_price - last_price) / last_price
        
        if move_pct > 0.002: signal = "BUY 🟢"
        elif move_pct < -0.002: signal = "SELL 🔴"
        else: signal = "WAIT ⚪"

        # --- SAVE TO JOURNAL (Auto-Save Logic) ---
        # This saves the prediction automatically every time it is generated
        saved = log_to_journal(
            date=current_row['Date'],
            ticker=selected_ticker,
            close=last_price,
            pred=pred_price,
            signal=signal,
            sentiment=current_row['Score_Sentiment'],
            headline=current_row.get('Headline', 'No News')
        )
        
        if saved:
            st.toast(f"✅ Logged {selected_ticker} to Trading Journey!", icon="📝")

        # --- DISPLAY DASHBOARD ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"₹{last_price:,.2f}")
        col2.metric("AI Target (1h)", f"₹{pred_price:,.2f}", f"{move_pct*100:.2f}%")
        col3.metric("Signal", signal)
        col4.metric("Sentiment Score", f"{current_row['Score_Sentiment']:.4f}")
        
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = pred_price,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "AI Confidence"},
            delta = {'reference': last_price},
            gauge = {
                'axis': {'range': [last_price*0.98, last_price*1.02]},
                'bar': {'color': "blue"},
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': pred_price}
            }))
        st.plotly_chart(fig, use_container_width=True)
        
        # --- RAG INTEGRATION SECTION ---
        st.divider()
        st.subheader("🤖 AI Executive Analysis (RAG)")

        # 1. Prepare the Data Context for the AI
        rag_context = f"""
        Asset: {selected_ticker}
        Current Price: ₹{last_price:,.2f}
        AI Predicted Price (1h): ₹{pred_price:,.2f}
        Signal: {signal} ({move_pct*100:.2f}%)
        News Headline: {current_row.get('Headline', 'No News available')}
        Extracted Insight: {current_row.get('Key_Insight', 'No Insight available')}
        Sentiment Score: {current_row['Score_Sentiment']:.4f}
        Technical RSI: {current_row.get('RSI', 'N/A')}
        """

        # 2. AUTO-VERDICT (Talk-First RAG)
        st.markdown("#### 📝 Automated Market Verdict")

        # We create a specific prompt for the "First Talk" summary
        summary_prompt = f"""
        You are a senior stock market analyst. Based on the following real-time context, 
        give a concise 3-sentence executive verdict. 
        Explain why the signal is {signal} and how the news impacts the price.

        CONTEXT:
        {rag_context}
        """

        with st.spinner("AI is generating your morning verdict..."):
            # This makes the AI "talk first" automatically
            executive_summary = get_gemini_response(summary_prompt)
            st.write(executive_summary)

        # 3. INTERACTIVE CHAT (Follow-up)
        st.divider()
        col_chat, col_raw = st.columns([2, 1])
        
        with col_chat:
            st.subheader("💬 Ask the Analyst")
            user_q = st.text_input("Ask about technicals, news impact, or risk...")
            
            if user_q:
                chat_prompt = f"""
                CONTEXT:
                {rag_context}
                
                USER QUESTION: {user_q}
                
                Answer professionally based ONLY on the provided context. 
                If you don't know, say it's not in the data.
                """
                with st.spinner("Analyzing..."):
                    response = get_gemini_response(chat_prompt)
                    st.markdown(f"**AI:** {response}")

        with col_raw:
            with st.expander("View Raw RAG Context"):
                st.code(rag_context)

        # --- VIEW HISTORY SECTION (NEW) ---
        st.divider()
        st.subheader("📜 Trading Journal History")
        if os.path.exists(JOURNAL_FILE):
            st.dataframe(pd.read_csv(JOURNAL_FILE).tail(10))
