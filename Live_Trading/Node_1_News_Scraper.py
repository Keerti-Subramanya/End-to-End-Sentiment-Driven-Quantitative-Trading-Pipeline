# 1. Install using the NEW package name (ddgs) safely
!pip install ddgs newspaper3k transformers torch nltk lxml_html_clean -q

# 2. THE SAFE WARNING SUPPRESSOR
import warnings
import logging
import os
import sys

# Silence standard warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger().setLevel(logging.ERROR)

if 'ipykernel' in sys.modules:
    import IPython
    IPython.Application.instance().log.setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="jupyter_client")

import pytz
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from newspaper import Article
import nltk
from google.colab import drive
from ddgs import DDGS
from google.colab import files

# Mount Drive
drive.mount('/content/drive')

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# --- CONFIGURATION ---
DAYS_TO_FETCH = 0
OUTPUT_FILE = '/content/drive/MyDrive/Trading_Bot/live_news_factors.csv'

output_dir = os.path.dirname(OUTPUT_FILE)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

STOCK_MAP = {
    'RELIANCE.NS': 'Reliance Industries', 'TCS.NS': 'Tata Consultancy Services',
    'HDFCBANK.NS': 'HDFC Bank', 'TATAMOTORS.NS': 'Tata Motors',
    'TATASTEEL.NS': 'Tata Steel', 'SUNPHARMA.NS': 'Sun Pharma',
    'ITC.NS': 'ITC Ltd', 'ADANIPORTS.NS': 'Adani Ports',
    'SBIN.NS': 'State Bank of India', 'LT.NS': 'Larsen & Toubro'
}

FACTORS = {
    'Macro': ['Global Dollar', 'Foreign Investment', 'Inflation', 'GDP', 'Interest Rate', 'Economy'],
    'Policy': ['Tariffs', 'Budget', 'Tax', 'Regulation', 'Government', 'RBI'],
    'Corporate': ['Merger', 'Acquisition', 'Product', 'Layoff', 'Hiring', 'CEO', 'Results', 'Q1', 'Q2', 'Q3', 'Q4'],
    'Financials': ['Revenue', 'Profit', 'Earnings', 'Dividend', 'Cash', 'Loss', 'Margin'],
    'Sentiment': ['Up', 'Down', 'Bullish', 'Bearish', 'Optimistic', 'Pessimistic', 'Strong', 'Weak']
}

JUNK_PHRASES = ["login", "subscribe", "register", "sign in", "cookies", "advertisement", "enable javascript", "access", "premium", "paywall"]

# --- AI SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 AI Device: {device}")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

# --- HELPER FUNCTIONS ---
# def get_sentiment(text):
#     if not text: return 0.0
#     sentences = nltk.sent_tokenize(text)[:10]
#     if not sentences: return 0.0
#     inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     scores = (probs[:, 0] - probs[:, 1]).cpu().numpy()
#     return np.mean(scores)
def get_deep_sentiment(text):
    if not text:
        return None, 0.0

    # Process the entire article
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return None, 0.0

    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # FinBERT labels: 0: positive, 1: negative, 2: neutral. Score = pos - neg
    scores = (probs[:, 0] - probs[:, 1]).cpu().numpy()

    # Find the single sentence with the highest absolute sentiment score
    abs_scores = np.abs(scores)
    idx = np.argmax(abs_scores)

    return sentences[idx], float(scores[idx])

def is_junk(text):
    return any(junk in text.lower() for junk in JUNK_PHRASES)

def get_best_article_for_ticker(ddgs_session, company, date_str):
    """
    Passes the persistent DDGS session to avoid opening new connections.
    """
    query = f"{company} stock market news India"
    print(f"   [Debug] Searching DDG for {company}...")

    try:
        # LEVEL 2 Execution: Uses the persistent session (Headers are handled automatically by the library now)
        results = ddgs_session.text(query, max_results=5) # Switched to .text() which is more stable than .news() in newest versions

        # LEVEL 3 Detection: Soft Block triggered if results are empty
        if not results:
            print(f"      [🚨 WARNING] Empty results for {company}. Possible Soft-Block!")
            return "Search Failed", 0.0, "Error", True

    except Exception as e:
        print(f"      [🚨 Error] DDG Search Failed: {e}")
        return "Search Failed", 0.0, "Error", True

    best_insight, best_score, best_headline = None, 0.0, ""

    for res in results:
        url = res.get('href', '')
        headline = res.get('title', '')

        # 🛠️ THE FIX: Extract the domain name from the URL to use as the source
        source = url.split('/')[2].replace('www.', '') if '//' in url else 'Unknown Source'
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text

            # if len(text) > 500 and not is_junk(text):
            #     score = get_sentiment(text)
            #     print(f"      ✅ Scraped {source} (Score: {score:.2f}): {headline[:30]}...")
            #     return text[:500], score, headline, False
            # else:
            #     print(f"      ⏭️ Rejected {source} (Vague/Paywall)")
            if len(text) > 500 and not is_junk(text):
                best_sentence, score = get_deep_sentiment(text)
                if best_sentence:
                    print(f"      ✅ Scraped {source} (Score: {score:.2f}): {headline[:30]}...")
                    # Returns the most impactful sentence instead of the first 500 chars
                    return best_sentence, score, headline, False
            else:
                print(f"      ⏭️ Rejected {source} (Vague/Paywall)")
        except:
            print(f"      ❌ Blocked by {source}: '{headline[:20]}...'. Trying next...")
            continue

    # if not best_insight and results:
    #     print(f"      ⚠️ All full articles blocked. Using DDG snippets for {company}")
    #     import re
    #     combined_text = re.sub(r'<[^>]+>', '', ". ".join([f"{r.get('title', '')} {r.get('body', '')}" for r in results[:3]]))
    #     return combined_text[:500], get_sentiment(combined_text), results[0].get('title', 'Global Market Summary'), False
    if not best_insight and results:
        print(f"      ⚠️ All full articles blocked. Using DDG snippets for {company}")
        import re
        combined_text = re.sub(r'<[^>]+>', '', ". ".join([f"{r.get('title', '')} {r.get('body', '')}" for r in results[:3]]))

        best_sentence, score = get_deep_sentiment(combined_text)
        fallback_insight = best_sentence if best_sentence else combined_text[:500]

        return fallback_insight, score, results[0].get('title', 'Global Market Summary'), False

    return best_insight, best_score, best_headline, False

# --- MAIN INFINITE LOOP ---
print(f"\n🚀 Starting Stealth News Fetcher (Loop Mode)...")
IST = pytz.timezone('Asia/Kolkata')

while True:
    current_time = datetime.now(IST).strftime('%H:%M')
    print(f"\n🔄 Running Update at {current_time}(IST)...")

    start_date = datetime.now() - timedelta(days=DAYS_TO_FETCH)
    current_date = start_date
    all_data = []

    print(f"🕵️ Initializing persistent browser session...")

    # LEVEL 2 Stickiness: Open DDGS as a context manager to reuse the connection
    # (The library automatically sets optimal, human-like headers internally)
    with DDGS() as ddgs_session:

        while current_date.date() <= datetime.now().date():
            date_str = current_date.strftime('%Y-%m-%d')
            print(f"\n📅 {date_str}")

            for ticker, company in STOCK_MAP.items():
                print(f"\n   🔍 Hunting news for {ticker}...", flush=True)

                insight, score, headline, is_blocked = get_best_article_for_ticker(ddgs_session, company, date_str)

                # LEVEL 3 Action: 30-Minute Exponential Backoff if flagged
                if is_blocked:
                    print("   🛑 IP Flagged. Forcing a 30-minute cool-down sleep to reset limits...")
                    time.sleep(1800)
                    continue # Skip saving bad data for this ticker

                row = {
                    'Date': date_str,
                    'Ticker': ticker,
                    'AI_Sentiment': score if insight else 0.0,
                    'Key_Insight': insight if insight else "No significant news.",
                    'Headline': headline
                }

                text_for_factors = (str(insight) + " " + str(headline)).lower()
                for cat, keywords in FACTORS.items():
                    row[f'Score_{cat}'] = score if any(k.lower() in text_for_factors for k in keywords) else 0.0

                all_data.append(row)
                print(f" Done.")

                # Randomized pause to avoid heartbeat detection
                time.sleep(random.uniform(10.5, 25.2))

            current_date += timedelta(days=1)

    # Save to Drive (Appending/Updating)
    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        updated_df = pd.concat([existing_df, pd.DataFrame(all_data)]).drop_duplicates(subset=['Date', 'Ticker'], keep='last')
        updated_df.to_csv(OUTPUT_FILE, index=False)
    else:
        pd.DataFrame(all_data).to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Saved to {OUTPUT_FILE}")

    # print("💤 Sleeping for 60 minutes...")
    # time.sleep(3600)
    # --- THE AUTO-DOWNLOAD COMMAND ---
    print(f"📥 Automatically triggering download to your local machine...")
    files.download(OUTPUT_FILE)

    # --- CRITICAL FIX: Break the loop so the download successfully triggers ---
    print(f"🏁 Run complete. Breaking loop to allow browser download.")
    break
