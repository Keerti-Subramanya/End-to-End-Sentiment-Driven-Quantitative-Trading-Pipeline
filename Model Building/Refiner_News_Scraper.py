import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
import os
from GoogleNews import GoogleNews
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fake_useragent import UserAgent
from newspaper import Article
import nltk

# --- 1. Setup ---
try:
    from google.colab import drive
    drive.mount('/content/drive')
    FILE_PATH = '/content/drive/MyDrive/news_factors_90d_deep.csv'
except ImportError:
    FILE_PATH = 'news_factors_90d_deep.csv'

nltk.download('punkt')

# --- 2. Configuration ---
JUNK_PHRASES = [
    "login", "subscribe", "register", "sign in", "cookies",
    "advertisement", "enable javascript", "access", "member",
    "premium", "sent to your email", "404", "page not found",
    "etprime", "subscription", "upgrade your plan", "paywall"
]
MIN_INSIGHT_LENGTH = 120

# --- 3. AI & Scraper Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
ua = UserAgent()

STOCK_MAP = {
    'RELIANCE.NS': 'Reliance Industries', 'TCS.NS': 'Tata Consultancy Services',
    'HDFCBANK.NS': 'HDFC Bank', 'TATAMOTORS.NS': 'Tata Motors',
    'TATASTEEL.NS': 'Tata Steel', 'SUNPHARMA.NS': 'Sun Pharma',
    'ITC.NS': 'ITC Ltd', 'ADANIPORTS.NS': 'Adani Ports',
    'SBIN.NS': 'State Bank of India', 'LT.NS': 'Larsen & Toubro'
}

FACTORS = {
    'Macro': ['Global Dollar', 'Foreign Investment', 'Inflation', 'GDP', 'Interest Rate'],
    'Policy': ['Tariffs', 'Budget', 'Tax', 'Regulation', 'Government', 'Policy'],
    'Corporate': ['Merger', 'Acquisition', 'New Product', 'Layoff', 'Hiring', 'CEO'],
    'Financials': ['Revenue', 'Profit', 'Earnings', 'Dividend', 'Cash Flow'],
    'Sentiment': ['Up', 'Down', 'Bullish', 'Bearish', 'Optimistic', 'Pessimistic']
}

# --- 4. Helper Functions ---
def clean_url(url):
    if not url: return None
    return url.split('&ved')[0].split('&usg')[0].split('?ved')[0].split('?usg')[0]

def is_valid_content(text, insight):
    """Checks validity. Insight must be substantial and junk-free."""
    if not text or not insight: return False

    # Check for Junk
    combined = (text + " " + insight).lower()
    for phrase in JUNK_PHRASES:
        if phrase in combined:
            return False

    # Check Length
    if len(insight) < MIN_INSIGHT_LENGTH:
        return False

    return True

def get_sentiment_scores(sentences):
    if not sentences: return []
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return (probs[:, 0] - probs[:, 1]).cpu().numpy()

def get_deep_insight(url):
    """Attempts to scrape the full article text."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text

        sentences = nltk.sent_tokenize(text)
        if not sentences: return None, None

        scores = get_sentiment_scores(sentences)
        idx = np.argmax(np.abs(scores))
        insight = sentences[idx]

        if is_valid_content(text, insight):
            return insight, scores[idx]
        return None, None
    except:
        return None, None

def fetch_top_10_results(company, date_str):
    """Fetches Top 10 results with Descriptions."""
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    next_date = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
    query = f'{company} after:{date_str} before:{next_date}'
    try:
        googlenews = GoogleNews(lang='en', region='IN')
        googlenews.clear()
        googlenews.user_agent = ua.random
        googlenews.search(query)
        return googlenews.results()[:10]
    except:
        return []

# --- 5. Main Repair Logic ---
if os.path.exists(FILE_PATH):
    print(f"📂 Loading {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH)

    # TARGET ONLY LOGIN/JUNK ROWS
    mask_bad = (
        df['Key_Insight'].str.contains('No valid article|No significant news', case=False, na=False) |
        df['Key_Insight'].str.lower().str.contains('|'.join(JUNK_PHRASES), na=False)
    )
    bad_indices = df[mask_bad].index.tolist()
    print(f"⚠️ Found {len(bad_indices)} Login/Junk Rows to Fix.")

    for idx in bad_indices:
        row = df.loc[idx]
        ticker = row['Ticker']
        company = STOCK_MAP.get(ticker, ticker)
        print(f"\n🔧 Fixing: {row['Date']} | {ticker}...")

        results = fetch_top_10_results(company, row['Date'])
        success = False

        if results:
            for i, res in enumerate(results):
                url = clean_url(res['link'])
                title = res.get('title', '')
                desc = res.get('desc', '') # The Google Snippet

                print(f"   [{i+1}/10] {title[:30]}...", end=" ")

                # STRATEGY 1: Full Article Scrape (Preferred)
                insight, score = get_deep_insight(url)

                if insight:
                    print("✅ SCRAPED FULL TEXT!")

                # STRATEGY 2: Snippet Fallback (If scrape failed/blocked)
                # We combine Title + Description to create a "Full" insight (~200+ chars)
                elif desc and len(desc) > 50:
                    combined_text = f"{title}. {desc}"
                    # Only accept if it looks substantial (avoid short garbage)
                    #Strategy 2 execution. If the full scrape returns None (usually due to a paywall), it checks if Google provided a sufficiently long summary snippet.
                    if len(combined_text) > 80 and not any(j in combined_text.lower() for j in JUNK_PHRASES):
                        print("⚠️ Scrape blocked -> USING SNIPPET.")
                        insight = combined_text
                        score = get_sentiment_scores([combined_text])[0]

                # SAVE DATA (If either Strategy worked)
                if insight:
                    df.at[idx, 'Key_Insight'] = insight
                    df.at[idx, 'Source_URL'] = url
                    df.at[idx, 'Headline'] = title
                    df.at[idx, 'Score_Sentiment'] = score

                    # Update Factors
                    full_text = (title + " " + insight).lower()
                    for cat, keywords in FACTORS.items():
                         if any(kw.lower() in full_text for kw in keywords):
                             df.at[idx, f'Score_{cat}'] = score

                    success = True
                    break # Stop looking for this row
                else:
                    print("❌ (Junk/Paywall)")

        if not success:
            print("   ⚠️ Failed. (Link dead & Snippet missing).")

        # Save progress
        if idx % 5 == 0: df.to_csv(FILE_PATH, index=False)
        time.sleep(random.uniform(5, 8))

    df.to_csv(FILE_PATH, index=False)
    print("\n🚀 Repair Complete!")
else:
    print("❌ File not found.")
