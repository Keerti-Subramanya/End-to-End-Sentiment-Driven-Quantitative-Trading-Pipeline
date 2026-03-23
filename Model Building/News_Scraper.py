import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
import os
import sys
from GoogleNews import GoogleNews
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fake_useragent import UserAgent
from newspaper import Article
import nltk

# --- 1. Setup & Safety ---
try:
    from google.colab import drive
    drive.mount('/content/drive')
    OUTPUT_FILE = '/content/drive/MyDrive/news_factors_90d_deep.csv'
except ImportError:
    print("Not running in Colab. Using local path.")
    OUTPUT_FILE = 'news_factors_90d_deep.csv'

# Download NLTK data for sentence tokenization
nltk.download('punkt')
nltk.download('punkt_tab')

# --- Configuration ---
STOCK_MAP = {
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'Tata Consultancy Services',
    'HDFCBANK.NS': 'HDFC Bank',
    'TATAMOTORS.NS': 'Tata Motors',
    'TATASTEEL.NS': 'Tata Steel',
    'SUNPHARMA.NS': 'Sun Pharma',
    'ITC.NS': 'ITC Ltd',
    'ADANIPORTS.NS': 'Adani Ports',
    'SBIN.NS': 'State Bank of India',
    'LT.NS': 'Larsen & Toubro'
}

FACTORS = {
    'Macro': ['Global Dollar', 'Foreign Investment', 'Investments Regulations', 'Inflation', 'GDP', 'Interest Rate'],
    'Policy': ['Tariffs', 'Budget', 'Tax', 'Regulation', 'Government', 'Policy'],
    'Corporate': ['Merger', 'Acquisition', 'New Product', 'Layoff', 'Hiring', 'Insider', 'CEO', 'Board'],
    'Financials': ['Revenue', 'Profit', 'Earnings', 'Quarterly', 'Dividend', 'Cash Flow'],
    'Sentiment': ['Up', 'Down', 'Bullish', 'Bearish', 'Optimistic', 'Pessimistic']
}

# --- GPU Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Initialize FinBERT ---
print("Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

def get_sentiment_scores(sentences):
    """Returns a list of sentiment scores for each sentence."""
    if not sentences:
        return []

    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # FinBERT labels: 0: positive, 1: negative, 2: neutral
    # Score = pos - neg
    scores = (probs[:, 0] - probs[:, 1]).cpu().numpy()
    return scores

def clean_url(url):
    """Removes Google tracking parameters from the URL."""
    if not url:
        return url
    # Google URLs often have &ved=... and &usg=...
    clean = url.split('&ved')[0].split('&usg')[0].split('?ved')[0].split('?usg')[0]
    return clean

def get_deep_insight(url):
    """Parses full article and extracts the most impactful sentence."""
    try:
        article = Article(url)
        article.download()
        article.parse()

        text = article.text
        if not text:
            return None, None

        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return None, None

        scores = get_sentiment_scores(sentences)

        # Find sentence with highest absolute sentiment score
        abs_scores = np.abs(scores)
        idx = np.argmax(abs_scores)

        return sentences[idx], scores[idx]
    except Exception as e:
        print(f"      Error parsing article: {e}")
        return None, None

# --- 2. Scraper Logic ---
ua = UserAgent()

def fetch_top_results(company_name, current_date):
    """Fetches top 3 results using strict date query."""
    next_date = current_date + timedelta(days=1)

    # Format: YYYY-MM-DD
    date_str_before = next_date.strftime('%Y-%m-%d')
    date_str_after = current_date.strftime('%Y-%m-%d')

    query = f'{company_name} after:{date_str_after} before:{date_str_before}'

    try:
        googlenews = GoogleNews(lang='en', region='IN')
        googlenews.clear()
        googlenews.user_agent = ua.random

        googlenews.search(query)
        results = googlenews.results()

        # Filter out results without link or title
        valid_results = [res for res in results if res.get('link') and res.get('title')]
        return valid_results[:3]
    except Exception as e:
        if "429" in str(e):
            print('\n🔴 BLOCKED. STOPPING EXECUTION.')
            sys.exit()
        print(f"    Error fetching search results: {e}")
        return []

# --- 3. Smart Resume Logic ---
default_start_date = datetime.now() - timedelta(days=90)
current_run_date = default_start_date

columns = ['Date', 'Ticker', 'Score_Macro', 'Score_Policy', 'Score_Corporate',
           'Score_Financials', 'Score_Sentiment', 'Key_Insight', 'Source_URL', 'Headline']

if os.path.exists(OUTPUT_FILE):
    try:
        df_existing = pd.read_csv(OUTPUT_FILE)
        if not df_existing.empty:
            last_date_str = df_existing['Date'].iloc[-1]
            last_date_dt = datetime.strptime(last_date_str, '%Y-%m-%d')
            current_run_date = last_date_dt + timedelta(days=1)
            print(f"Resuming from {current_run_date.strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"Error reading existing file: {e}. Starting fresh.")
else:
    pd.DataFrame(columns=columns).to_csv(OUTPUT_FILE, index=False)
    print(f"Creating new file. Starting from {current_run_date.strftime('%Y-%m-%d')}")

# --- 4. Main Execution Loop ---
end_date = datetime.now()

while current_run_date < end_date:
    date_str = current_run_date.strftime('%Y-%m-%d')
    print(f"\nProcessing Date: {date_str}")

    daily_results = []

    for ticker, company_name in STOCK_MAP.items():
        print(f"  Scraping {ticker} ({company_name})...")

        top_3 = fetch_top_results(company_name, current_run_date)

        row = {
            'Date': date_str,
            'Ticker': ticker,
            'Score_Macro': 0.0,
            'Score_Policy': 0.0,
            'Score_Corporate': 0.0,
            'Score_Financials': 0.0,
            'Score_Sentiment': 0.0,
            'Key_Insight': 'No valid article found',
            'Source_URL': 'N/A',
            'Headline': 'No significant news'
        }

        article_found = False
        for result in top_3:
            raw_url = result['link']
            url = clean_url(raw_url)
            headline = result['title']

            print(f"    Trying: {url}")
            insight, insight_score = get_deep_insight(url)

            if insight:
                row['Key_Insight'] = insight
                row['Source_URL'] = url
                row['Headline'] = headline
                row['Score_Sentiment'] = insight_score

                # Check keywords in insight/headline for categories
                combined_text = (headline + " " + insight).lower()
                for cat, keywords in FACTORS.items():
                    if any(kw.lower() in combined_text for kw in keywords):
                        # For simplicity, assign the main insight score to the category
                        row[f'Score_{cat}'] = insight_score

                article_found = True
                print(f"      ✅ Success: {headline[:50]}...")
                break # Found a valid article
            else:
                print(f"      ❌ Failed (Paywall/Parse Error)")

        daily_results.append(row)

        # 'Turtle Mode' Delay
        delay = random.uniform(30, 60)
        print(f"    Waiting {delay:.2f}s...")
        time.sleep(delay)

    # Incremental save
    if daily_results:
        pd.DataFrame(daily_results).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        print(f"  Saved {date_str} to file.")

    current_run_date += timedelta(days=1)

print("\nDeep Insight News Dataset Build Complete.")
