import yfinance as yf
import pandas as pd
import numpy as np

# ===== Question 4: Earnings Surprise Analysis for Amazon (AMZN) =====

ticker = 'AMZN'
ticker_obj = yf.Ticker(ticker)

# Step 1: Load earnings data
print("Fetching earnings dates...")
earnings = ticker_obj.get_earnings_dates(limit=30)
print(f"Total earnings entries fetched: {len(earnings)}")
print(earnings.to_string())
print()

# Filter: keep only rows where both EPS columns are filled (actual reported quarters)
# The future date row won't have 'Reported EPS' or 'Surprise(%)' filled
earnings = earnings.dropna(subset=['Reported EPS', 'Surprise(%)'])
print(f"Entries with reported EPS (excluding future): {len(earnings)}")

# Normalize index to date only (remove time/timezone)
earnings.index = pd.to_datetime(earnings.index).normalize().tz_localize(None)
earnings = earnings.sort_index()
print(f"\nEarnings dates range: {earnings.index[0].date()} to {earnings.index[-1].date()}")

# Step 2: Download AMZN historical price data
print("\nDownloading AMZN historical price data...")
hist = yf.download(ticker, start='2018-01-01', progress=False)
if isinstance(hist.columns, pd.MultiIndex):
    hist.columns = hist.columns.get_level_values(0)

hist = hist[['Close']].dropna()
hist.index = pd.to_datetime(hist.index).normalize().tz_localize(None)
hist = hist.sort_index()
print(f"Price data: {hist.index[0].date()} to {hist.index[-1].date()}, {len(hist)} rows")

# Step 3: Calculate 2-day percentage changes
# For each Day 2 (announcement day), find Day 1 (previous trading day) and Day 3 (next trading day)
# Return = Close_Day3 / Close_Day1 - 1
trading_days = hist.index.tolist()

results = []
for earn_date in earnings.index:
    # Find the position of earn_date in trading days (Day 2)
    if earn_date in hist.index:
        idx = trading_days.index(earn_date)
    else:
        # Find the nearest next trading day
        future = [d for d in trading_days if d >= earn_date]
        if not future:
            continue
        idx = trading_days.index(future[0])

    # Need at least 1 day before and 1 day after
    if idx < 1 or idx >= len(trading_days) - 1:
        continue

    day1 = trading_days[idx - 1]
    day2 = trading_days[idx]      # earnings announcement day
    day3 = trading_days[idx + 1]

    close_day1 = hist.loc[day1, 'Close']
    close_day3 = hist.loc[day3, 'Close']

    two_day_return = (close_day3 / close_day1 - 1) * 100

    surprise_pct = earnings.loc[earn_date, 'Surprise(%)']
    reported_eps = earnings.loc[earn_date, 'Reported EPS']
    est_eps      = earnings.loc[earn_date, 'EPS Estimate']

    results.append({
        'earn_date':      earn_date,
        'day1':           day1,
        'day3':           day3,
        'close_day1':     round(float(close_day1), 2),
        'close_day3':     round(float(close_day3), 2),
        'two_day_return': round(float(two_day_return), 4),
        'reported_eps':   float(reported_eps),
        'est_eps':        float(est_eps) if pd.notna(est_eps) else None,
        'surprise_pct':   float(surprise_pct)
    })

df = pd.DataFrame(results).set_index('earn_date')
print(f"\nEarnings rows matched with price data: {len(df)}")
print("\nFull earnings + 2-day return table:")
print(df[['close_day1', 'close_day3', 'two_day_return', 'surprise_pct']].to_string())

# Step 4: Filter for positive earnings surprises
df_pos = df[df['surprise_pct'] > 0].copy()
print(f"\nPositive earnings surprise events: {len(df_pos)}")

median_return = df_pos['two_day_return'].median()
print(f"\nMedian 2-day return (positive surprises): {median_return:.4f}%")

# Step 5: Correlation between 2-day return and surprise magnitude
corr_all = df[['two_day_return', 'surprise_pct']].corr()
corr_pos = df_pos[['two_day_return', 'surprise_pct']].corr()

print(f"\nCorrelation (all events):             {corr_all.loc['two_day_return','surprise_pct']:.4f}")
print(f"Correlation (positive surprise only): {corr_pos.loc['two_day_return','surprise_pct']:.4f}")

# Bonus: bull vs bear market reaction (using S&P 500 trend)
print("\n--- Bull vs Bear Market Analysis (S&P 500 200-day MA) ---")
sp500 = yf.download('^GSPC', start='2018-01-01', progress=False)
if isinstance(sp500.columns, pd.MultiIndex):
    sp500.columns = sp500.columns.get_level_values(0)
sp500.index = pd.to_datetime(sp500.index).normalize().tz_localize(None)
sp500['ma200'] = sp500['Close'].rolling(200).mean()
sp500['bull'] = sp500['Close'] > sp500['ma200']

df['bull_market'] = df.index.map(
    lambda d: bool(sp500.loc[sp500.index.asof(d), 'bull']) if sp500.index.asof(d) in sp500.index else None
)

bull_median  = df[df['bull_market'] == True]['two_day_return'].median()
bear_median  = df[df['bull_market'] == False]['two_day_return'].median()
print(f"Median 2-day return in BULL market: {bull_median:.4f}%")
print(f"Median 2-day return in BEAR market: {bear_median:.4f}%")

# ===== FINAL ANSWER =====
print("\n" + "="*55)
print("QUESTION 4 — FINAL ANSWER")
print("="*55)
print(f"  Positive surprise events          : {len(df_pos)}")
print(f"  Median 2-day return (pos surprise): {median_return:.4f}%  --> Answer: {round(median_return)}")
print(f"  Surprise vs Return correlation    : {corr_all.loc['two_day_return','surprise_pct']:.4f}")
print("="*55)
