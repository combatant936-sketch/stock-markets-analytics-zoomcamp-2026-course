I want to build a weekly market direction prediction model for the S&P 500, predicting whether the index will close higher or lower 5 trading days from today.

Data: S&P 500 daily OHLCV history from 1990 to present via yfinance

Features I plan to use:

RSI (14-day) — momentum / overbought-oversold signal
MACD and MACD signal line crossover — trend change detection
50-day and 200-day moving averages — short and long-term trend
Weekly return of the past 1, 2, and 4 weeks — recent momentum
VIX level — market fear/uncertainty
Model: Binary classification using Random Forest (up/down label), with a strict train/test time split to avoid data leakage. I will evaluate using accuracy and a simple backtest comparing cumulative returns vs a passive buy-and-hold strategy, measuring Sharpe Ratio.