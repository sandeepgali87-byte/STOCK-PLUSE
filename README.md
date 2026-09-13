# Market Pulse — India (Stocks & Mutual Funds)

A free, self-hosted web app that gives you a **transparent, multi-factor
read** on NSE stocks and Indian mutual funds — technical trend + basic
fundamentals + recent news sentiment — with a suggested Buy/Hold/Sell call,
a rough holding horizon, and heuristic stop-loss/target levels.

## ⚠️ Read this before using it

- **No system reliably predicts short-term market moves with anywhere near
  99% accuracy** — not this tool, not professional quant funds. Markets
  price in new information almost instantly, and short-term price action
  has a large random component.
- This tool turns public data into a **rule-based score you can see and
  question** (click "Why this suggestion?" in the app). It is a research
  aid, not a crystal ball.
- **Paper-trade first.** Log every suggestion this tool gives you (symbol,
  date, suggestion, price) in a spreadsheet for several weeks. Compare
  against what actually happened before ever using real money. Track your
  own hit-rate — don't assume the tool's internal score equals real-world
  accuracy.
- Data comes from free sources (Yahoo Finance via `yfinance`, `mfapi.in`,
  Google News RSS) and can be delayed, incomplete, or occasionally wrong.
  Always cross-check with your broker/AMC before acting.

## What it actually does

**Stocks (NSE):**
1. Pulls price history via Yahoo Finance (`yfinance`) — free, no API key.
2. Computes technical indicators: SMA20/50, RSI(14), MACD, ATR(14).
3. Pulls fundamentals from Yahoo's data: P/E, ROE, debt/equity, profit
   margin, revenue growth.
4. Pulls recent (14-day) news headlines via Google News RSS and scores
   sentiment with VADER (a standard open-source sentiment model).
5. Combines these into a weighted composite score (technical 45%,
   fundamental 35%, sentiment 20%) → Strong Buy / Buy / Hold / Sell /
   Strong Sell, plus a "conviction" rating based on how many of the three
   signal groups agree.
6. Suggests a holding horizon and stop-loss/target based on recent
   volatility (ATR) — a standard risk-sizing heuristic, not a promise.

**Mutual Funds:**
1. Searches scheme names via `mfapi.in` (free, community-run NAV database).
2. Pulls full NAV history and computes trailing 1M/3M/6M/1Y/3Y returns.
3. Gives a simple momentum read. Note: there's no free benchmark/category
   data source, so this does **not** compare against a category average —
   for mutual funds, short-term NAV trend is a weak signal anyway; they're
   built for long-term, SIP-style investing.

## Setup

Requires Python 3.9+.

```bash
cd stock_pulse
pip install -r requirements.txt
streamlit run app.py
```

It will open in your browser at `http://localhost:8501`. To view it on
your phone, open the same URL using your computer's local network IP
(shown in the terminal as "Network URL") while both devices are on the
same Wi-Fi.

## Customizing the logic

All the scoring rules live in plain Python in `app.py` — search for
`tech_score`, `fund_score`, and `sent_score`. Adjust weights, thresholds,
or add new signals (e.g. sector comparison, more fundamental ratios) as
you learn what actually works from your paper-trading log. That feedback
loop — not any one-shot formula — is what improves this over time.

## Known limitations

- `yfinance` fundamentals for smaller/less-covered NSE stocks can be
  sparse — the app skips missing fields rather than guessing.
- Google News RSS returns general headlines matching the company name;
  it isn't filtered for relevance the way a paid news API would be.
- No options/derivatives, no intraday tick data, no order execution —
  this is analysis-only by design.
