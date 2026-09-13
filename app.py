"""
Market Pulse — Indian Stocks & Mutual Funds decision-support tool.

IMPORTANT: This is a heuristic decision-support tool, not a prediction engine.
No system can reliably hit anywhere near 99% accuracy on short-term market
timing. Use this to organize information and PAPER-TRADE for weeks before
risking real money. See README.md.
"""

import datetime as dt

import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Market Pulse (India)", layout="wide")

analyzer = SentimentIntensityAnalyzer()

# --------------------------------------------------------------------------
# Disclaimer (always visible)
# --------------------------------------------------------------------------
st.title("📈 Market Pulse — India (Stocks & Mutual Funds)")
st.error(
    "⚠️ **Not financial advice.** This tool combines public data into a "
    "transparent heuristic score. It is NOT a prediction engine and cannot "
    "guarantee accuracy. Paper-trade (track suggestions without real money) "
    "for at least a few weeks before considering real capital, and never "
    "invest more than you can afford to lose."
)

mode = st.sidebar.radio("What do you want to analyze?", ["Stock", "Mutual Fund"])


# ==========================================================================
# Shared helpers
# ==========================================================================
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def get_news_sentiment(query: str, max_items: int = 12):
    """Free Google News RSS + VADER sentiment. Returns (score -1..1, headlines)."""
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}%20when:14d&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        feed = feedparser.parse(url)
        headlines = [e.title for e in feed.entries[:max_items]]
    except Exception:
        headlines = []
    if not headlines:
        return 0.0, []
    scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
    return float(np.mean(scores)), headlines


def clamp(x, lo=-1, hi=1):
    return max(lo, min(hi, x))


def score_to_label(score: float) -> str:
    if score >= 0.5:
        return "🟢 Strong Buy"
    if score >= 0.15:
        return "🟢 Buy"
    if score > -0.15:
        return "🟡 Hold"
    if score > -0.5:
        return "🔴 Sell"
    return "🔴 Strong Sell"


# ==========================================================================
# STOCK MODE
# ==========================================================================
if mode == "Stock":
    st.sidebar.markdown("Enter an NSE symbol, e.g. `RELIANCE`, `TCS`, `INFY`, `HDFCBANK`")
    raw_symbol = st.sidebar.text_input("NSE Symbol", value="RELIANCE").strip().upper()
    period = st.sidebar.selectbox("History window", ["6mo", "1y", "2y"], index=1)
    analyze = st.sidebar.button("Analyze", type="primary")

    if analyze and raw_symbol:
        ticker_symbol = raw_symbol if raw_symbol.endswith(".NS") else raw_symbol + ".NS"

        with st.spinner(f"Fetching data for {ticker_symbol}..."):
            tk = yf.Ticker(ticker_symbol)
            hist = tk.history(period=period)
            info = {}
            try:
                info = tk.info or {}
            except Exception:
                info = {}

        if hist.empty:
            st.error(
                f"No data found for `{raw_symbol}`. Check the symbol is correct "
                "(use the NSE trading symbol, e.g. RELIANCE, TCS)."
            )
            st.stop()

        hist = hist.dropna()
        hist["SMA20"] = hist["Close"].rolling(20).mean()
        hist["SMA50"] = hist["Close"].rolling(50).mean()
        hist["RSI14"] = rsi(hist["Close"])
        macd_line, signal_line, hist_bar = macd(hist["Close"])
        hist["ATR14"] = atr(hist)

        last = hist.iloc[-1]
        price = last["Close"]
        atr_val = last["ATR14"] if not np.isnan(last["ATR14"]) else price * 0.02
        atr_pct = atr_val / price * 100

        # ---------------- Technical score ----------------
        tech_score = 0.0
        reasons_tech = []

        if not np.isnan(last["SMA20"]) and not np.isnan(last["SMA50"]):
            if price > last["SMA20"] > last["SMA50"]:
                tech_score += 0.4
                reasons_tech.append("Price above both SMA20 & SMA50 (uptrend)")
            elif price < last["SMA20"] < last["SMA50"]:
                tech_score -= 0.4
                reasons_tech.append("Price below both SMA20 & SMA50 (downtrend)")
            else:
                reasons_tech.append("Mixed trend signal from moving averages")

        rsi_val = last["RSI14"]
        if not np.isnan(rsi_val):
            if rsi_val < 30:
                tech_score += 0.3
                reasons_tech.append(f"RSI {rsi_val:.0f} — oversold, potential bounce")
            elif rsi_val > 70:
                tech_score -= 0.3
                reasons_tech.append(f"RSI {rsi_val:.0f} — overbought, pullback risk")
            else:
                reasons_tech.append(f"RSI {rsi_val:.0f} — neutral zone")

        last_macd_hist = hist_bar.iloc[-1]
        if not np.isnan(last_macd_hist):
            if last_macd_hist > 0:
                tech_score += 0.3
                reasons_tech.append("MACD histogram positive (bullish momentum)")
            else:
                tech_score -= 0.3
                reasons_tech.append("MACD histogram negative (bearish momentum)")

        tech_score = clamp(tech_score)

        # ---------------- Fundamental score ----------------
        fund_score = 0.0
        reasons_fund = []

        pe = info.get("trailingPE")
        if pe:
            if pe < 15:
                fund_score += 0.25
                reasons_fund.append(f"P/E {pe:.1f} — reasonably valued")
            elif pe > 40:
                fund_score -= 0.25
                reasons_fund.append(f"P/E {pe:.1f} — expensive vs earnings")
            else:
                reasons_fund.append(f"P/E {pe:.1f} — moderate valuation")

        roe = info.get("returnOnEquity")
        if roe:
            if roe > 0.15:
                fund_score += 0.25
                reasons_fund.append(f"ROE {roe*100:.1f}% — efficient use of equity")
            elif roe < 0.05:
                fund_score -= 0.25
                reasons_fund.append(f"ROE {roe*100:.1f}% — weak returns on equity")

        d2e = info.get("debtToEquity")
        if d2e is not None:
            if d2e < 50:
                fund_score += 0.2
                reasons_fund.append(f"Debt/Equity {d2e:.0f} — low leverage")
            elif d2e > 150:
                fund_score -= 0.2
                reasons_fund.append(f"Debt/Equity {d2e:.0f} — high leverage")

        margin = info.get("profitMargins")
        if margin:
            if margin > 0.12:
                fund_score += 0.15
                reasons_fund.append(f"Profit margin {margin*100:.1f}% — healthy")
            elif margin < 0.03:
                fund_score -= 0.15
                reasons_fund.append(f"Profit margin {margin*100:.1f}% — thin")

        rev_growth = info.get("revenueGrowth")
        if rev_growth:
            if rev_growth > 0.10:
                fund_score += 0.15
                reasons_fund.append(f"Revenue growth {rev_growth*100:.1f}% YoY — expanding")
            elif rev_growth < 0:
                fund_score -= 0.15
                reasons_fund.append(f"Revenue growth {rev_growth*100:.1f}% YoY — shrinking")

        fund_score = clamp(fund_score)
        if not reasons_fund:
            reasons_fund.append("Limited fundamental data available for this symbol")

        # ---------------- Sentiment score ----------------
        company_name = info.get("shortName", raw_symbol)
        sent_score, headlines = get_news_sentiment(f"{company_name} stock NSE")
        sent_score = clamp(sent_score * 2)  # VADER compound is usually small; amplify a bit

        # ---------------- Composite ----------------
        composite = clamp(0.45 * tech_score + 0.35 * fund_score + 0.20 * sent_score)
        label = score_to_label(composite)

        agree_count = sum(
            1 for s in [tech_score, fund_score, sent_score] if abs(s) > 0.1 and (s > 0) == (composite > 0)
        )
        conviction = "High" if agree_count == 3 else "Medium" if agree_count == 2 else "Low"

        # ---------------- Holding period / risk levels heuristic ----------------
        if atr_pct < 1.5:
            horizon = "Medium-term (3–8 weeks) — low volatility supports holding through noise"
        elif atr_pct < 3.5:
            horizon = "Short-to-medium (2–4 weeks) — moderate volatility, monitor weekly"
        else:
            horizon = "Short-term / trade only (3–7 days) — high volatility, tight risk control needed"

        if composite > 0:
            stop_loss = price - 1.5 * atr_val
            target = price + 2.5 * atr_val
        else:
            stop_loss = price + 1.5 * atr_val
            target = price - 2.5 * atr_val

        # ---------------- Display ----------------
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price (₹)", f"{price:,.2f}")
        c2.metric("Suggestion", label)
        c3.metric("Conviction", conviction)
        c4.metric("Suggested horizon", horizon.split(" — ")[0])

        st.caption(
            f"Composite score: {composite:+.2f} (range -1 strong sell to +1 strong buy) · "
            f"Technical {tech_score:+.2f} · Fundamental {fund_score:+.2f} · News sentiment {sent_score:+.2f}"
        )

        st.subheader("Suggested risk levels (heuristic, based on 14-day ATR)")
        r1, r2, r3 = st.columns(3)
        r1.metric("Entry / Current", f"₹{price:,.2f}")
        r2.metric("Stop-loss", f"₹{stop_loss:,.2f}")
        r3.metric("Target", f"₹{target:,.2f}")
        st.caption(horizon)

        with st.expander("Why this suggestion? (full reasoning)"):
            st.markdown("**Technical signals:**")
            for r in reasons_tech:
                st.markdown(f"- {r}")
            st.markdown("**Fundamental signals:**")
            for r in reasons_fund:
                st.markdown(f"- {r}")
            st.markdown(f"**News sentiment (last 14 days, {len(headlines)} headlines):**")
            if headlines:
                for h in headlines[:8]:
                    st.markdown(f"- {h}")
            else:
                st.markdown("- No recent headlines found")

        # ---------------- Chart ----------------
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.2, 0.25],
            vertical_spacing=0.03, subplot_titles=("Price & Moving Averages", "RSI (14)", "MACD"),
        )
        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"],
            low=hist["Low"], close=hist["Close"], name="Price"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist["SMA20"], name="SMA20", line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist["SMA50"], name="SMA50", line=dict(width=1)), row=1, col=1)

        fig.add_trace(go.Scatter(x=hist.index, y=hist["RSI14"], name="RSI14", line=dict(color="purple")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", row=2, col=1)

        fig.add_trace(go.Bar(x=hist.index, y=hist_bar, name="MACD Hist"), row=3, col=1)

        fig.update_layout(height=800, xaxis_rangeslider_visible=False, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    elif not analyze:
        st.info("Enter an NSE symbol in the sidebar and click **Analyze**.")


# ==========================================================================
# MUTUAL FUND MODE
# ==========================================================================
else:
    st.sidebar.markdown("Search by scheme name, e.g. `Parag Parikh Flexi Cap`")
    query = st.sidebar.text_input("Mutual Fund name / keyword", value="Parag Parikh Flexi Cap")
    search_btn = st.sidebar.button("Search", type="primary")

    if search_btn and query:
        with st.spinner("Searching mfapi.in..."):
            try:
                resp = requests.get(f"https://api.mfapi.in/mf/search?q={requests.utils.quote(query)}", timeout=15)
                results = resp.json()
            except Exception as e:
                st.error(f"Could not reach mfapi.in: {e}")
                st.stop()

        if not results:
            st.warning("No schemes found. Try a different / shorter keyword.")
            st.stop()

        options = {f"{r['schemeName']} (Code {r['schemeCode']})": r["schemeCode"] for r in results[:25]}
        chosen = st.selectbox("Select scheme", list(options.keys()))
        scheme_code = options[chosen]

        with st.spinner("Fetching NAV history..."):
            nav_resp = requests.get(f"https://api.mfapi.in/mf/{scheme_code}", timeout=15).json()

        nav_data = nav_resp.get("data", [])
        if not nav_data:
            st.error("No NAV history available for this scheme.")
            st.stop()

        df = pd.DataFrame(nav_data)
        df["date"] = pd.to_datetime(df["date"], dayfirst=True)
        df["nav"] = df["nav"].astype(float)
        df = df.sort_values("date").reset_index(drop=True)

        latest_date = df["date"].iloc[-1]
        latest_nav = df["nav"].iloc[-1]

        def trailing_return(months):
            target_date = latest_date - pd.DateOffset(months=months)
            past = df[df["date"] <= target_date]
            if past.empty:
                return None
            past_nav = past["nav"].iloc[-1]
            years = months / 12
            if years <= 1:
                return (latest_nav / past_nav - 1) * 100
            return ((latest_nav / past_nav) ** (1 / years) - 1) * 100

        r1m = trailing_return(1)
        r3m = trailing_return(3)
        r6m = trailing_return(6)
        r1y = trailing_return(12)
        r3y = trailing_return(36)

        st.subheader(nav_resp.get("meta", {}).get("scheme_name", chosen))
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("1M", f"{r1m:+.1f}%" if r1m is not None else "N/A")
        m2.metric("3M", f"{r3m:+.1f}%" if r3m is not None else "N/A")
        m3.metric("6M", f"{r6m:+.1f}%" if r6m is not None else "N/A")
        m4.metric("1Y", f"{r1y:+.1f}%" if r1y is not None else "N/A")
        m5.metric("3Y (CAGR)", f"{r3y:+.1f}%" if r3y is not None else "N/A")

        # Simple trend-based suggestion (NOT a category comparison — mfapi has no benchmark data)
        signals = [x for x in [r1m, r3m, r6m] if x is not None]
        momentum = np.mean(signals) if signals else 0
        if momentum > 3:
            mf_label = "🟢 Positive momentum — reasonable to continue/increase SIP"
        elif momentum > -1:
            mf_label = "🟡 Flat/mixed — continue existing SIP, avoid large lumpsum for now"
        else:
            mf_label = "🔴 Declining trend — review before adding lumpsum; SIPs generally ride through dips"

        st.metric("Heuristic read", mf_label)
        st.caption(
            "Mutual funds are long-term vehicles — short-term NAV trend is a weak signal. "
            "This does not compare against category average or benchmark (no free data source for that)."
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["nav"], mode="lines", name="NAV"))
        fig.update_layout(title="NAV History", height=450)
        st.plotly_chart(fig, use_container_width=True)

    elif not search_btn:
        st.info("Enter a mutual fund name in the sidebar and click **Search**.")
