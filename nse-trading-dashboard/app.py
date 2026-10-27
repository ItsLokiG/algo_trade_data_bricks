import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import requests
import xml.etree.ElementTree as ET
import json, time, warnings
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
warnings.filterwarnings('ignore')

st.set_page_config(page_title="NSE Trading Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
analyzer = SentimentIntensityAnalyzer()

# ═══════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS (inline implementations)
# ═══════════════════════════════════════════════════════════════
def sma(s, p): return s.rolling(p).mean()
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def rsi(s, p=14):
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(p).mean()
    loss = (-delta.clip(upper=0)).rolling(p).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))
def atr(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()
def macd(s, fast=12, slow=26, signal=9):
    ema_f, ema_s = ema(s, fast), ema(s, slow)
    macd_line = ema_f - ema_s
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist
def bollinger_bands(s, p=20, mult=2.0):
    m = s.rolling(p).mean()
    sd = s.rolling(p).std()
    return m + mult*sd, m, m - mult*sd
def stochastic(h, l, c, k_p=14, d_p=3):
    ll, hh = l.rolling(k_p).min(), h.rolling(k_p).max()
    k = 100 * ((c - ll) / (hh - ll + 1e-8))
    return k, k.rolling(d_p).mean()

# ═══════════════════════════════════════════════════════════════
# STOCK UNIVERSE (Large + Midcap)
# ═══════════════════════════════════════════════════════════════
UNIVERSE = [
    "HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","INDUSINDBK","BANDHANBNK","AUBANK","PNB","CANBK",
    "FEDERALBNK","BANKBARODA","IDFCFIRSTB","BAJFINANCE","BAJAJFINSV","CHOLAFIN","MUTHOOTFIN","PFC","RECLTD","SBILIFE",
    "HDFCLIFE","ICICIPRULI","ICICIGI","TCS","INFY","WIPRO","HCLTECH","TECHM","PERSISTENT","COFORGE",
    "MPHASIS","RELIANCE","ONGC","IOC","BPCL","GAIL","PETRONET","POWERGRID","NTPC","NHPC",
    "TATAPOWER","TATASTEEL","JSWSTEEL","HINDALCO","VEDL","NMDC","SAIL","COALINDIA","JINDALSTEL","MARUTI",
    "M&M","BAJAJ-AUTO","EICHERMOT","HEROMOTOCO","ASHOKLEY","TVSMOTOR","BOSCHLTD","MRF","HINDUNILVR","ITC",
    "NESTLEIND","BRITANNIA","TATACONSUM","DABUR","GODREJCP","COLPAL","UBL","MARICO","SUNPHARMA","DRREDDY",
    "CIPLA","DIVISLAB","AUROPHARMA","APOLLOHOSP","LALPATHLAB","ZYDUSLIFE","BIOCON","TORNTPHARM","ULTRACEMCO","GRASIM",
    "AMBUJACEM","SHREECEM","ACC","DALBHARAT","RAMCOCEM","DLF","PRESTIGE","OBEROIRLTY","BHARTIARTL","SUNTV",
    "NAUKRI","JUBLFOOD","LT","SIEMENS","ABB","BHEL","BEL","CUMMINSIND","THERMAX","HAVELLS",
    "POLYCAB","ADANIPORTS","ADANIENT","TITAN","ASIANPAINT","BERGEPAINT","DMART","PAGEIND","IRCTC","CONCOR",
    "TATACOMM","IDEA","INDIGO","ZYDUSLIFE","MAXHEALTH","KIMS","RAINBOW","ASTERDM","KPITTECH","TATAELXSI",
    "CDSL","INDIAMART","TRENT","VMART","DEVYANI","INOXWIND","SUZLON","RVNL","IRFC","MAZDOCK",
    "BDL","SJVN","NLCINDIA","EXIDEIND","AMARAJABAT","BAYERCROP","PIDILITIND","APLAPOLLO","WELCORP","ASTRAL",
    "SUPREMEIND","KINGFA","CEATLTD","APOLLOTYRE","JKTYRE","ESCORTS","BRIGADE","GODREJPROP","PHOENIXMILL","TATVA",
    "GLENMARK","LAURUSLABS","AJIPHARMA","AARTIIND","ALKEM","LUPIN","IPCALAB","ABBOTINDIA","COROMANDEL","PIIND",
    "CHAMBALFERT","GMRINFRA","ADANIENSOL","FSL","RBLBANK","YESBANK","IOB","UCOBANK","MAHABANK","IDBI",
    "HONASA","NYKAA","PAYTM","POLICYBZR","CARTRADE","EASEMYTRIP","ZOMATO","MCDOWELL-N","GUJGASLTD",
]
seen = set()
UNIVERSE = [x for x in UNIVERSE if not (x in seen or seen.add(x))]

# ═══════════════════════════════════════════════════════════════
# DATA FETCHING (cached)
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock(sym, period="1y"):
    try:
        df = yf.Ticker(f"{sym}.NS").history(period=period, interval="1d")
        if len(df) < 20: return None
        df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        return df
    except: return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_nifty(period="2y"):
    try:
        df = yf.Ticker("^NSEI").history(period=period, interval="1d")
        df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        return df
    except: return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_stocks(symbols, period="1y"):
    results = {}
    for i, sym in enumerate(symbols):
        df = fetch_stock(sym, period)
        if df is not None:
            results[sym] = df
        if (i+1) % 20 == 0: time.sleep(0.3)
    return results

@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(sym, max_items=5):
    try:
        url = f"https://news.google.com/rss/search?q={sym}+NSE+stock&hl=en-IN&gl=IN&ceid=IN:en"
        resp = requests.get(url, timeout=10)
        root = ET.fromstring(resp.content)
        items = root.findall('.//item')[:max_items]
        news = []
        for item in items:
            title = item.find('title').text if item.find('title') is not None else ""
            pub = item.find('pubDate').text if item.find('pubDate') is not None else ""
            link = item.find('link').text if item.find('link') is not None else ""
            source = item.find('source').text if item.find('source') is not None else ""
            sentiment = analyzer.polarity_scores(title)['compound']
            news.append({"title": title, "source": source, "date": pub, "link": link, "sentiment": sentiment})
        return news
    except: return []

@st.cache_data(ttl=600, show_spinner=False)
def compute_indicators(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    out = df.copy()
    out["sma20"] = sma(c, 20)
    out["sma50"] = sma(c, 50)
    out["rsi"] = rsi(c, 14)
    out["atr"] = atr(h, l, c, 14)
    out["atr_pct"] = (out["atr"] / c) * 100
    m, s, out["macd_hist"] = macd(c)
    bb_u, bb_m, bb_l = bollinger_bands(c)
    out["bb_upper"], out["bb_lower"] = bb_u, bb_l
    out["vol_ratio"] = v / v.rolling(5).mean()
    out["mom_1d"] = c.pct_change(1) * 100
    out["mom_5d"] = c.pct_change(5) * 100
    out["mom_10d"] = c.pct_change(10) * 100
    out["mom_20d"] = c.pct_change(20) * 100
    out["above_sma50"] = (c > out["sma50"]).astype(int)
    out["52w_high"] = c.rolling(252).max()
    out["52w_low"] = c.rolling(252).min()
    return out

# ═══════════════════════════════════════════════════════════════
# PAPER TRADE STATE
# ═══════════════════════════════════════════════════════════════
STATE_PATH_DAILY = "/Workspace/Repos/lokigovindasamy@gmail.com/algo_trade_data_bricks/daily_swing_paper_trade_state.json"
STATE_PATH_WEEKLY = "/Workspace/Repos/lokigovindasamy@gmail.com/algo_trade_data_bricks/swing_paper_trade_state.json"

def load_state(path):
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        import io
        resp = w.workspace.download(path)
        content = resp.read() if hasattr(resp, 'read') else io.BytesIO(resp).read()
        return json.loads(content)
    except:
        return {"capital": 100000, "positions": {}, "trade_log": [], "start_date": datetime.now().isoformat()}

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📊 NSE Trading Dashboard")
    st.markdown("---")
    strategy = st.radio("Strategy", ["Daily Swing (10-Day)", "Weekly Swing (4-Week)"], index=0)
    state_path = STATE_PATH_DAILY if "Daily" in strategy else STATE_PATH_WEEKLY
    max_hold = 10 if "Daily" in strategy else 28
    stop_mult = 1.5
    target_mult = 2.0 if "Daily" in strategy else 3.0
    st.markdown(f"**Max Hold:** {max_hold} days")
    st.markdown(f"**Stop:** {stop_mult}x ATR | **Target:** {target_mult}x ATR")
    st.markdown("---")
    st.subheader("Filters")
    rsi_min = st.slider("RSI Min", 0, 100, 35)
    rsi_max = st.slider("RSI Max", 0, 100, 72)
    only_above_sma50 = st.checkbox("Only Above SMA50", True)
    only_nifty_bull = st.checkbox("Only NIFTY Bull", True)
    st.markdown("---")
    st.subheader("Stock Search")
    stock_search = st.selectbox("Select Stock", [""] + sorted(UNIVERSE), index=0)
    st.markdown("---")
    st.caption("Built on Databricks Apps | Real-time data via yfinance")
    st.caption("⚠️ Research only — not investment advice")

# ═══════════════════════════════════════════════════════════════
# MAIN CONTENT - TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["📈 Dashboard", "💼 Paper Trading", "🔍 Screener", "✅ Signals", "👁️ Watchlist", "📰 News", "📊 Fundamentals", "ℹ️ About", "📉 Backtest Results"])

# ─── TAB 1: DASHBOARD ───
with tab1:
    st.header("📈 Market Overview")
    with st.spinner("Loading market data..."):
        nifty = fetch_nifty()
        state = load_state(state_path)

    if nifty is not None and len(nifty) > 50:
        nc = nifty["close"]
        ns = sma(nc, 50)
        nifty_bull = nc.iloc[-1] > ns.iloc[-1] if not pd.isna(ns.iloc[-1]) else False
        nifty_ret = (nc.iloc[-1] / nc.iloc[0] - 1) * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("NIFTY 50", f"{nc.iloc[-1]:,.0f}", f"{nifty_ret:+.1f}%")
        with col2:
            badge = "🟢 BULL" if nifty_bull else "🔴 BEAR"
            st.metric("Regime", badge, f"SMA50: {ns.iloc[-1]:,.0f}")
        with col3:
            pv = state.get("capital", 100000)
            for sym, pos in state.get("positions", {}).items():
                df = fetch_stock(sym)
                if df is not None:
                    pv += pos.get("shares", 0) * df["close"].iloc[-1]
            total_ret = (pv / state.get("initial_capital", 100000) - 1) * 100
            st.metric("Portfolio", f"₹{pv:,.0f}", f"{total_ret:+.1f}%")
        with col4:
            st.metric("Open Positions", str(len(state.get("positions", {}))))

        st.markdown("---")
        col_l, col_r = st.columns([2, 1])
        with col_l:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nifty.index, y=nc, name="NIFTY 50", line=dict(color='#00d4ff', width=2)))
            fig.add_trace(go.Scatter(x=nifty.index, y=ns, name="SMA50", line=dict(color='#ff9900', width=1, dash='dash')))
            fig.update_layout(title="NIFTY 50 with SMA50", height=400, template="plotly_dark",
                xaxis_rangeslider_visible=False, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)
        with col_r:
            st.subheader("Trade Stats")
            tl = state.get("trade_log", [])
            if tl:
                tdf = pd.DataFrame(tl)
                wins = (tdf["return_pct"] > 0).sum()
                wr = wins / len(tdf) * 100 if len(tdf) > 0 else 0
                pf = abs(tdf[tdf["return_pct"]>0]["return_pct"].sum() / tdf[tdf["return_pct"]<0]["return_pct"].sum()) if (tdf[tdf["return_pct"]<0]["return_pct"].sum() != 0 and len(tdf[tdf["return_pct"]<0]) > 0) else 0
                st.metric("Total Trades", str(len(tdf)))
                st.metric("Win Rate", f"{wr:.1f}%")
                st.metric("Profit Factor", f"{pf:.2f}")
                st.metric("Avg Return", f"{tdf['return_pct'].mean():.2f}%")
                exit_counts = tdf["exit_reason"].value_counts()
                fig2 = go.Figure(data=[go.Pie(labels=exit_counts.index, values=exit_counts.values, hole=0.4)])
                fig2.update_layout(title="Exit Reasons", height=300, template="plotly_dark")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No trades yet. Start paper trading to see stats.")

        # Market breadth
        st.markdown("---")
        st.subheader("📊 Market Breadth")
        with st.spinner("Scanning stocks..."):
            sample = UNIVERSE[:50]
            breadth_data = []
            for sym in sample:
                df = fetch_stock(sym)
                if df is not None and len(df) > 50:
                    c = df["close"]
                    s50 = sma(c, 50).iloc[-1]
                    if not pd.isna(s50):
                        breadth_data.append({"symbol": sym, "above_sma50": c.iloc[-1] > s50, "rsi": rsi(c,14).iloc[-1]})
            if breadth_data:
                bdf = pd.DataFrame(breadth_data)
                above = bdf["above_sma50"].sum()
                below = len(bdf) - above
                col1, col2 = st.columns(2)
                with col1:
                    fig3 = go.Figure(data=[go.Pie(labels=["Above SMA50","Below SMA50"], values=[above,below], hole=0.4,
                        marker=dict(colors=["#00ff88","#ff4444"]))])
                    fig3.update_layout(title="Stocks Above/Below SMA50", height=300, template="plotly_dark")
                    st.plotly_chart(fig3, use_container_width=True)
                with col2:
                    rsi_ranges = {"Oversold (<30)": (0,30), "Neutral (30-70)": (30,70), "Overbought (>70)": (70,100)}
                    rsi_counts = {k: ((bdf["rsi"]>=v[0]) & (bdf["rsi"]<v[1])).sum() for k,v in rsi_ranges.items()}
                    fig4 = go.Figure(data=[go.Bar(x=list(rsi_counts.keys()), y=list(rsi_counts.values()),
                        marker=dict(color=["#ff4444","#00d4ff","#ff9900"]))])
                    fig4.update_layout(title="RSI Distribution", height=300, template="plotly_dark")
                    st.plotly_chart(fig4, use_container_width=True)
    else:
        st.error("Could not load NIFTY data")

# ─── TAB 2: PAPER TRADING ───
with tab2:
    st.header("💼 Paper Trading")
    state = load_state(state_path)

    col1, col2, col3 = st.columns(3)
    with col1:
        pv = state.get("capital", 100000)
        for sym, pos in state.get("positions", {}).items():
            df = fetch_stock(sym)
            if df is not None: pv += pos.get("shares",0) * df["close"].iloc[-1]
        st.metric("Portfolio Value", f"₹{pv:,.0f}", f"{(pv/state.get('initial_capital',100000)-1)*100:+.1f}%")
    with col2:
        st.metric("Cash", f"₹{state.get('capital',100000):,.0f}")
    with col3:
        st.metric("Open Positions", str(len(state.get("positions", {}))))

    st.markdown("---")
    st.subheader("📋 Current Holdings")
    positions = state.get("positions", {})
    if positions:
        pos_data = []
        for sym, pos in positions.items():
            df = fetch_stock(sym)
            cp = df["close"].iloc[-1] if df is not None else pos.get("entry_price",0)
            pnl = (cp - pos["entry_price"]) / pos["entry_price"] * 100
            pos_data.append({
                "Symbol": sym, "Entry Price": pos["entry_price"], "Current Price": cp,
                "P&L %": round(pnl,2), "Stop Loss": pos.get("stop",0), "Target": pos.get("target",0),
                "Shares": round(pos.get("shares",0),2), "Entry Date": pos.get("entry_date",""),
            })
        pdf = pd.DataFrame(pos_data).sort_values("P&L %", ascending=False)
        st.dataframe(pdf, use_container_width=True, hide_index=True)

        pnl_vals = [p["P&L %"] for p in pos_data]
        colors = ['#00ff88' if v > 0 else '#ff4444' for v in pnl_vals]
        fig = go.Figure(data=[go.Bar(x=[p["Symbol"] for p in pos_data], y=pnl_vals, marker_color=colors)])
        fig.update_layout(title="P&L by Position", height=300, template="plotly_dark", yaxis_title="P&L %")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No open positions. Check the Signals tab for trade ideas.")

    st.markdown("---")
    st.subheader("📜 Trade History")
    tl = state.get("trade_log", [])
    if tl:
        tdf = pd.DataFrame(tl)
        st.dataframe(tdf, use_container_width=True, hide_index=True)
        fig2 = go.Figure()
        if "exit_date" in tdf.columns:
            tdf["cum_ret"] = tdf["return_pct"].cumsum()
            fig2.add_trace(go.Scatter(x=tdf["exit_date"], y=tdf["cum_ret"], mode='lines+markers',
                line=dict(color='#00d4ff', width=2), name="Cumulative Return %"))
            fig2.update_layout(title="Cumulative P&L", height=300, template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No trade history yet.")

# ─── TAB 3: STOCK SCREENER ───
with tab3:
    st.header("🔍 Stock Screener")
    with st.spinner("Scanning universe..."):
        scan_data = []
        for sym in UNIVERSE[:100]:
            df = fetch_stock(sym)
            if df is not None and len(df) > 50:
                ind = compute_indicators(df)
                last = ind.iloc[-1]
                scan_data.append({
                    "Symbol": sym, "Price": round(last["close"],2),
                    "RSI": round(last["rsi"],1) if not pd.isna(last["rsi"]) else 0,
                    "SMA50 Ratio": round(last["close"]/last["sma50"],3) if not pd.isna(last["sma50"]) else 0,
                    "ATR %": round(last["atr_pct"],2) if not pd.isna(last["atr_pct"]) else 0,
                    "Vol Ratio": round(last["vol_ratio"],2) if not pd.isna(last["vol_ratio"]) else 0,
                    "Mom 1d": round(last["mom_1d"],2) if not pd.isna(last["mom_1d"]) else 0,
                    "Mom 5d": round(last["mom_5d"],2) if not pd.isna(last["mom_5d"]) else 0,
                    "Mom 10d": round(last["mom_10d"],2) if not pd.isna(last["mom_10d"]) else 0,
                    "Mom 20d": round(last["mom_20d"],2) if not pd.isna(last["mom_20d"]) else 0,
                    "Above SMA50": bool(last["above_sma50"]),
                    "52w High": round(last["52w_high"],2) if not pd.isna(last["52w_high"]) else 0,
                    "52w Low": round(last["52w_low"],2) if not pd.isna(last["52w_low"]) else 0,
                })
    if scan_data:
        sdf = pd.DataFrame(scan_data)
        filtered = sdf.copy()
        filtered = filtered[(filtered["RSI"] >= rsi_min) & (filtered["RSI"] <= rsi_max)]
        if only_above_sma50:
            filtered = filtered[filtered["Above SMA50"] == True]
        sort_col = st.selectbox("Sort by", filtered.columns[1:], index=3)
        filtered = filtered.sort_values(sort_col, ascending=False)
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(filtered)} of {len(sdf)} stocks")

        st.markdown("---")
        st.subheader("📈 Stock Detail")
        detail_sym = st.selectbox("Select stock for chart", sorted(UNIVERSE), key="screener_detail")
        if detail_sym:
            df = fetch_stock(detail_sym, "6mo")
            if df is not None:
                ind = compute_indicators(df)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=ind.index, open=ind["open"], high=ind["high"],
                    low=ind["low"], close=ind["close"], name="OHLC"))
                fig.add_trace(go.Scatter(x=ind.index, y=ind["sma20"], name="SMA20", line=dict(color='#ff9900', width=1)))
                fig.add_trace(go.Scatter(x=ind.index, y=ind["sma50"], name="SMA50", line=dict(color='#ff4444', width=1)))
                fig.update_layout(title=f"{detail_sym} - Candlestick", height=500, template="plotly_dark",
                    xaxis_rangeslider_visible=False, margin=dict(l=20,r=20,t=40,b=20))
                st.plotly_chart(fig, use_container_width=True)
                col1, col2, col3, col4 = st.columns(4)
                last = ind.iloc[-1]
                with col1: st.metric("RSI", f"{last['rsi']:.1f}")
                with col2: st.metric("Price", f"₹{last['close']:.1f}")
                with col3: st.metric("ATR %", f"{last['atr_pct']:.2f}%")
                with col4: st.metric("Vol Ratio", f"{last['vol_ratio']:.2f}")
    else:
        st.error("Could not scan stocks")

# ─── TAB 4: RECOMMENDED TRADES (Signals) ───
with tab4:
    st.header("✅ Recommended Trades")
    nifty = fetch_nifty()
    nifty_bull = False
    if nifty is not None and len(nifty) > 50:
        nifty_bull = nifty["close"].iloc[-1] > sma(nifty["close"], 50).iloc[-1]
    if only_nifty_bull and not nifty_bull:
        st.warning("⚠️ NIFTY is in BEAR regime. No new entries recommended. Strategy is in cash.")
        st.info("💡 The swing strategy only buys when NIFTY 50 is above its 50-day SMA (bull regime). Check back when the market turns bullish.")
    else:
        with st.spinner("Scanning for trade signals..."):
            signals = []
            for sym in UNIVERSE[:100]:
                df = fetch_stock(sym)
                if df is None or len(df) < 50: continue
                ind = compute_indicators(df)
                last = ind.iloc[-1]
                if pd.isna(last["rsi"]) or pd.isna(last["sma50"]) or pd.isna(last["atr"]): continue
                if last["rsi"] < rsi_min or last["rsi"] > rsi_max: continue
                if only_above_sma50 and last["close"] <= last["sma50"]: continue
                stop = last["close"] - stop_mult * last["atr"]
                target = last["close"] + target_mult * last["atr"]
                rr = (target - last["close"]) / (last["close"] - stop) if (last["close"] - stop) > 0 else 0
                signals.append({
                    "Symbol": sym, "Price": round(last["close"],2), "RSI": round(last["rsi"],1),
                    "ATR %": round(last["atr_pct"],2), "Stop Loss": round(stop,2),
                    "Target": round(target,2), "Risk/Reward": round(rr,2),
                    "Mom 5d": round(last["mom_5d"],2), "Mom 10d": round(last["mom_10d"],2),
                    "Vol Ratio": round(last["vol_ratio"],2),
                })
        if signals:
            sig_df = pd.DataFrame(signals).sort_values("Risk/Reward", ascending=False)
            st.dataframe(sig_df, use_container_width=True, hide_index=True)
            st.success(f"Found {len(sig_df)} trade candidates")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sig_df["Symbol"], y=sig_df["Risk/Reward"], mode='markers',
                marker=dict(size=10, color=sig_df["RSI"], colorscale='RdYlGn', showscale=True),
                text=sig_df.apply(lambda r: f"RSI:{r['RSI']} Price:{r['Price']}", axis=1)))
            fig.update_layout(title="Risk/Reward by Stock (color = RSI)", height=400, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No stocks pass all filters. Try adjusting RSI range or unchecking filters in the sidebar.")

# ─── TAB 5: WATCHLIST ───
with tab5:
    st.header("👁️ Watchlist")
    if 'watchlist' not in st.session_state:
        st.session_state['watchlist'] = ['HDFCBANK', 'RELIANCE', 'TCS', 'INFY', 'BAJFINANCE']
    new_stock = st.selectbox("Add stock to watchlist", [""] + sorted(UNIVERSE), key="watchlist_add")
    if st.button("➕ Add to Watchlist") and new_stock:
        if new_stock not in st.session_state['watchlist']:
            st.session_state['watchlist'].append(new_stock)
            st.success(f"Added {new_stock} to watchlist")
        else:
            st.warning(f"{new_stock} already in watchlist")
    watchlist = st.session_state['watchlist']
    if watchlist:
        st.markdown(f"**{len(watchlist)} stocks on watchlist**")
        if st.button("🗑️ Clear Watchlist"):
            st.session_state['watchlist'] = []
            st.rerun()
        wl_data = []
        for sym in watchlist:
            df = fetch_stock(sym)
            if df is not None and len(df) > 50:
                ind = compute_indicators(df)
                last = ind.iloc[-1]
                rsi_val = last["rsi"] if not pd.isna(last["rsi"]) else 0
                alert = ""
                if rsi_val > 70: alert = "🔴 Overbought"
                elif rsi_val < 30: alert = "🟢 Oversold"
                elif last["close"] < last["sma50"] and not pd.isna(last["sma50"]): alert = "⚠️ Below SMA50"
                wl_data.append({
                    "Symbol": sym, "Price": round(last["close"],2),
                    "RSI": round(rsi_val,1), "SMA50": round(last["sma50"],2) if not pd.isna(last["sma50"]) else 0,
                    "Mom 5d": round(last["mom_5d"],2) if not pd.isna(last["mom_5d"]) else 0,
                    "Mom 20d": round(last["mom_20d"],2) if not pd.isna(last["mom_20d"]) else 0,
                    "52w High": round(last["52w_high"],2) if not pd.isna(last["52w_high"]) else 0,
                    "52w Low": round(last["52w_low"],2) if not pd.isna(last["52w_low"]) else 0,
                    "Alert": alert,
                })
        if wl_data:
            wdf = pd.DataFrame(wl_data)
            st.dataframe(wdf, use_container_width=True, hide_index=True)
            selected = st.selectbox("View chart", watchlist)
            if selected:
                df = fetch_stock(selected, "3mo")
                if df is not None:
                    ind = compute_indicators(df)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ind.index, y=ind["close"], name="Close", line=dict(color='#00d4ff', width=2)))
                    fig.add_trace(go.Scatter(x=ind.index, y=ind["sma20"], name="SMA20", line=dict(color='#ff9900', width=1)))
                    fig.add_trace(go.Scatter(x=ind.index, y=ind["sma50"], name="SMA50", line=dict(color='#ff4444', width=1)))
                    bb_u, _, bb_l = bollinger_bands(ind["close"])
                    fig.add_trace(go.Scatter(x=ind.index, y=bb_u, name="BB Upper", line=dict(color='gray', width=1, dash='dash')))
                    fig.add_trace(go.Scatter(x=ind.index, y=bb_l, name="BB Lower", line=dict(color='gray', width=1, dash='dash')))
                    fig.update_layout(title=f"{selected} - Price & Indicators", height=400, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Watchlist is empty. Add stocks using the dropdown above.")

# ─── TAB 6: MARKET NEWS ───
with tab6:
    st.header("📰 Market News & Sentiment")
    news_sym = st.selectbox("Select stock for news", ["NIFTY"] + sorted(UNIVERSE), key="news_sym")
    search_sym = "NSE" if news_sym == "NIFTY" else news_sym
    with st.spinner(f"Fetching news for {news_sym}..."):
        news_items = fetch_news(search_sym, 10)
    if news_items:
        avg_sent = np.mean([n["sentiment"] for n in news_items])
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Avg Sentiment", f"{avg_sent:.3f}")
        with col2: st.metric("Positive", str(sum(1 for n in news_items if n["sentiment"] > 0.05)))
        with col3: st.metric("Negative", str(sum(1 for n in news_items if n["sentiment"] < -0.05)))
        st.markdown("---")
        for item in news_items:
            color = "🟢" if item["sentiment"] > 0.05 else ("🔴" if item["sentiment"] < -0.05 else "⚪")
            st.markdown(f"{color} **{item['title']}**")
            st.caption(f"{item['source']} | {item['date']} | Sentiment: {item['sentiment']:.3f}")
            if item.get('link'): st.markdown(f"[Read more]({item['link']})")
            st.markdown("")
    else:
        st.info(f"No news found for {news_sym}")
    st.markdown("---")
    st.subheader("📈 Market Sentiment Scan")
    with st.spinner("Scanning sentiment for top stocks..."):
        sent_data = []
        for sym in UNIVERSE[:20]:
            news = fetch_news(sym, 3)
            if news:
                avg_s = np.mean([n["sentiment"] for n in news])
                sent_data.append({"Symbol": sym, "Avg Sentiment": round(avg_s,3),
                    "Headlines": len(news), "Latest": news[0]["title"][:80] + "..." if news[0]["title"] else ""})
    if sent_data:
        sent_df = pd.DataFrame(sent_data).sort_values("Avg Sentiment", ascending=False)
        st.dataframe(sent_df, use_container_width=True, hide_index=True)
        fig = go.Figure()
        colors = ['#00ff88' if s > 0 else '#ff4444' for s in sent_df["Avg Sentiment"]]
        fig.add_trace(go.Bar(x=sent_df["Symbol"], y=sent_df["Avg Sentiment"], marker_color=colors))
        fig.update_layout(title="Sentiment by Stock", height=350, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# ─── TAB 7: FUNDAMENTALS ───
with tab7:
    st.header("📊 Fundamental Analysis")
    st.markdown("*Data from Screener.in (if accessible). ROCE, P/E, revenue growth, profit growth.*")
    fund_sym = st.selectbox("Select stock for fundamentals", sorted(UNIVERSE), key="fund_sym")
    if fund_sym:
        df = fetch_stock(fund_sym, "1y")
        if df is not None:
            ind = compute_indicators(df)
            last = ind.iloc[-1]
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Price", f"₹{last['close']:.1f}")
            with col2: st.metric("52W High", f"₹{last['52w_high']:.1f}" if not pd.isna(last['52w_high']) else "N/A")
            with col3: st.metric("52W Low", f"₹{last['52w_low']:.1f}" if not pd.isna(last['52w_low']) else "N/A")
            with col4: st.metric("ATR %", f"{last['atr_pct']:.2f}%" if not pd.isna(last['atr_pct']) else "N/A")
            st.markdown("---")
            st.subheader("📈 Technical Summary")
            tech_data = {
                "Indicator": ["RSI (14)", "Price vs SMA20", "Price vs SMA50", "MACD Histogram", "Volume Ratio", "Momentum 5d", "Momentum 20d", "Bollinger %B"],
                "Value": [
                    f"{last['rsi']:.1f}",
                    f"{((last['close']/last['sma20']-1)*100):+.1f}%" if not pd.isna(last['sma20']) else "N/A",
                    f"{((last['close']/last['sma50']-1)*100):+.1f}%" if not pd.isna(last['sma50']) else "N/A",
                    f"{last['macd_hist']:.2f}" if not pd.isna(last['macd_hist']) else "N/A",
                    f"{last['vol_ratio']:.2f}" if not pd.isna(last['vol_ratio']) else "N/A",
                    f"{last['mom_5d']:.2f}%" if not pd.isna(last['mom_5d']) else "N/A",
                    f"{last['mom_20d']:.2f}%" if not pd.isna(last['mom_20d']) else "N/A",
                    f"{((last['close']-last['bb_lower'])/(last['bb_upper']-last['bb_lower'])*100):.1f}%" if not pd.isna(last['bb_upper']) and not pd.isna(last['bb_lower']) else "N/A",
                ],
            }
            st.table(pd.DataFrame(tech_data))
            st.markdown("---")
            st.subheader("📋 Fundamental Screener")
            st.info("💡 To enable Screener.in scraping, ensure the app has network access to screener.in. "
                    "The following table shows technical quality metrics as a proxy:")
            quality_data = []
            for sym in UNIVERSE[:50]:
                df2 = fetch_stock(sym)
                if df2 is not None and len(df2) > 252:
                    ind2 = compute_indicators(df2)
                    last2 = ind2.iloc[-1]
                    ann_ret = (df2["close"].iloc[-1] / df2["close"].iloc[-252] - 1) * 100
                    quality_data.append({
                        "Symbol": sym, "Annual Return": round(ann_ret,1),
                        "RSI": round(last2["rsi"],1) if not pd.isna(last2["rsi"]) else 0,
                        "Volatility": round(df2["close"].pct_change().rolling(20).std().iloc[-1]*np.sqrt(252)*100,1) if len(df2)>20 else 0,
                        "Above SMA50": bool(last2["above_sma50"]),
                        "Price vs 52W High": round((last2["close"]/last2["52w_high"]-1)*100,1) if not pd.isna(last2["52w_high"]) else 0,
                    })
            if quality_data:
                qdf = pd.DataFrame(quality_data).sort_values("Annual Return", ascending=False)
                st.dataframe(qdf, use_container_width=True, hide_index=True)
                st.markdown("---")
                st.subheader("🏆 Top Performers (Annual Return)")
                top10 = qdf.head(10)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=top10["Symbol"], y=top10["Annual Return"],
                    marker_color=['#00ff88' if v > 0 else '#ff4444' for v in top10["Annual Return"]]))
                fig.update_layout(title="Top 10 Stocks by Annual Return", height=350, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

# ─── TAB 8: ABOUT ───
with tab8:
    st.header("ℹ️ About This App")
    st.markdown("---")
    st.subheader("🎯 What It Does")
    st.markdown("""
    The **NSE Trading Dashboard** is an experimental interface for Indian-stock research.
    It displays third-party market data, technical indicators, candidate signals, and simulated
    paper-trading history. Its data is not an exchange-certified live feed, and its strategy
    performance has not been validated.
    """)
    st.markdown("---")
    st.subheader("📊 Candidate Strategies — Unvalidated")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Weekly Swing (4-Week Max Hold)
        - **Universe**: 130 NIFTY 200 stocks
        - **Features**: 53 (technical + Google News RSS sentiment + Screener.in fundamentals)
        - **ML Model**: LightGBM with triple-barrier labeling
        - **Entry**: ML top picks + NIFTY bull regime + uptrend (close>SMA50) + RSI 40-70 + ROCE>15%
        - **Exit**: ATR stop (-1.5x), ATR target (+3x), 4-week hold, trend exit
        - **Rebalance**: Weekly (Friday)
        - **Performance**: Not validated; no return claim is available
        """)
    with col2:
        st.markdown("""
        ### Daily Swing (10-Day Max Hold)
        - **Universe**: 224 stocks (large + midcap)
        - **Features**: 37 (technical + cross-sectional + NIFTY regime)
        - **ML Model**: LightGBM with 10-day triple-barrier labeling
        - **Entry**: ML top picks + NIFTY bull regime + uptrend (close>SMA50) + RSI 35-72
        - **Exit**: ATR stop (-1.5x), ATR target (+2x), 10-day hold, trend exit
        - **Rebalance**: Daily
        - **Performance**: Not validated; no return claim is available
        """)
    st.markdown("---")
    st.subheader("⚙️ How The Algorithm Works")
    st.markdown("""
    #### 1. Data Collection (legacy dashboard)
    - **Price Data**: Daily OHLCV from Yahoo Finance (yfinance); unofficial and not guaranteed real-time
    - **News Sentiment**: Google News RSS feed parsed with VADER sentiment analysis (10 headlines per stock)
    - **Fundamentals**: ROCE, revenue growth, profit growth from Screener.in
    - **NIFTY Regime**: NIFTY 50 daily close vs SMA50 (bull/bear filter)
    #### 2. Feature Engineering (53 Features)
    - **Technical**: RSI, SMA ratios, MACD, Bollinger Bands, ATR%, Stochastic, Williams %R
    - **Momentum**: 1d, 2d, 3d, 5d, 10d, 20d, 50d returns
    - **Cross-sectional**: Rank and z-score of momentum/RSI/volume across all stocks
    - **Regime**: NIFTY bull/bear, NIFTY momentum, NIFTY volatility
    - **Sentiment**: Avg news sentiment, positive/negative headline counts
    - **Fundamental**: ROCE, revenue growth, profit growth, P/E, P/B
    #### 3. ML Model (legacy notebook design)
    - The notebooks use LightGBM and triple-barrier labels.
    - Their test period is reused for early stopping, so reported test metrics are not independent.
    - Predictions and backtest fills require an execution-timing and point-in-time-data rebuild.
    #### 4. Entry Rules (All Must Pass)
    1. NIFTY in bull regime (close > SMA50)
    2. Stock in uptrend (close > SMA50)
    3. RSI between 35-72 (not overbought)
    4. ML probability ranked in top picks
    5. (Weekly only) ROCE > 15% (quality filter)
    #### 5. Exit Rules (First Trigger Wins)
    - **Stop Loss**: Price hits entry - 1.5x ATR(14)
    - **Take Profit**: Price hits entry + 2-3x ATR(14)
    - **Time Exit**: Max holding period reached (10 days / 4 weeks)
    - **Trend Exit**: Price drops 3% below SMA50
    #### 6. Risk Management
    - Max 12 positions (equal weight)
    - No leverage, no short selling
    - NIFTY bear regime = 100% cash (no new entries)
    - ATR-based position sizing
    """)
    st.markdown("---")
    st.subheader("❓ Q&A")
    with st.expander("Is this financial advice?"):
        st.markdown("**No.** This is a research and education tool. All trading is simulated (paper trading). Historical performance does not guarantee future results. Never treat AI-generated signals as proof of future profitability.")
    with st.expander("What data sources are used?"):
        st.markdown("- **Yahoo Finance** (yfinance): Unofficial daily/weekly OHLCV; availability, adjustments, and freshness need validation\n- **Google News RSS**: Headlines scored with VADER\n- **Screener.in**: Fundamental data\n- **Tapetide MCP**: Optional assistant research connection; not wired into this dashboard or Python backtest\nNo source here is treated as an exchange-certified execution feed.")
    with st.expander("What is the ML model's accuracy?"):
        st.markdown("The legacy notebooks reported AUC values, but those test metrics are not reliable because the test period was used for early stopping. The dashboard does not currently have a validated model evaluation or established trading edge.")
    with st.expander("Why are there two strategies?"):
        st.markdown("They are two research prototypes with different bar frequencies. Neither has a validated performance record; choose neither on the basis of the legacy backtest figures.")
    with st.expander("When does the strategy go to cash?"):
        st.markdown("When **NIFTY 50 closes below its 50-day SMA**, the strategy enters BEAR regime and stops buying. All existing positions are managed with stops/targets but no new entries are made. This is the single most important risk management rule.")
    with st.expander("How do I use paper trading?"):
        st.markdown("1. Go to the **Paper Trading** tab to see current positions and trade history\n2. Go to the **Signals** tab to see today's recommended trades\n3. The strategy automatically generates BUY signals when NIFTY is in bull regime\n4. State persists between app restarts (saved in workspace JSON files)\n5. Run the notebook cells to update the ML model and generate fresh signals")
    with st.expander("What are the costs?"):
        st.markdown("- **App compute**: May be billed while running, depending on deployment\n- **Data**: Provider terms and limits apply\n- **Trading**: This dashboard simulates trades and does not send broker orders")
    with st.expander("Can I use this for real trading?"):
        st.markdown("**Not yet.** Before going live you would need:\n1. Broker API integration (Zerodha Kite, Upstox, Angel One)\n2. Real-time tick data (not daily bars)\n3. Rigorous out-of-sample testing with live paper trading for 3-6 months\n4. Proper order management and slippage modeling\n5. Risk limits and daily loss circuit breakers")
    with st.expander("How accurate is the news sentiment?"):
        st.markdown("Uses **VADER sentiment analysis** on Google News headlines (not full articles). Sentiment scores range from -1 (very negative) to +1 (very positive). The ML model uses this as one of 53 features — sentiment alone is NOT a trading signal.")
    st.markdown("---")
    st.subheader("🛠️ Tech Stack")
    st.markdown("| Component | Technology |\n|---|---|\n| UI Framework | Streamlit |\n| Charts | Plotly (interactive) |\n| Data | yfinance (Yahoo Finance) |\n| ML | LightGBM |\n| Sentiment | VADER + Google News RSS |\n| Fundamentals | Screener.in |\n| Platform | Databricks Apps |\n| Backend | Python 3.11 |")
    st.markdown("---")
    st.caption("⚠️ This is a research system. Historical performance does not guarantee future results. All execution is simulated.")

# ─── TAB 9: BACKTEST RESULTS ───
with tab9:
    st.header("📉 Backtest Results")
    st.warning(
        "No validated strategy backtest is available yet. Legacy headline "
        "returns were removed because the notebooks contain execution-timing, "
        "validation, and cost-modeling issues. Paper-trade history below is "
        "simulation output, not proof of profitability."
    )
    st.subheader("📜 Legacy Paper-Simulation Trade Log")
    bt_strategy = st.radio("Select strategy", ["Daily Swing", "Weekly Swing"], horizontal=True, key="bt_strategy")
    bt_path = STATE_PATH_DAILY if "Daily" in bt_strategy else STATE_PATH_WEEKLY
    bt_state = load_state(bt_path)
    tl = bt_state.get("trade_log", [])
    if tl:
        tdf = pd.DataFrame(tl)
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Trades", str(len(tdf)))
        with col2:
            wins = (tdf["return_pct"] > 0).sum()
            st.metric("Win Rate", f"{wins/len(tdf)*100:.1f}%")
        with col3: st.metric("Avg Return", f"{tdf['return_pct'].mean():.2f}%")
        with col4:
            best = tdf.loc[tdf["return_pct"].idxmax()]
            st.metric("Best Trade", f"{best['return_pct']:+.1f}%", best["symbol"])
        st.markdown("---")
        st.dataframe(tdf, use_container_width=True, hide_index=True)
        st.markdown("---")
        if "exit_date" in tdf.columns:
            tdf_sorted = tdf.sort_values("exit_date")
            tdf_sorted["cum_ret"] = tdf_sorted["return_pct"].cumsum()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tdf_sorted["exit_date"], y=tdf_sorted["cum_ret"], mode='lines+markers', line=dict(color='#00d4ff', width=2), name="Cumulative Return %"))
            fig.update_layout(title=f"{bt_strategy} — Cumulative P&L", height=400, template="plotly_dark", yaxis_title="Cumulative Return %", xaxis_title="Exit Date")
            st.plotly_chart(fig, use_container_width=True)
        fig2 = go.Figure()
        colors = ['#00ff88' if v > 0 else '#ff4444' for v in tdf["return_pct"]]
        fig2.add_trace(go.Bar(x=tdf["symbol"], y=tdf["return_pct"], marker_color=colors))
        fig2.update_layout(title=f"{bt_strategy} — P&L by Trade", height=350, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🏆 Top 10 Winners**")
            winners = tdf[tdf["return_pct"] > 0].nlargest(10, "return_pct")
            if len(winners) > 0: st.dataframe(winners, use_container_width=True, hide_index=True)
            else: st.info("No winning trades yet.")
        with col2:
            st.markdown("**💀 Top 10 Losers**")
            losers = tdf[tdf["return_pct"] <= 0].nsmallest(10, "return_pct")
            if len(losers) > 0: st.dataframe(losers, use_container_width=True, hide_index=True)
            else: st.info("No losing trades yet.")
    else:
        st.info(f"No {bt_strategy} trades executed yet. Run the strategy notebook to generate trades.")