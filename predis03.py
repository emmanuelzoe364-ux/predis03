import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import requests

# --- INITIAL SETTINGS ---
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="Revelation Engine Pro")

# --- SIDEBAR & CACHE BUSTER ---
if 'refresh_count' not in st.session_state:
    st.session_state.refresh_count = 0

with st.sidebar:
    st.header("🚀 Turbo Control")
    primary_ticker = st.text_input("Primary Ticker", value="BTC-USD").strip().upper()
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h"], index=0)
    period = st.selectbox("Period", ["1d", "3d", "7d"], index=0)
    
    st.divider()
    ema_fast_val = st.number_input("Fast EMA", value=30)
    ema_slow_val = st.number_input("Slow EMA", value=72)
    z_win_val = st.number_input("Z-Score Window", value=20)

    if st.button("🔥 FORCE LIVE REFRESH"):
        st.cache_data.clear()
        st.session_state.refresh_count += 1
        st.rerun()

# --- HIGH-PRIORITY DATA ENGINE ---
@st.cache_data(ttl=15)
def fetch_turbo_data(main_t, tf, prd, refresh_id):
    try:
        # Browser Spoofing to prevent Cloud Throttling
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
        }
        session = requests.Session()
        session.headers.update(headers)

        tickers = [main_t, "PAXG-USD", "BTC-USD", "ETH-USD"]
        
        # KEY FIX: Removed 'proxy' keyword. Using session only.
        data = yf.download(
            tickers=tickers,
            period=prd,
            interval=tf,
            auto_adjust=True,
            prepost=True,
            session=session, 
            threads=False,
            progress=False,
            group_by='ticker'
        )
        
        if data.empty: 
            return None, None, None, "Yahoo returned no data. Try refreshing in 30 seconds."

        # Extract primary data
        m_df = data[main_t].copy()
        idx = m_df.index
        
        # Align Intermarket Ratios
        p_c = data["PAXG-USD"]['Close'].reindex(idx).ffill().bfill()
        b_c = data["BTC-USD"]['Close'].reindex(idx).ffill().bfill()
        e_c = data["ETH-USD"]['Close'].reindex(idx).ffill().bfill()

        return m_df, (p_c/b_c), (b_c/e_c), None
    except Exception as e:
        return None, None, None, str(e)

# EXECUTE DATA FETCH
df, pb_ratio, be_ratio, err = fetch_turbo_data(primary_ticker, timeframe, period, st.session_state.refresh_count)

if err:
    st.error(f"📡 Connection Issue: {err}")
elif df is not None:
    # LIVE LATENCY MONITOR
    last_candle = df.index[-1]
    now_utc = datetime.utcnow()
    diff_seconds = (now_utc - last_candle.replace(tzinfo=None)).total_seconds()
    
    with st.sidebar:
        st.write(f"**Last Candle (UTC):** {last_candle.strftime('%H:%M:%S')}")
        if diff_seconds > 180:
            st.error(f"⚠️ Yahoo Lag: {int(diff_seconds/60)}m")
        else:
            st.success(f"✅ Data is Live ({int(diff_seconds)}s lag)")

    # --- MATH SECTION ---
    # Native EMAs
    df['EMA_30'] = df['Close'].ewm(span=ema_fast_val, adjust=False).mean()
    df['EMA_72'] = df['Close'].ewm(span=ema_slow_val, adjust=False).mean()
    
    # EMA Divergence Z-Score
    df['Spread'] = df['EMA_30'] - df['EMA_72']
    df['Z_EMA'] = (df['Spread'] - df['Spread'].rolling(z_win_val).mean()) / (df['Spread'].rolling(z_win_val).std() + 1e-9)

    # Intermarket Z-Scores
    def get_z(ser, win):
        return (ser - ser.rolling(win).mean()) / (ser.rolling(win).std() + 1e-9)
    
    df['Z_GOLD_BTC'] = get_z(pb_ratio, z_win_val)
    df['Z_BTC_ETH'] = get_z(be_ratio, z_win_val)

    # REVELATION PHASE (-2 to 2 ORIGINAL SCALE)
    rng = (df['High'] - df['Low']) + 1e-9
    pd_v = 1.0 - ((2 * df['Close'] - (df['High'] + df['Low'])) / rng)
    u, d = (df['High'] - df['High'].shift(1)), (df['Low'].shift(1) - df['Low'])
    
    phases, h_vals = [], []
    for up, dw, val in zip(u, d, pd_v):
        if up > dw and up > 0:
            phases.append("Dp" if val > 1.0 else "rP")
            h_vals.append(val)
        elif dw > up and dw > 0:
            phases.append("rD" if val > 1.0 else "Ad")
            h_vals.append(-val)
        else:
            phases.append("Neutral")
            h_vals.append(0)
    df['H_Val'], df['Phase'] = h_vals, phases

    # --- PLOT SECTION ---
    st.title(f"⚖️ {primary_ticker} Revelation Engine")
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, 
        vertical_spacing=0.02, 
        row_heights=[0.35, 0.15, 0.15, 0.15, 0.20],
        subplot_titles=(f"Price Action ({timeframe})", "Revelation P/D Scale", "EMA 30/72 Z-Score", "Gold/BTC Z-Score", "BTC/Eth Z-Score")
    )

    # Row 1: Candlesticks + EMAs
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_30'], line=dict(color='orange', width=1.5), name="EMA 30"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_72'], line=dict(color='cyan', width=1.5), name="EMA 72"), row=1, col=1)
    
    # Row 2: Original P/D
    c_map = {"rD":'red', "Ad":'orange', "rP":'lime', "Dp":'green', "Neutral":'gray'}
    fig.add_trace(go.Bar(x=df.index, y=df['H_Val'], marker_color=[c_map.get(p, 'gray') for p in df['Phase']], name="P/D"), row=2, col=1)
    
    # Row 3-5: Z-Scores
    fig.add_trace(go.Scatter(x=df.index, y=df['Z_EMA'], line=dict(color='yellow'), name="Z-EMA"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Z_GOLD_BTC'], line=dict(color='magenta'), name="Z-Gold"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Z_BTC_ETH'], line=dict(color='deepskyblue'), name="Z-Alts"), row=5, col=1)

    # Reference Lines for Z-Scores
    for r in [3, 4, 5]:
        fig.add_hline(y=2.0, line_dash="dash", line_color="white", row=r, col=1, opacity=0.3)
        fig.add_hline(y=-2.0, line_dash="dash", line_color="white", row=r, col=1, opacity=0.3)

    fig.update_layout(height=1200, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- DATA TABLE ---
    st.subheader("📋 Output Data Table (EMA Values Included)")
    # Included EMA_30 and EMA_72 in display columns
    cols = ['Close', 'EMA_30', 'EMA_72', 'H_Val', 'Z_EMA', 'Z_GOLD_BTC', 'Z_BTC_ETH', 'Phase']
    st.dataframe(df[cols].sort_index(ascending=False).head(100).round(4), use_container_width=True)
