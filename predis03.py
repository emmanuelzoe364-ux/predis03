import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import time

# --- INITIAL SETTINGS ---
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="Revelation Engine Cloud-Fix")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Primary Ticker", value="BTC-USD").strip().upper()
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "1d"], index=0)
    period = st.selectbox("Data Period", ["1d", "3d", "7d", "1mo"], index=2)
    
    st.divider()
    ema_fast_val = st.number_input("Fast EMA", value=30)
    ema_slow_val = st.number_input("Slow EMA", value=72)
    z_win_val = st.number_input("Z-Score Window", value=20)

    # CLOUD FIX: We use a session state counter to force the cache to break
    if 'refresh_count' not in st.session_state:
        st.session_state.refresh_count = 0

    if st.button("🚀 FORCE CLOUD UPDATE"):
        st.session_state.refresh_count += 1
        st.cache_data.clear()
        st.rerun()

    st.write(f"Cloud ID: {st.session_state.refresh_count}")
    st.write(f"Last Sync: {datetime.now().strftime('%H:%M:%S')} UTC")

st.title(f"⚖️ Revelation Engine: {ticker}")
st.caption("Environment: Streamlit Cloud Optimized | Real-Time Sync")

# --- CLOUD-OPTIMIZED DATA FETCH ---
# We pass 'refresh_id' to the function. Every time the button is clicked, 
# the ID changes, forcing the cloud server to bypass its old saved data.
@st.cache_data(ttl=30)
def fetch_cloud_data(main_t, tf, prd, refresh_id):
    try:
        # Standardize tickers
        t_list = [main_t, "PAXG-USD", "BTC-USD", "ETH-USD"]
        
        # We download in one block using 'prepost=True' for the latest ticks
        # and 'threads=False' for better stability on cloud servers
        data = yf.download(
            tickers=t_list,
            period=prd,
            interval=tf,
            auto_adjust=True,
            prepost=True,
            threads=False,
            progress=False,
            group_by='ticker'
        )
        
        if data.empty: return None, None, None, "Yahoo returned empty data."

        # Extracting data for each ticker manually to handle Cloud MultiIndex
        m_df = data[main_t].copy()
        p_df = data["PAXG-USD"].copy()
        b_df = data["BTC-USD"].copy()
        e_df = data["ETH-USD"].copy()

        # Clean individual dataframes
        idx = m_df.index
        p_c = p_df['Close'].reindex(idx).ffill().bfill()
        b_c = b_df['Close'].reindex(idx).ffill().bfill()
        e_c = e_df['Close'].reindex(idx).ffill().bfill()

        return m_df, (p_c/b_c), (b_c/e_c), None
    except Exception as e:
        return None, None, None, str(e)

# We pass the session_state refresh_count here
df, pb_ratio, be_ratio, err = fetch_cloud_data(ticker, timeframe, period, st.session_state.refresh_count)

if err:
    st.error(f"Cloud Connection Error: {err}")
    st.info("Try clicking 'FORCE CLOUD UPDATE' again. Yahoo sometimes rejects the first cloud request.")
elif df is not None:
    # --- MATH SECTION ---
    # EMA Calculation (Pandas native for stability)
    df['EMA_30'] = df['Close'].ewm(span=ema_fast_val, adjust=False).mean()
    df['EMA_72'] = df['Close'].ewm(span=ema_slow_val, adjust=False).mean()
    
    # Z-Scores
    df['Spread'] = df['EMA_30'] - df['EMA_72']
    df['Z_EMA'] = (df['Spread'] - df['Spread'].rolling(z_win_val).mean()) / (df['Spread'].rolling(z_win_val).std() + 1e-9)
    df['Z_GOLD_BTC'] = (pb_ratio - pb_ratio.rolling(z_win_val).mean()) / (pb_ratio.rolling(z_win_val).std() + 1e-9)
    df['Z_BTC_ETH'] = (be_ratio - be_ratio.rolling(z_win_val).mean()) / (be_ratio.rolling(z_win_val).std() + 1e-9)

    # --- REVELATION P/D LOGIC ---
    rng = (df['High'] - df['Low']) + 1e-9
    pd_raw = (2 * df['Close'] - (df['High'] + df['Low'])) / rng
    v = 1.0 - pd_raw 
    u = df['High'] - df['High'].shift(1)
    d = df['Low'].shift(1) - df['Low']
    
    h_vals, phases = [], []
    for up, dw, val in zip(u, d, v):
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

    # --- PLOTTING ---
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.35, 0.15, 0.15, 0.15, 0.20])
    
    # 1. Price
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_30'], line=dict(color='orange', width=1.5), name="EMA 30"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_72'], line=dict(color='cyan', width=1.5), name="EMA 72"), row=1, col=1)
    
    # 2. P/D Scale
    c_map = {"rD":'red', "Ad":'orange', "rP":'lime', "Dp":'green', "Neutral":'gray'}
    colors = [c_map.get(p, 'gray') for p in df['Phase']]
    fig.add_trace(go.Bar(x=df.index, y=df['H_Val'], marker_color=colors, name="P/D"), row=2, col=1)

    # 3-5. Z-Scores
    fig.add_trace(go.Scatter(x=df.index, y=df['Z_EMA'], line=dict(color='yellow', width=2), mode='lines', name="Z-EMA"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Z_GOLD_BTC'], line=dict(color='magenta', width=2), mode='lines', name="Z-Gold"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Z_BTC_ETH'], line=dict(color='deepskyblue', width=2), mode='lines', name="Z-Alts"), row=5, col=1)

    fig.update_layout(height=1200, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- DATA TABLE ---
    st.subheader("📋 Data Output (Recent at Top)")
    # Show the tail (most recent) but sorted descending
    st.dataframe(df[['Close', 'EMA_30', 'EMA_72', 'H_Val', 'Z_EMA', 'Z_GOLD_BTC', 'Z_BTC_ETH', 'Phase']].sort_index(ascending=False).head(40), use_container_width=True)
