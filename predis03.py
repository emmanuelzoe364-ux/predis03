import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import warnings
from datetime import datetime
import time
import random

# --- INITIAL SETTINGS ---
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="Revelation Engine: Cloud-Bypass")

# --- SIDEBAR & CACHE BUSTER ---
if 'refresh_count' not in st.session_state:
    st.session_state.refresh_count = 0

with st.sidebar:
    st.header("⚙️ Configuration")
    primary_ticker = st.text_input("Primary Ticker", value="BTC-USD").strip().upper()
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "1d"], index=0)
    
    st.divider()
    ema_fast_val = st.number_input("Fast EMA", value=30)
    ema_slow_val = st.number_input("Slow EMA", value=72)
    z_win_val = st.number_input("Z-Score Window", value=20)

    if st.button("🔥 FORCE LIVE SYNC"):
        st.cache_data.clear()
        st.session_state.refresh_count += 1
        st.rerun()

    st.write(f"Cloud Status: **Syncing...**")
    st.write(f"Last Refresh: {datetime.now().strftime('%H:%M:%S')} UTC")

# --- DIRECT API FETCH ENGINE (Bypasses yfinance) ---
def fetch_direct_api(ticker, interval):
    """Fetches data directly from Yahoo Query API with a cache-busting timestamp."""
    try:
        # Range logic based on interval
        range_str = "1d" if interval in ["1m", "5m"] else "7d"
        # Cache Buster: Random digit to trick the cloud proxy
        cb = random.randint(1000, 9999)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval={interval}&range={range_str}&cache_buster={cb}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        json_data = response.json()
        
        # Parsing JSON structure
        result = json_data['chart']['result'][0]
        timestamps = result['timestamp']
        ohlc = result['indicators']['quote'][0]
        adj_close = result['indicators']['adjclose'][0]['adjclose']
        
        df = pd.DataFrame({
            'Open': ohlc['open'],
            'High': ohlc['high'],
            'Low': ohlc['low'],
            'Close': adj_close
        }, index=pd.to_datetime(timestamps, unit='s'))
        
        return df.ffill()
    except:
        return None

@st.cache_data(ttl=10)
def fetch_all_layers_direct(main_t, tf, refresh_id):
    # Fetch all 4 assets directly
    m_df = fetch_direct_api(main_t, tf)
    paxg = fetch_direct_api("PAXG-USD", tf)
    btc = fetch_direct_api("BTC-USD", tf)
    eth = fetch_direct_api("ETH-USD", tf)
    
    if m_df is None:
        return None, None, None, "API connection failed. Yahoo is throttling this cloud server."
        
    idx = m_df.index
    def align(source_df):
        if source_df is not None:
            return source_df['Close'].reindex(idx).ffill().bfill()
        return pd.Series(np.nan, index=idx)
    
    p_c = align(paxg)
    b_c = align(btc)
    e_c = align(eth)
    
    return m_df, (p_c/(b_c+1e-9)), (b_c/(e_c+1e-9)), None

# Execute
df, pb_ratio, be_ratio, err = fetch_all_layers_direct(primary_ticker, timeframe, st.session_state.refresh_count)

if err:
    st.error(err)
elif df is not None:
    # --- MATH SECTION (Native) ---
    df['EMA_30'] = df['Close'].ewm(span=ema_fast_val, adjust=False).mean()
    df['EMA_72'] = df['Close'].ewm(span=ema_slow_val, adjust=False).mean()
    df['Spread'] = df['EMA_30'] - df['EMA_72']
    df['Z_EMA'] = (df['Spread'] - df['Spread'].rolling(z_win_val).mean()) / (df['Spread'].rolling(z_win_val).std() + 1e-9)

    def get_z(ser, win):
        return (ser - ser.rolling(win).mean()) / (ser.rolling(win).std() + 1e-9)
    
    df['Z_GOLD_BTC'] = get_z(pb_ratio, z_win_val)
    df['Z_BTC_ETH'] = get_z(be_ratio, z_win_val)

    # --- REVELATION PHASE LOGIC (-2 to 2) ---
    rng = (df['High'] - df['Low']) + 1e-9
    pd_v = 1.0 - ((2 * df['Close'] - (df['High'] + df['Low'])) / rng)
    u, d = (df['High'] - df['High'].shift(1)), (df['Low'].shift(1) - df['Low'])
    
    h_vals, phases = [], []
    for up, dw, val in zip(u, d, pd_v):
        if up > dw and up > 0:
            phases.append("Dp" if val > 1.0 else "rP"); h_vals.append(val)
        elif dw > up and dw > 0:
            phases.append("rD" if val > 1.0 else "Ad"); h_vals.append(-val)
        else:
            phases.append("Neutral"); h_vals.append(0)
    df['H_Val'], df['Phase'] = h_vals, phases

    # --- VISUALIZATION ---
    st.title(f"⚖️ {primary_ticker} Revelation")
    
    # Show the latest candle time as a big metric to confirm it is live
    last_ts = df.index[-1].strftime('%H:%M:%S')
    st.metric("Latest Data Timestamp (UTC)", last_ts)

    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02, 
                        row_heights=[0.35, 0.15, 0.15, 0.15, 0.20])

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_30'], line=dict(color='orange', width=1.5), name="EMA 30"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_72'], line=dict(color='cyan', width=1.5), name="EMA 72"), row=1, col=1)

    c_map = {"rD":'red', "Ad":'orange', "rP":'lime', "Dp":'green', "Neutral":'gray'}
    fig.add_trace(go.Bar(x=df.index, y=df['H_Val'], marker_color=[c_map.get(p, 'gray') for p in df['Phase']], name="P/D"), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['Z_EMA'], line=dict(color='yellow'), name="Z-EMA"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Z_GOLD_BTC'], line=dict(color='magenta'), name="Z-Gold"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Z_BTC_ETH'], line=dict(color='deepskyblue'), name="Z-Alts"), row=5, col=1)

    fig.update_layout(height=1100, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- TABLE (EMAs included) ---
    st.subheader("📋 Output Data Table")
    cols = ['Close', 'EMA_30', 'EMA_72', 'H_Val', 'Z_EMA', 'Z_GOLD_BTC', 'Z_BTC_ETH', 'Phase']
    st.dataframe(df[cols].sort_index(ascending=False).head(50), use_container_width=True)
