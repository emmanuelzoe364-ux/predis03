import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import warnings
from datetime import datetime
import time

# --- INITIAL SETTINGS ---
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="Revelation Engine Cloud")

# --- SIDEBAR & REFRESH ---
if 'refresh_count' not in st.session_state:
    st.session_state.refresh_count = 0

with st.sidebar:
    st.header("⚙️ Cloud Configuration")
    primary_ticker = st.text_input("Ticker (Yahoo Format)", value="BTC-USD").strip().upper()
    timeframe = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=1)
    
    st.divider()
    ema_fast_val = st.number_input("Fast EMA", value=30)
    ema_slow_val = st.number_input("Slow EMA", value=72)
    z_win_val = st.number_input("Z-Score Window", value=20)

    if st.button("🚀 FORCE CLOUD UPDATE"):
        st.cache_data.clear()
        st.session_state.refresh_count += 1
        st.rerun()

# --- DIRECT YAHOO API ENGINE (CLOUD COMPATIBLE) ---
@st.cache_data(ttl=30)
def fetch_direct(ticker, interval, refresh_id):
    try:
        # Range adjustment for small intervals
        range_str = "1d" if interval in ["1m", "5m"] else "7d" if interval != "1d" else "1y"
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval={interval}&range={range_str}"
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/110.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        json_data = response.json()
        
        result = json_data['chart']['result'][0]
        df = pd.DataFrame({
            'Open': result['indicators']['quote'][0]['open'],
            'High': result['indicators']['quote'][0]['high'],
            'Low': result['indicators']['quote'][0]['low'],
            'Close': result['indicators']['quote'][0]['close']
        }, index=pd.to_datetime(result['timestamp'], unit='s'))
        
        return df.ffill().dropna()
    except:
        return pd.DataFrame()

# Fetch All Layers
main_df = fetch_direct(primary_ticker, timeframe, st.session_state.refresh_count)
paxg_df = fetch_direct("PAXG-USD", timeframe, st.session_state.refresh_count)
btc_ref = fetch_direct("BTC-USD", timeframe, st.session_state.refresh_count)
eth_df = fetch_direct("ETH-USD", timeframe, st.session_state.refresh_count)

# --- VALIDATION & PROCESSING ---
if main_df.empty:
    st.error(f"❌ Could not reach Yahoo Data. Wait 30 seconds and refresh.")
else:
    # ALIGNMENT
    idx = main_df.index
    p_c = paxg_df['Close'].reindex(idx).ffill().bfill() if not paxg_df.empty else pd.Series(0, index=idx)
    b_c = btc_ref['Close'].reindex(idx).ffill().bfill() if not btc_ref.empty else pd.Series(1, index=idx)
    e_c = eth_df['Close'].reindex(idx).ffill().bfill() if not eth_df.empty else pd.Series(1, index=idx)
    
    pb_ratio = p_c / (b_c + 1e-9)
    be_ratio = b_c / (e_c + 1e-9)

    # --- MATH SECTION (Native Pandas) ---
    main_df['EMA_30'] = main_df['Close'].ewm(span=ema_fast_val, adjust=False).mean()
    main_df['EMA_72'] = main_df['Close'].ewm(span=ema_slow_val, adjust=False).mean()
    main_df['Spread'] = main_df['EMA_30'] - main_df['EMA_72']
    
    def get_z(ser, win):
        return (ser - ser.rolling(win).mean()) / (ser.rolling(win).std() + 1e-9)
    
    main_df['Z_EMA'] = get_z(main_df['Spread'], z_win_val)
    main_df['Z_GOLD_BTC'] = get_z(pb_ratio, z_win_val)
    main_df['Z_BTC_ETH'] = get_z(be_ratio, z_win_val)

    # REVELATION PHASE
    rng_p = (main_df['High'] - main_df['Low']) + 1e-9
    pd_v = 1.0 - ((2 * main_df['Close'] - (main_df['High'] + main_df['Low'])) / rng_p)
    u, d = (main_df['High'] - main_df['High'].shift(1)), (main_df['Low'].shift(1) - main_df['Low'])
    
    h_vals, phases = [], []
    for up, dw, val in zip(u, d, pd_v):
        if up > dw and up > 0:
            phases.append("Dp" if val > 1.0 else "rP"); h_vals.append(val)
        elif dw > up and dw > 0:
            phases.append("rD" if val > 1.0 else "Ad"); h_vals.append(-val)
        else:
            phases.append("Neutral"); h_vals.append(0)
    main_df['H_Val'], main_df['Phase'] = h_vals, phases

    # --- VISUALS ---
    st.title(f"⚖️ {primary_ticker} Engine")
    st.metric("Latest Data Time (UTC)", main_df.index[-1].strftime('%H:%M:%S'))

    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02, 
                        row_heights=[0.35, 0.15, 0.15, 0.15, 0.20])

    # Enhanced Visibility Candles
    fig.add_trace(go.Candlestick(
        x=main_df.index, 
        open=main_df['Open'], high=main_df['High'], low=main_df['Low'], close=main_df['Close'], 
        name="Price",
        increasing_line_width=2, decreasing_line_width=2
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['EMA_30'], line=dict(color='orange', width=2), name="EMA 30"), row=1, col=1)
    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['EMA_72'], line=dict(color='cyan', width=2), name="EMA 72"), row=1, col=1)

    c_map = {"rD":'red', "Ad":'orange', "rP":'lime', "Dp":'green', "Neutral":'gray'}
    fig.add_trace(go.Bar(x=main_df.index, y=main_df['H_Val'], marker_color=[c_map.get(p, 'gray') for p in main_df['Phase']]), row=2, col=1)

    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['Z_EMA'], line=dict(color='yellow'), name="Z-EMA"), row=3, col=1)
    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['Z_GOLD_BTC'], line=dict(color='magenta'), name="Z-Gold"), row=4, col=1)
    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['Z_BTC_ETH'], line=dict(color='deepskyblue'), name="Z-Alts"), row=5, col=1)

    fig.update_layout(height=1100, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- DATA TABLE ---
    st.subheader("📋 Output Data Table (Recent at Top)")
    cols = ['Close', 'EMA_30', 'EMA_72', 'H_Val', 'Z_EMA', 'Z_GOLD_BTC', 'Z_BTC_ETH', 'Phase']
    st.dataframe(main_df[cols].sort_index(ascending=False).head(50), use_container_width=True)
