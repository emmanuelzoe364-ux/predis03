import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# --- INITIAL SETTINGS ---
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="Revelation Engine")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Primary Ticker", value="BTC-USD").strip().upper()
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "1d"], index=3)
    period = st.selectbox("Data Period", ["7d", "1mo", "3mo", "1y"], index=1)
    
    st.divider()
    ema_fast_val = st.number_input("Fast EMA", value=30)
    ema_slow_val = st.number_input("Slow EMA", value=72)
    z_win_val = st.number_input("Z-Score Window", value=20)

    st.divider()
    st.subheader("Data Status")
    status_box = st.empty()

    if st.button("🔄 Force Refresh"):
        st.cache_data.clear()
        st.rerun()

# --- ROBUST DATA ENGINE ---
@st.cache_data(ttl=300)
def fetch_and_align(main_t, tf, prd):
    try:
        # Fetch tickers
        assets = {
            "Main": main_t,
            "PAXG": "PAXG-USD",
            "BTC": "BTC-USD",
            "ETH": "ETH-USD"
        }
        
        data_frames = {}
        for key, sym in assets.items():
            d = yf.download(sym, period=prd, interval=tf, auto_adjust=True, progress=False)
            if d.empty:
                return None, f"Failed to fetch {sym}. Yahoo may be rate-limiting. Try 1h timeframe."
            
            # Flatten MultiIndex if necessary
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            data_frames[key] = d

        # Standardize and Concatenate (The Fix for the loading issue)
        # We align all data by the 'Main' ticker's time index
        main_df = data_frames["Main"].copy()
        main_df['PAXG_C'] = data_frames["PAXG"]['Close'].reindex(main_df.index).ffill().bfill()
        main_df['BTC_C'] = data_frames["BTC"]['Close'].reindex(main_df.index).ffill().bfill()
        main_df['ETH_C'] = data_frames["ETH"]['Close'].reindex(main_df.index).ffill().bfill()

        return main_df, None
    except Exception as e:
        return None, str(e)

# EXECUTE
df_raw, error_msg = fetch_and_align(ticker, timeframe, period)

if error_msg:
    status_box.error(error_msg)
    st.error(f"Engine Error: {error_msg}")
elif df_raw is not None:
    status_box.success("Connection: OK")
    df = df_raw.copy()

    # --- MATH SECTION ---
    # EMA Calculation
    df['EMA_30'] = df['Close'].ewm(span=ema_fast_val, adjust=False).mean()
    df['EMA_72'] = df['Close'].ewm(span=ema_slow_val, adjust=False).mean()
    
    # Z-Scores
    df['Spread'] = df['EMA_30'] - df['EMA_72']
    df['Z_EMA'] = (df['Spread'] - df['Spread'].rolling(z_win_val).mean()) / (df['Spread'].rolling(z_win_val).std() + 1e-9)
    
    ratio_pb = df['PAXG_C'] / (df['BTC_C'] + 1e-9)
    df['Z_GOLD_BTC'] = (ratio_pb - ratio_pb.rolling(z_win_val).mean()) / (ratio_pb.rolling(z_win_val).std() + 1e-9)
    
    ratio_be = df['BTC_C'] / (df['ETH_C'] + 1e-9)
    df['Z_BTC_ETH'] = (ratio_be - ratio_be.rolling(z_win_val).mean()) / (ratio_be.rolling(z_win_val).std() + 1e-9)

    # Revelation Phase Logic
    rng = (df['High'] - df['Low']) + 1e-9
    pd_scale = 1.0 - ((2 * df['Close'] - (df['High'] + df['Low'])) / rng)
    u = df['High'] - df['High'].shift(1)
    d = df['Low'].shift(1) - df['Low']
    
    h_vals, phases = [], []
    for up, dw, val in zip(u, d, pd_scale):
        if up > dw and up > 0:
            phases.append("Dp" if val > 1.0 else "rP")
            h_vals.append(val)
        elif dw > up and dw > 0:
            phases.append("rD" if val > 1.0 else "Ad")
            h_vals.append(-val)
        else:
            phases.append("Neutral")
            h_vals.append(0)
    
    df['H_Val'] = h_vals
    df['Phase'] = phases

    # --- PLOTTING ---
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.35, 0.15, 0.15, 0.15, 0.20])
    
    # Candle + EMAs
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_30'], line=dict(color='orange', width=1.5), name="EMA 30"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_72'], line=dict(color='cyan', width=1.5), name="EMA 72"), row=1, col=1)
    
    # P/D Scale
    c_map = {"rD":'#FF0000', "Ad":'#FFA500', "rP":'#00FF00', "Dp":'#006400', "Neutral":'gray'}
    colors = [c_map.get(p, 'gray') for p in df['Phase']]
    fig.add_trace(go.Bar(x=df.index, y=df['H_Val'], marker_color=colors, name="P/D Scale"), row=2, col=1)

    # Z-Scores
    fig.add_trace(go.Scatter(x=df.index, y=df['Z_EMA'], line=dict(color='yellow', width=2), name="Z-EMA"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Z_GOLD_BTC'], line=dict(color='magenta', width=2), name="Z-Gold"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Z_BTC_ETH'], line=dict(color='deepskyblue', width=2), name="Z-Alts"), row=5, col=1)

    # Styling
    fig.update_layout(height=1100, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- DATA TABLE (THE MASTER OUTPUT) ---
    st.divider()
    st.subheader("📋 Revelation Output Data")
    
    # Rounding and formatting for the final display
    table_df = df[['Close', 'EMA_30', 'EMA_72', 'H_Val', 'Z_EMA', 'Z_GOLD_BTC', 'Z_BTC_ETH', 'Phase']].copy()
    table_df = table_df.round(4)
    
    # Show it
    st.dataframe(table_df.sort_index(ascending=False), use_container_width=True)
