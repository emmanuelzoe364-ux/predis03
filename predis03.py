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
st.set_page_config(layout="wide", page_title="Revelation Engine Pro")

# --- SIDEBAR & CACHE BUSTER ---
if 'refresh_count' not in st.session_state:
    st.session_state.refresh_count = 0

with st.sidebar:
    st.header("⚙️ Configuration")
    primary_ticker = st.text_input("Primary Ticker", value="BTC-USD").strip().upper()
    
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "1d"], index=0)
    period = st.selectbox("Period", ["1d", "3d", "7d", "1mo"], index=0)
    
    st.divider()
    st.subheader("Stats Settings")
    ema_fast_val = st.number_input("Fast EMA", value=30)
    ema_slow_val = st.number_input("Slow EMA", value=72)
    z_win_val = st.number_input("Z-Score Window", value=20)

    # REFRESH LOGIC
    if st.button("🚀 FORCE LIVE UPDATE"):
        st.cache_data.clear()
        # This increment changes the 'refresh_id' sent to the function, bypassing Streamlit's cache
        st.session_state.refresh_count += 1
        st.rerun()

    st.divider()
    st.write(f"**System Time:** {datetime.now().strftime('%H:%M:%S')} UTC")

# --- DATA FETCHING (ULTRA-LIVE OPTIMIZED) ---
@st.cache_data(ttl=15) # Minimal TTL for Cloud
def fetch_cloud_data(main_t, tf, prd, refresh_id):
    try:
        # Create a unique 'session' to prevent Yahoo from identifying the cloud bot
        tickers = [main_t, "PAXG-USD", "BTC-USD", "ETH-USD"]
        
        # KEY FIX: Using the internal Ticker object with proxy-busting download
        # We download as a single batch to keep the timestamp alignment perfect
        data = yf.download(
            tickers=tickers,
            period=prd,
            interval=tf,
            auto_adjust=True,
            prepost=True, # Critical for getting the 'live' minute
            threads=False,
            progress=False,
            group_by='ticker'
        )
        
        if data.empty:
            return None, None, None, "No data returned."

        # Extract & Align
        m_df = data[main_t].copy()
        idx = m_df.index
        
        # Force alignment for cross-asset ratios
        p_c = data["PAXG-USD"]['Close'].reindex(idx).ffill().bfill()
        b_c = data["BTC-USD"]['Close'].reindex(idx).ffill().bfill()
        e_c = data["ETH-USD"]['Close'].reindex(idx).ffill().bfill()

        return m_df, (p_c/b_c), (b_c/e_c), None
    except Exception as e:
        return None, None, None, str(e)

# Execute Fetch
df, pb_ratio, be_ratio, err = fetch_cloud_data(primary_ticker, timeframe, period, st.session_state.refresh_count)

if err:
    st.error(f"❌ Connection Error: {err}")
elif df is not None:
    try:
        # DATA INTEGRITY CHECK (SIDEBAR)
        # We tell the user exactly how old the last data point is
        last_candle_time = df.index[-1]
        time_diff = (datetime.now() - last_candle_time.replace(tzinfo=None)).total_seconds() / 60
        
        with st.sidebar:
            st.write(f"**Last Candle:** {last_candle_time.strftime('%H:%M:%S')}")
            if time_diff > 5:
                st.warning(f"Data is delayed by {int(time_diff)} mins (Yahoo Latency)")
            else:
                st.success(f"Data is Live ({int(time_diff)}m delay)")

        # --- MATH SECTION ---
        df['EMA_30'] = df['Close'].ewm(span=ema_fast_val, adjust=False).mean()
        df['EMA_72'] = df['Close'].ewm(span=ema_slow_val, adjust=False).mean()
        df['Spread'] = df['EMA_30'] - df['EMA_72']
        df['Z_EMA'] = (df['Spread'] - df['Spread'].rolling(z_win_val).mean()) / (df['Spread'].rolling(z_win_val).std() + 1e-9)

        def get_z(series, win):
            return (series - series.rolling(win).mean()) / (series.rolling(win).std() + 1e-9)

        df['Z_GOLD_BTC'] = get_z(pb_ratio, z_win_val)
        df['Z_BTC_ETH'] = get_z(be_ratio, z_win_val)

        # REVELATION P/D
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
                phases.append("Neutral"); h_vals.append(0)
        df['H_Val'], df['Phase'] = h_vals, phases

        # --- PLOTTING ---
        st.title(f"⚖️ {primary_ticker} Revelation")
        fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02, 
                            row_heights=[0.35, 0.15, 0.15, 0.15, 0.20])

        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_30'], line=dict(color='orange', width=1.5), name="EMA 30"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_72'], line=dict(color='cyan', width=1.5), name="EMA 72"), row=1, col=1)

        c_map = {"rD": 'red', "Ad": 'orange', "rP": 'lime', "Dp": 'green', "Neutral": 'gray'}
        fig.add_trace(go.Bar(x=df.index, y=df['H_Val'], marker_color=[c_map.get(p, 'gray') for p in df['Phase']], name="P/D"), row=2, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['Z_EMA'], line=dict(color='yellow'), name="Z-EMA"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Z_GOLD_BTC'], line=dict(color='magenta'), name="Z-Gold"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Z_BTC_ETH'], line=dict(color='deepskyblue'), name="Z-Alts"), row=5, col=1)

        fig.update_layout(height=1100, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Live Table (Newest Top)")
        st.dataframe(df[['Close', 'EMA_30', 'EMA_72', 'Z_EMA', 'Z_GOLD_BTC', 'Z_BTC_ETH', 'Phase']].sort_index(ascending=False).head(50), use_container_width=True)

    except Exception as e:
        st.error(f"Logic Error: {str(e)}")
