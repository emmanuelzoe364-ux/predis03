import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime

# --- INITIAL SETTINGS ---
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="Revelation Engine Pro")

# --- SIDEBAR & CACHE BUSTER ---
if 'refresh_count' not in st.session_state:
    st.session_state.refresh_count = 0

with st.sidebar:
    st.header("⚙️ Configuration")
    primary_ticker = st.text_input("Primary Ticker", value="BTC-USD").strip().upper()
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "1d"], index=0)
    period = st.selectbox("Period", ["1d", "3d", "7d", "1mo"], index=2)
    
    st.divider()
    ema_fast_val = st.number_input("Fast EMA", value=30)
    ema_slow_val = st.number_input("Slow EMA", value=72)
    z_win_val = st.number_input("Z-Score Window", value=20)

    if st.button("🚀 FORCE LIVE UPDATE"):
        st.cache_data.clear()
        st.session_state.refresh_count += 1
        st.rerun()

    st.write(f"Refreshed: {datetime.now().strftime('%H:%M:%S')} UTC")

st.title(f"⚖️ Revelation Engine")
st.caption("Cloud-Optimized | Native Math (No pandas_ta) | Live Sync")

# --- DATA FETCHING (CLOUD OPTIMIZED) ---
@st.cache_data(ttl=30) # 30 second cache for live feel
def fetch_cloud_data(main_t, tf, prd, refresh_id):
    try:
        tickers = [main_t, "PAXG-USD", "BTC-USD", "ETH-USD"]
        # Download as one block with prepost=True for latest ticks
        data = yf.download(
            tickers=tickers,
            period=prd,
            interval=tf,
            auto_adjust=True,
            prepost=True,
            threads=False,
            progress=False,
            group_by='ticker'
        )
        
        if data.empty:
            return None, None, None, "No data returned."

        # Extract & Align
        m_df = data[main_t].copy()
        idx = m_df.index
        p_c = data["PAXG-USD"]['Close'].reindex(idx).ffill().bfill()
        b_c = data["BTC-USD"]['Close'].reindex(idx).ffill().bfill()
        e_c = data["ETH-USD"]['Close'].reindex(idx).ffill().bfill()

        return m_df, (p_c/b_c), (b_c/e_c), None
    except Exception as e:
        return None, None, None, str(e)

# Execute Fetch with refresh_count to bust cache
df, pb_ratio, be_ratio, err = fetch_cloud_data(primary_ticker, timeframe, period, st.session_state.refresh_count)

if err:
    st.error(f"❌ Connection Error: {err}")
elif df is not None:
    try:
        # --- 1. NATIVE EMA CALC (No pandas_ta needed) ---
        df['EMA_30'] = df['Close'].ewm(span=ema_fast_val, adjust=False).mean()
        df['EMA_72'] = df['Close'].ewm(span=ema_slow_val, adjust=False).mean()
        df['Spread'] = df['EMA_30'] - df['EMA_72']
        
        # Z-Score
        df['Z_EMA'] = (df['Spread'] - df['Spread'].rolling(z_win_val).mean()) / (df['Spread'].rolling(z_win_val).std() + 1e-9)

        # --- 2. INTERMARKET Z-SCORES ---
        def get_z(series, win):
            return (series - series.rolling(win).mean()) / (series.rolling(win).std() + 1e-9)

        df['Z_GOLD_BTC'] = get_z(pb_ratio, z_win_val)
        df['Z_BTC_ETH'] = get_z(be_ratio, z_win_val)

        # --- 3. REVELATION P/D (-2 to 2 Scale) ---
        rng = (df['High'] - df['Low']) + 1e-9
        pd_raw = (2 * df['Close'] - (df['High'] + df['Low'])) / rng
        v = 1.0 - pd_raw # Internal Intensity 0 to 2
        
        u = df['High'] - df['High'].shift(1)
        d = df['Low'].shift(1) - df['Low']
        
        h_vals, phases = [], []
        for up, dw, val in zip(u, d, v):
            if up > dw and up > 0: # Premium
                phases.append("Dp" if val > 1.0 else "rP")
                h_vals.append(val)
            elif dw > up and dw > 0: # Discount
                phases.append("rD" if val > 1.0 else "Ad")
                h_vals.append(-val)
            else:
                phases.append("Neutral")
                h_vals.append(0)
        df['H_Val'], df['Phase'] = h_vals, phases

        # --- 4. VISUALIZATION ---
        fig = make_subplots(
            rows=5, cols=1, shared_xaxes=True, 
            vertical_spacing=0.02, 
            row_heights=[0.35, 0.15, 0.15, 0.15, 0.20],
            subplot_titles=("Price & EMAs", "Revelation P/D Scale", "EMA Divergence Z", "Gold/BTC Z", "BTC/Eth Z")
        )

        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_30'], line=dict(color='orange', width=1.5), name="EMA 30"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_72'], line=dict(color='cyan', width=1.5), name="EMA 72"), row=1, col=1)

        c_map = {"rD": 'red', "Ad": 'orange', "rP": 'lime', "Dp": 'green', "Neutral": 'gray'}
        colors = [c_map.get(p, 'gray') for p in df['Phase']]
        fig.add_trace(go.Bar(x=df.index, y=df['H_Val'], marker_color=colors, name="P/D"), row=2, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['Z_EMA'], line=dict(color='yellow', width=2), mode='lines', name="Z-EMA"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Z_GOLD_BTC'], line=dict(color='magenta', width=2), mode='lines', name="Z-Gold"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Z_BTC_ETH'], line=dict(color='deepskyblue', width=2), mode='lines', name="Z-Alts"), row=5, col=1)

        fig.update_layout(height=1200, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- 5. DATA TABLE (REcent at Top) ---
        st.divider()
        st.subheader("📋 Output Data Table")
        # Included EMA_30 and EMA_72 in display columns
        cols = ['Close', 'EMA_30', 'EMA_72', 'H_Val', 'Z_EMA', 'Z_GOLD_BTC', 'Z_BTC_ETH', 'Phase']
        st.dataframe(df[cols].sort_index(ascending=False).head(50), use_container_width=True)

    except Exception as e:
        st.error(f"Logic Error: {str(e)}")
