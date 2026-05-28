import sys
import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# --- INITIAL SETTINGS ---
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="Revelation Engine: Dashboard")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Engine Settings")
    ticker = st.text_input("Primary Ticker", value="BTC-USD").strip().upper()
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "1d"], index=3)
    period = st.selectbox("Data Period", ["7d", "1mo", "3mo", "1y"], index=1)
    
    st.divider()
    st.subheader("Statistical Parameters")
    ema_fast_val = st.number_input("Fast EMA", value=30)
    ema_slow_val = st.number_input("Slow EMA", value=72)
    z_win_val = st.number_input("Z-Score Window", value=20)

    if st.button("🔄 Clear Cache & Refresh"):
        st.cache_data.clear()
        st.rerun()

st.title(f"⚖️ Revelation Engine: {ticker}")
st.caption("Monitoring EMA Divergence, Gold/BTC, and BTC/ETH Statistical Extremes")

# --- ROBUST DATA FETCHING ---
@st.cache_data(ttl=600)
def fetch_all_data(main_t, tf, prd):
    try:
        # Fetching tickers individually for maximum stability
        m_data = yf.download(main_t, period=prd, interval=tf, auto_adjust=True, progress=False)
        p_data = yf.download("PAXG-USD", period=prd, interval=tf, auto_adjust=True, progress=False)
        b_data = yf.download("BTC-USD", period=prd, interval=tf, auto_adjust=True, progress=False)
        e_data = yf.download("ETH-USD", period=prd, interval=tf, auto_adjust=True, progress=False)
        
        if m_data.empty:
            return None, None, None, f"Ticker {main_t} returned no data."

        # Flatten MultiIndex columns if they exist
        for d in [m_data, p_data, b_data, e_data]:
            if not d.empty and isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)

        # Align series to primary index
        target_idx = m_data.index
        p_c = p_data['Close'].reindex(target_idx).ffill().bfill()
        b_c = b_data['Close'].reindex(target_idx).ffill().bfill()
        e_c = e_data['Close'].reindex(target_idx).ffill().bfill()

        # Calculate Ratios
        paxg_btc_ratio = p_c / (b_c + 1e-9)
        btc_eth_ratio = b_c / (e_c + 1e-9)
        
        return m_data, paxg_btc_ratio, btc_eth_ratio, None
    except Exception as e:
        return None, None, None, str(e)

# EXECUTE DATA FETCH
df, pb_ratio, be_ratio, err = fetch_all_data(ticker, timeframe, period)

if err:
    st.error(f"❌ Connection Error: {err}")
elif df is not None:
    try:
        # --- 1. EMA DIVERGENCE (30/72) ---
        df['EMA_30'] = ta.ema(df['Close'], length=ema_fast_val)
        df['EMA_72'] = ta.ema(df['Close'], length=ema_slow_val)
        df['Spread'] = df['EMA_30'] - df['EMA_72']
        
        # Manual Z-Score for stability
        df['Z_EMA'] = (df['Spread'] - df['Spread'].rolling(z_win_val).mean()) / (df['Spread'].rolling(z_win_val).std() + 1e-9)

        # --- 2. INTERMARKET Z-SCORES ---
        def get_z(series, win):
            return (series - series.rolling(win).mean()) / (series.rolling(win).std() + 1e-9)

        df['Z_GOLD_BTC'] = get_z(pb_ratio, z_win_val)
        df['Z_BTC_ETH'] = get_z(be_ratio, z_win_val)

        # --- 3. REVELATION P/D LOGIC ---
        rng = df['High'] - df['Low']
        pd_val = (2 * df['Close'] - (df['High'] + df['Low'])) / (rng + 1e-9)
        v, u, d = (1.0 - pd_val), (df['High'] - df['High'].shift(1)), (df['Low'].shift(1) - df['Low'])
        
        df['Phase'] = [("Dp" if val > 1.0 else "rP") if up > dw and up > 0 else 
                       ("rD" if val > 1.0 else "Ad") if dw > up and dw > 0 else "Neutral" 
                       for up, dw, val in zip(u, d, v)]

        # --- 4. VISUALIZATION (5 ROWS) ---
        fig = make_subplots(
            rows=5, cols=1, shared_xaxes=True, 
            vertical_spacing=0.02, 
            row_heights=[0.35, 0.1, 0.18, 0.18, 0.19],
            subplot_titles=(
                f"{ticker} Price Action", "Revelation P/D Scale", 
                "EMA 30/72 Z-Score", "Gold/BTC Z-Score", "BTC/ETH Z-Score"
            )
        )

        # Main Price
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_30'], line=dict(color='orange', width=1.2), name=f"EMA {ema_fast_val}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_72'], line=dict(color='cyan', width=1.2), name=f"EMA {ema_slow_val}"), row=1, col=1)

        # Phase Bar
        c_map = {"rD": 'firebrick', "Ad": 'orange', "rP": 'limegreen', "Dp": 'darkgreen', "Neutral": 'gray'}
        colors = [c_map.get(p, 'gray') for p in df['Phase']]
        fig.add_trace(go.Bar(x=df.index, y=(df['High']-df['Low']), marker_color=colors, name="Phase"), row=2, col=1)

        # Z-Scores
        fig.add_trace(go.Scatter(x=df.index, y=df['Z_EMA'], line=dict(color='yellow', width=2), mode='lines', name="Z-EMA"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Z_GOLD_BTC'], line=dict(color='magenta', width=2), mode='lines', name="Z-Gold"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Z_BTC_ETH'], line=dict(color='deepskyblue', width=2), mode='lines', name="Z-Alts"), row=5, col=1)

        # Reference Lines
        for r in [3, 4, 5]:
            fig.add_hline(y=2.0, line_dash="dash", line_color="white", row=r, col=1, opacity=0.3)
            fig.add_hline(y=-2.0, line_dash="dash", line_color="white", row=r, col=1, opacity=0.3)
            fig.add_hline(y=0, line_width=1, line_color="gray", row=r, col=1, opacity=0.5)

        fig.update_layout(height=1200, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- 5. DATA TABLE (UPDATED WITH EMA VALUES) ---
        st.divider()
        st.subheader("📋 Output Data Table")
        
        # We select the requested columns: Close, EMAs, and Z-Scores
        display_df = df[[
            'Close', 
            'EMA_30', 
            'EMA_72', 
            'Z_EMA', 
            'Z_GOLD_BTC', 
            'Z_BTC_ETH', 
            'Phase'
        ]].copy()
        
        # Rounding for cleanliness
        display_df = display_df.round(4)
        
        # Sort by most recent candle first
        st.dataframe(display_df.sort_index(ascending=False), use_container_width=True)

    except Exception as e:
        st.error(f"Logic Processing Error: {str(e)}")
