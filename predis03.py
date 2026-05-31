import sys
import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# --- SUPPRESS WARNINGS ---
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Revelation Engine: YF Optimized")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    primary_ticker = st.text_input("Primary Ticker", value="BTC-USD").strip().upper()
    
    # Intraday data is best limited to '1mo' for consistency
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "1d"], index=3)
    period = st.selectbox("Period", ["7d", "1mo", "3mo", "1y"], index=1)
    
    st.divider()
    st.subheader("Stats Settings")
    ema_fast_val = st.number_input("Fast EMA", value=30)
    ema_slow_val = st.number_input("Slow EMA", value=72)
    z_win_val = st.number_input("Z-Score Window", value=20)

    if st.button("🔄 Clear & Refresh"):
        st.cache_data.clear()
        st.rerun()

st.title(f"⚖️ Revelation Engine")
st.caption("Standardized YFinance Handling | Intermarket Divergence")

# --- DATA FETCHING (INTERNAL YF HANDLING) ---
@st.cache_data(ttl=600)
def fetch_engine_data(main_t, tf, prd):
    try:
        # Fetching each ticker individually without passing a custom session
        # This allows yfinance to use its own curl_cffi or default internal downloader
        m_df = yf.download(main_t, period=prd, interval=tf, progress=False, auto_adjust=True)
        paxg = yf.download("PAXG-USD", period=prd, interval=tf, progress=False, auto_adjust=True)
        btc = yf.download("BTC-USD", period=prd, interval=tf, progress=False, auto_adjust=True)
        eth = yf.download("ETH-USD", period=prd, interval=tf, progress=False, auto_adjust=True)

        if m_df.empty:
            return None, None, None, f"No data returned for {main_t}. Try a larger timeframe or check the symbol."

        # Flatten MultiIndex columns if they exist
        for d in [m_df, paxg, btc, eth]:
            if not d.empty and isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)

        # Align series to primary index
        target_idx = m_df.index
        p_c = paxg['Close'].reindex(target_idx).ffill().bfill()
        b_c = btc['Close'].reindex(target_idx).ffill().bfill()
        e_c = eth['Close'].reindex(target_idx).ffill().bfill()

        # Calculate Divergence Ratios
        r_gold_btc = p_c / (b_c + 1e-9)
        r_btc_eth = b_c / (e_c + 1e-9)

        return m_df, r_gold_btc, r_btc_eth, None
    except Exception as e:
        # Return 4 values to match the unpacker
        return None, None, None, str(e)

# RUN FETCH - Unpacking 4 values
df, pb_ratio, be_ratio, err = fetch_engine_data(primary_ticker, timeframe, period)

if err:
    st.error(f"❌ Connection Error: {err}")
elif df is not None:
    try:
        # --- 1. PRIMARY EMA DIVERGENCE (30/72) ---
        df['EMA_30'] = ta.ema(df['Close'], length=ema_fast_val)
        df['EMA_72'] = ta.ema(df['Close'], length=ema_slow_val)
        df['Spread'] = df['EMA_30'] - df['EMA_72']
        
        # Manual Z-Score
        df['Z_EMA'] = (df['Spread'] - df['Spread'].rolling(z_win_val).mean()) / (df['Spread'].rolling(z_win_val).std() + 1e-9)

        # --- 2. INTERMARKET Z-SCORES ---
        def get_z(series, win):
            return (series - series.rolling(win).mean()) / (series.rolling(win).std() + 1e-9)

        df['Z_GOLD_BTC'] = get_z(pb_ratio, z_win_val)
        df['Z_BTC_ETH'] = get_z(be_ratio, z_win_val)

        # --- 3. REVELATION P/D ---
        rng = df['High'] - df['Low']
        df['pd'] = (2 * df['Close'] - (df['High'] + df['Low'])) / (rng + 1e-9)
        df['pd_scale'] = 1.0 - df['pd']
        df['up'] = df['High'] - df['High'].shift(1)
        df['down'] = df['Low'].shift(1) - df['Low']

        def map_phase(row):
            u, d, v = row['up'], row['down'], row['pd_scale']
            if u > d and u > 0: return "Premium", ("rP" if v <= 1.0 else "Dp"), v
            elif d > u and d > 0: return "Discount", ("Ad" if v <= 1.0 else "rD"), -v
            return "Neutral", "Neutral", 0.0

        res = df.apply(map_phase, axis=1)
        df['Side'], df['Phase'], df['H_Val'] = [x[0] for x in res], [x[1] for x in res], [x[2] for x in res]

        # --- 4. VISUALIZATION (5 ROWS) ---
        fig = make_subplots(
            rows=5, cols=1, shared_xaxes=True, 
            vertical_spacing=0.02, 
            row_heights=[0.35, 0.1, 0.18, 0.18, 0.19],
            subplot_titles=("Price Action (30/72 EMAs)", "Revelation P/D Scale", "EMA Divergence Z-Score", "Gold/BTC Z-Score", "BTC/Eth Z-Score")
        )

        # Price
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_30'], line=dict(color='orange', width=1.5), name="30 EMA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_72'], line=dict(color='cyan', width=1.5), name="72 EMA"), row=1, col=1)

        # P/D Bar
        c_map = {"rD": 'firebrick', "Ad": 'orange', "rP": 'limegreen', "Dp": 'darkgreen', "Neutral": 'gray'}
        colors = [c_map.get(p, 'gray') for p in df['Phase']]
        fig.add_trace(go.Bar(x=df.index, y=df['H_Val'], marker_color=colors, name="P/D Scale"), row=2, col=1)

        # Z-Scores (Solid Lines)
        fig.add_trace(go.Scatter(x=df.index, y=df['Z_EMA'], line=dict(color='yellow', width=2), mode='lines', name="Z-EMA"), row=3, col=1)
        fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=3, col=1, opacity=0.4)
        fig.add_hline(y=-2.0, line_dash="dash", line_color="lime", row=3, col=1, opacity=0.4)

        fig.add_trace(go.Scatter(x=df.index, y=df['Z_GOLD_BTC'], line=dict(color='magenta', width=2), mode='lines', name="Z-Gold"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Z_BTC_ETH'], line=dict(color='deepskyblue', width=2), mode='lines', name="Z-Alts"), row=5, col=1)

        fig.update_layout(height=1200, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- 5. DATA TABLE (BOTTOM) ---
        st.divider()
        st.subheader("📋 Output Data")
        st.dataframe(df[['Close', 'Z_EMA', 'Z_GOLD_BTC', 'Z_BTC_ETH', 'Side', 'Phase']].sort_index(ascending=False), use_container_width=True)

    except Exception as e:
        st.error(f"Logic Error: {str(e)}")
