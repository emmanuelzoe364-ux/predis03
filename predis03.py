import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# --- SETTINGS ---
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="Revelation Engine")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    ticker = st.text_input("Primary Ticker", value="BTC-USD").strip().upper()
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "1d"], index=3)
    period = st.selectbox("Data Period", ["7d", "1mo", "3mo", "1y"], index=1)
    
    st.divider()
    ema_fast_val = st.number_input("Fast EMA", value=30)
    ema_slow_val = st.number_input("Slow EMA", value=72)
    z_win_val = st.number_input("Z-Score Window", value=20)

    if st.button("🔄 Force Refresh"):
        st.cache_data.clear()
        st.rerun()

st.title(f"⚖️ Revelation Engine: {ticker}")

# --- DATA FETCHING ---
@st.cache_data(ttl=600)
def fetch_data(main_t, tf, prd):
    try:
        # Download tickers individually for stability
        m_df = yf.download(main_t, period=prd, interval=tf, auto_adjust=True, progress=False)
        p_df = yf.download("PAXG-USD", period=prd, interval=tf, auto_adjust=True, progress=False)
        b_df = yf.download("BTC-USD", period=prd, interval=tf, auto_adjust=True, progress=False)
        e_df = yf.download("ETH-USD", period=prd, interval=tf, auto_adjust=True, progress=False)
        
        if m_df.empty: return None, None, None, "No Data"

        # Flatten columns
        for d in [m_df, p_df, b_df, e_df]:
            if not d.empty and isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)

        # Align to primary index
        idx = m_df.index
        p_c = p_df['Close'].reindex(idx).ffill().bfill()
        b_c = b_df['Close'].reindex(idx).ffill().bfill()
        e_c = e_df['Close'].reindex(idx).ffill().bfill()

        return m_df, (p_c/b_c), (b_c/e_c), None
    except Exception as e:
        return None, None, None, str(e)

df, pb_ratio, be_ratio, err = fetch_data(ticker, timeframe, period)

if err:
    st.error(f"Error: {err}")
elif df is not None:
    # --- MATH SECTION (Native Pandas - No pandas_ta required) ---
    # EMA Calculation using EWM (Exponential Weighted Moving Average)
    df['EMA_30'] = df['Close'].ewm(span=ema_fast_val, adjust=False).mean()
    df['EMA_72'] = df['Close'].ewm(span=ema_slow_val, adjust=False).mean()
    
    # EMA Divergence Z-Score
    df['Spread'] = df['EMA_30'] - df['EMA_72']
    df['Z_EMA'] = (df['Spread'] - df['Spread'].rolling(z_win_val).mean()) / (df['Spread'].rolling(z_win_val).std() + 1e-9)
    
    # Intermarket Z-Scores
    df['Z_GOLD_BTC'] = (pb_ratio - pb_ratio.rolling(z_win_val).mean()) / (pb_ratio.rolling(z_win_val).std() + 1e-9)
    df['Z_BTC_ETH'] = (be_ratio - be_ratio.rolling(z_win_val).mean()) / (be_ratio.rolling(z_win_val).std() + 1e-9)

    # --- ORIGINAL REVELATION P/D LOGIC ---
    rng = (df['High'] - df['Low']) + 1e-9
    pd_raw = (2 * df['Close'] - (df['High'] + df['Low'])) / rng
    v = 1.0 - pd_raw # Scale 0 to 2
    
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

    # --- VISUALIZATION ---
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.35, 0.15, 0.15, 0.15, 0.20])
    
    # 1. Price + EMAs
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_30'], line=dict(color='orange', width=1.5), name="EMA 30"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_72'], line=dict(color='cyan', width=1.5), name="EMA 72"), row=1, col=1)
    
    # 2. Original P/D Scale
    c_map = {"rD":'red', "Ad":'orange', "rP":'lime', "Dp":'green', "Neutral":'gray'}
    colors = [c_map.get(p, 'gray') for p in df['Phase']]
    fig.add_trace(go.Bar(x=df.index, y=df['H_Val'], marker_color=colors, name="P/D"), row=2, col=1)

    # 3-5. Z-Scores (Solid Lines)
    fig.add_trace(go.Scatter(x=df.index, y=df['Z_EMA'], line=dict(color='yellow', width=2), mode='lines', name="Z-EMA"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Z_GOLD_BTC'], line=dict(color='magenta', width=2), mode='lines', name="Z-Gold"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Z_BTC_ETH'], line=dict(color='deepskyblue', width=2), mode='lines', name="Z-Alts"), row=5, col=1)

    # Styling
    for r in [3,4,5]:
        fig.add_hline(y=2.0, line_dash="dash", line_color="white", row=r, col=1, opacity=0.3)
        fig.add_hline(y=-2.0, line_dash="dash", line_color="white", row=r, col=1, opacity=0.3)

    fig.update_layout(height=1200, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- DATA TABLE ---
    st.subheader("📋 Output Data")
    # Added EMA_30 and EMA_72 to the dataframe view
    st.dataframe(df[['Close', 'EMA_30', 'EMA_72', 'H_Val', 'Z_EMA', 'Z_GOLD_BTC', 'Z_BTC_ETH', 'Phase']].sort_index(ascending=False), use_container_width=True)
