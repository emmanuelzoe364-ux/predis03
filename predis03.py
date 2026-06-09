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
st.set_page_config(layout="wide", page_title="Revelation Engine: Deep-Sync")

# --- SIDEBAR & CACHE BUSTER ---
if 'refresh_count' not in st.session_state:
    st.session_state.refresh_count = 0

with st.sidebar:
    st.header("⚙️ Configuration")
    primary_ticker = st.text_input("Primary Ticker", value="BTC-USD").strip().upper()
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h"], index=0)
    period = st.selectbox("Period", ["1d", "3d", "7d"], index=0)
    
    st.divider()
    ema_fast_val = st.number_input("Fast EMA", value=30)
    ema_slow_val = st.number_input("Slow EMA", value=72)
    z_win_val = st.number_input("Z-Score Window", value=20)

    if st.button("🔥 FORCE DEEP SYNC"):
        st.cache_data.clear()
        st.session_state.refresh_count += 1
        st.rerun()

    st.divider()
    st.subheader("📡 Live Status Console")
    status_box = st.empty()

# --- ROBUST INDIVIDUAL FETCH ENGINE ---
def get_asset_data(ticker_str, tf, prd):
    """Uses .history() for better cookie handling on cloud servers."""
    try:
        t_obj = yf.Ticker(ticker_str)
        # Fetching with prepost=True to get the absolute latest forming candle
        data = t_obj.history(period=prd, interval=tf, auto_adjust=True, prepost=True)
        if data.empty:
            return None
        return data[['Close', 'High', 'Low']]
    except:
        return None

@st.cache_data(ttl=15)
def fetch_all_layers(main_t, tf, prd, refresh_id):
    # 1. Fetch Primary
    m_df = get_asset_data(main_t, tf, prd)
    if m_df is None:
        return None, None, None, f"Failed to fetch {main_t}"
    
    # 2. Fetch Comparison Assets
    p_df = get_asset_data("PAXG-USD", tf, prd)
    b_df = get_asset_data("BTC-USD", tf, prd)
    e_df = get_asset_data("ETH-USD", tf, prd)

    # 3. Align & Ratio Calculation
    idx = m_df.index
    
    def sync_close(source_df, target_idx):
        if source_df is not None:
            return source_df['Close'].reindex(target_idx).ffill().bfill()
        return pd.Series(np.nan, index=target_idx)

    p_c = sync_close(p_df, idx)
    b_c = sync_close(b_df, idx)
    e_c = sync_close(e_df, idx)

    return m_df, (p_c / (b_c + 1e-9)), (b_c / (e_c + 1e-9)), None

# RUN FETCH
df, pb_ratio, be_ratio, err = fetch_all_layers(primary_ticker, timeframe, period, st.session_state.refresh_count)

if err:
    status_box.error(err)
    st.error(f"📡 Connection Issue: {err}")
elif df is not None:
    # --- LIVE INTEGRITY CHECK ---
    last_candle = df.index[-1]
    now_utc = datetime.utcnow()
    diff_sec = (now_utc - last_candle.replace(tzinfo=None)).total_seconds()
    
    with st.sidebar:
        if diff_sec < 120:
            status_box.success(f"LIVE: {int(diff_sec)}s lag")
        else:
            status_box.warning(f"DELAYED: {int(diff_sec/60)}m lag")
        st.write(f"System UTC: {now_utc.strftime('%H:%M:%S')}")
        st.write(f"Last Candle: {last_candle.strftime('%H:%M:%S')}")

    # --- MATH SECTION (Native Pandas) ---
    df['EMA_30'] = df['Close'].ewm(span=ema_fast_val, adjust=False).mean()
    df['EMA_72'] = df['Close'].ewm(span=ema_slow_val, adjust=False).mean()
    df['Spread'] = df['EMA_30'] - df['EMA_72']
    
    df['Z_EMA'] = (df['Spread'] - df['Spread'].rolling(z_win_val).mean()) / (df['Spread'].rolling(z_win_val).std() + 1e-9)

    def get_z(ser, win):
        return (ser - ser.rolling(win).mean()) / (ser.rolling(win).std() + 1e-9)
    
    df['Z_GOLD_BTC'] = get_z(pb_ratio, z_win_val)
    df['Z_BTC_ETH'] = get_z(be_ratio, z_win_val)

    # --- ORIGINAL REVELATION P/D (-2 to 2) ---
    rng = (df['High'] - df['Low']) + 1e-9
    pd_v = 1.0 - ((2 * df['Close'] - (df['High'] + df['Low'])) / rng)
    u, d = (df['High'] - df['High'].shift(1)), (df['Low'].shift(1) - df['Low'])
    
    h_vals, phases = [], []
    for up, dw, val in zip(u, d, pd_v):
        if up > dw and up > 0:
            phases.append("Dp" if val > 1.0 else "rP")
            h_vals.append(val)
        elif dw > up and dw > 0:
            phases.append("rD" if val > 1.0 else "Ad")
            h_vals.append(-val)
        else:
            phases.append("Neutral"); h_vals.append(0)
    df['H_Val'], df['Phase'] = h_vals, phases

    # --- VISUALIZATION ---
    st.title(f"⚖️ {primary_ticker} Revelation")
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02, 
                        row_heights=[0.35, 0.15, 0.15, 0.15, 0.20])

    fig.add_trace(go.Candlestick(x=df.index, open=df.get('Open', df['Close']), high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_30'], line=dict(color='orange', width=1.5), name="EMA 30"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_72'], line=dict(color='cyan', width=1.5), name="EMA 72"), row=1, col=1)

    colors = [{"rD":'red', "Ad":'orange', "rP":'lime', "Dp":'green'}.get(p, 'gray') for p in df['Phase']]
    fig.add_trace(go.Bar(x=df.index, y=df['H_Val'], marker_color=colors, name="P/D"), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['Z_EMA'], line=dict(color='yellow'), name="Z-EMA"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Z_GOLD_BTC'], line=dict(color='magenta'), name="Z-Gold"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Z_BTC_ETH'], line=dict(color='deepskyblue'), name="Z-Alts"), row=5, col=1)

    fig.update_layout(height=1100, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- DATA TABLE ---
    st.subheader("📋 Data Output (Recent at Top)")
    st.dataframe(df[['Close', 'EMA_30', 'EMA_72', 'Z_EMA', 'Z_GOLD_BTC', 'Z_BTC_ETH', 'Phase']].sort_index(ascending=False).head(50), use_container_width=True)
