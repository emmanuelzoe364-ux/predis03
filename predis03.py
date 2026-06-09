import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import warnings
from datetime import datetime

# --- INITIAL SETTINGS ---
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="Revelation Engine: Binance Live")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    # Ticker input (Defaults to BTC)
    primary_ticker = st.text_input("Primary Ticker (Binance Format)", value="BTCUSDT").strip().upper()
    
    # Binance Intervals
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=0)
    
    st.divider()
    ema_fast_val = st.number_input("Fast EMA", value=30)
    ema_slow_val = st.number_input("Slow EMA", value=72)
    z_win_val = st.number_input("Z-Score Window", value=20)

    if st.button("🔄 REFRESH LIVE DATA"):
        st.cache_data.clear()
        st.rerun()

st.title(f"⚖️ Revelation Engine: {primary_ticker}")
st.caption("Source: Binance Real-Time API | No Throttling | Cloud Optimized")

# --- BINANCE DATA ENGINE ---
@st.cache_data(ttl=5) # 5 second cache for true live feel
def fetch_binance_data(symbol, interval):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": 500}
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
            'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
        ])
        
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('Date', inplace=True)
        
        cols = ['Open', 'High', 'Low', 'Close']
        df[cols] = df[cols].astype(float)
        return df[cols]
    except:
        return None

# Fetch all 3 layers from Binance
main_df = fetch_binance_data(primary_ticker, timeframe)
paxg_df = fetch_binance_data("PAXGUSDT", timeframe)
eth_df = fetch_binance_data("ETHUSDT", timeframe)
btc_ref = fetch_binance_data("BTCUSDT", timeframe)

if main_df is None or paxg_df is None:
    st.error("📡 API Connection Error. Please ensure the ticker symbols are correct Binance pairs (e.g., BTCUSDT).")
else:
    # --- ALIGNMENT ---
    idx = main_df.index
    p_c = paxg_df['Close'].reindex(idx).ffill().bfill()
    b_c = btc_ref['Close'].reindex(idx).ffill().bfill()
    e_c = eth_df['Close'].reindex(idx).ffill().bfill()
    
    pb_ratio = p_c / (b_c + 1e-9)
    be_ratio = b_c / (e_c + 1e-9)

    # --- MATH SECTION ---
    # Native EMAs
    main_df['EMA_30'] = main_df['Close'].ewm(span=ema_fast_val, adjust=False).mean()
    main_df['EMA_72'] = main_df['Close'].ewm(span=ema_slow_val, adjust=False).mean()
    
    # Z-Scores
    main_df['Spread'] = main_df['EMA_30'] - main_df['EMA_72']
    main_df['Z_EMA'] = (main_df['Spread'] - main_df['Spread'].rolling(z_win_val).mean()) / (main_df['Spread'].rolling(z_win_val).std() + 1e-9)
    
    def get_z(ser, win):
        return (ser - ser.rolling(win).mean()) / (ser.rolling(win).std() + 1e-9)
    
    main_df['Z_GOLD_BTC'] = get_z(pb_ratio, z_win_val)
    main_df['Z_BTC_ETH'] = get_z(be_ratio, z_win_val)

    # --- ORIGINAL REVELATION P/D (-2 to 2) ---
    rng = (main_df['High'] - main_df['Low']) + 1e-9
    pd_v = 1.0 - ((2 * main_df['Close'] - (main_df['High'] + main_df['Low'])) / rng)
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

    # --- VISUALIZATION ---
    # Latest Sync Display
    last_ts = main_df.index[-1].strftime('%H:%M:%S')
    st.metric("Latest Binance Candle (UTC)", last_ts)

    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02, 
                        row_heights=[0.35, 0.15, 0.15, 0.15, 0.20])

    fig.add_trace(go.Candlestick(x=main_df.index, open=main_df['Open'], high=main_df['High'], low=main_df['Low'], close=main_df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['EMA_30'], line=dict(color='orange', width=1.5), name="EMA 30"), row=1, col=1)
    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['EMA_72'], line=dict(color='cyan', width=1.5), name="EMA 72"), row=1, col=1)

    c_map = {"rD":'red', "Ad":'orange', "rP":'lime', "Dp":'green', "Neutral":'gray'}
    fig.add_trace(go.Bar(x=main_df.index, y=main_df['H_Val'], marker_color=[c_map.get(p, 'gray') for p in main_df['Phase']]), row=2, col=1)

    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['Z_EMA'], line=dict(color='yellow'), name="Z-EMA"), row=3, col=1)
    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['Z_GOLD_BTC'], line=dict(color='magenta'), name="Z-Gold"), row=4, col=1)
    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['Z_BTC_ETH'], line=dict(color='deepskyblue'), name="Z-Alts"), row=5, col=1)

    fig.update_layout(height=1100, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- DATA TABLE ---
    st.subheader("📋 Output Data Table (Real-Time)")
    cols = ['Close', 'EMA_30', 'EMA_72', 'H_Val', 'Z_EMA', 'Z_GOLD_BTC', 'Z_BTC_ETH', 'Phase']
    st.dataframe(main_df[cols].sort_index(ascending=False).head(50), use_container_width=True)
