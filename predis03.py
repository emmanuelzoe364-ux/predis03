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
st.set_page_config(layout="wide", page_title="Revelation Engine: Binance Pro")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    # Ticker input
    raw_ticker = st.text_input("Primary Ticker", value="BTCUSDT").strip().upper()
    
    # Helper to fix common ticker mistakes for Binance
    ticker = raw_ticker.replace("-", "").replace("/", "")
    if not ticker.endswith("USDT") and not ticker.endswith("BTC"):
        ticker = ticker + "USDT"

    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=0)
    
    st.divider()
    ema_fast_val = st.number_input("Fast EMA", value=30)
    ema_slow_val = st.number_input("Slow EMA", value=72)
    z_win_val = st.number_input("Z-Score Window", value=20)

    if st.button("🔄 REFRESH LIVE DATA"):
        st.cache_data.clear()
        st.rerun()

st.title(f"⚖️ Revelation Engine: {ticker}")
st.caption("Data Source: Binance Public API (Real-Time)")

# --- BINANCE DATA ENGINE ---
@st.cache_data(ttl=5)
def fetch_binance_data(symbol, interval):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": 500}
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        
        # Check if the API returned an error or empty list
        if isinstance(data, dict) or not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data, columns=[
            'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
            'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
        ])
        
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('Date', inplace=True)
        
        cols = ['Open', 'High', 'Low', 'Close']
        df[cols] = df[cols].astype(float)
        return df[cols]
    except Exception as e:
        return pd.DataFrame()

# Fetch required layers
main_df = fetch_binance_data(ticker, timeframe)
paxg_df = fetch_binance_data("PAXGUSDT", timeframe)
eth_df = fetch_binance_data("ETHUSDT", timeframe)
btc_ref = fetch_binance_data("BTCUSDT", timeframe)

# --- VALIDATION GUARD ---
if main_df.empty:
    st.error(f"❌ No data found for '{ticker}'. Please ensure this is a valid Binance pair (e.g., BTCUSDT, ETHBTC).")
    st.info("💡 Note: Binance symbols usually end in 'USDT'. Your input was converted to: " + ticker)
elif paxg_df.empty or eth_df.empty:
    st.warning("⚠️ Primary data loaded, but Intermarket assets (PAXG or ETH) failed to load. Divergence charts will be empty.")
    # Create dummy data for ratios to prevent crash
    idx = main_df.index
    pb_ratio = pd.Series(0, index=idx)
    be_ratio = pd.Series(0, index=idx)
else:
    # --- ALIGNMENT ---
    idx = main_df.index
    p_c = paxg_df['Close'].reindex(idx).ffill().bfill()
    b_c = btc_ref['Close'].reindex(idx).ffill().bfill()
    e_c = eth_df['Close'].reindex(idx).ffill().bfill()
    
    pb_ratio = p_c / (b_c + 1e-9)
    be_ratio = b_c / (e_c + 1e-9)

    # --- MATH SECTION ---
    main_df['EMA_30'] = main_df['Close'].ewm(span=ema_fast_val, adjust=False).mean()
    main_df['EMA_72'] = main_df['Close'].ewm(span=ema_slow_val, adjust=False).mean()
    main_df['Spread'] = main_df['EMA_30'] - main_df['EMA_72']
    main_df['Z_EMA'] = (main_df['Spread'] - main_df['Spread'].rolling(z_win_val).mean()) / (main_df['Spread'].rolling(z_win_val).std() + 1e-9)
    
    def get_z(ser, win):
        return (ser - ser.rolling(win).mean()) / (ser.rolling(win).std() + 1e-9)
    
    main_df['Z_GOLD_BTC'] = get_z(pb_ratio, z_win_val)
    main_df['Z_BTC_ETH'] = get_z(be_ratio, z_win_val)

    # --- REVELATION P/D (-2 to 2) ---
    rng = (main_df['High'] - main_df['Low']) + 1e-9
    pd_v = 1.0 - ((2 * main_df['Close'] - (main_df['High'] + main_df['Low'])) / rng)
    u = main_df['High'] - main_df['High'].shift(1)
    d = main_df['Low'].shift(1) - main_df['Low']
    
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
    # Safe check for the timestamp to avoid IndexError
    if not main_df.empty:
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
    fig.add_trace(go.Scatter(x=main_df.index, y=df['Z_BTC_ETH'] if 'df' in locals() else main_df['Z_BTC_ETH'], line=dict(color='deepskyblue'), name="Z-Alts"), row=5, col=1)

    fig.update_layout(height=1100, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Output Data Table (Real-Time)")
    cols = ['Close', 'EMA_30', 'EMA_72', 'H_Val', 'Z_EMA', 'Z_GOLD_BTC', 'Z_BTC_ETH', 'Phase']
    st.dataframe(main_df[cols].sort_index(ascending=False).head(50), use_container_width=True)
