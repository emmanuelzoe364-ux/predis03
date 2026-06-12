import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# --- INITIAL SETTINGS ---
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="Revelation Engine: Smoothed")

# --- SIDEBAR & REFRESH ---
if 'refresh_count' not in st.session_state:
    st.session_state.refresh_count = 0

with st.sidebar:
    st.header("🎨 Professional Display")
    primary_ticker = st.text_input("Ticker (Yahoo Format)", value="BTC-USD").strip().upper()
    timeframe = st.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "1d"], index=1)
    lookback = st.selectbox("Data Lookback", ["1d", "3d", "7d", "1mo"], index=2)
    
    st.divider()
    ema_fast = st.number_input("Fast EMA", value=30)
    ema_slow = st.number_input("Slow EMA", value=72)
    z_win = st.number_input("Z-Score Window", value=20)
    smooth_period = st.number_input("Z-Smoothing Period", value=30) # User requested 30

    if st.button("🔄 FORCE SYNC & REFRESH"):
        st.cache_data.clear()
        st.session_state.refresh_count += 1
        st.rerun()

# --- HIGH-QUALITY YAHOO DATA ENGINE ---
def fetch_yahoo_direct(ticker, interval, range_str):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval={interval}&range={range_str}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()['chart']['result'][0]
        
        df = pd.DataFrame({
            'Open': data['indicators']['quote'][0]['open'],
            'High': data['indicators']['quote'][0]['high'],
            'Low': data['indicators']['quote'][0]['low'],
            'Close': data['indicators']['quote'][0]['close']
        }, index=pd.to_datetime(data['timestamp'], unit='s'))
        
        return df.ffill().dropna()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=30)
def get_all_data_parallel(main_t, tf, lb, refresh_id):
    tickers = [main_t, "PAXG-USD", "BTC-USD", "ETH-USD"]
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_ticker = {executor.submit(fetch_yahoo_direct, s, tf, lb): s for s in tickers}
        for future in future_to_ticker:
            res_df = future.result()
            results[future_to_ticker[future]] = res_df
    return results

# EXECUTE FETCH
data_map = get_all_data_parallel(primary_ticker, timeframe, lookback, st.session_state.refresh_count)
main_df = data_map.get(primary_ticker, pd.DataFrame())

if main_df.empty:
    st.error("❌ Yahoo Finance is temporarily unreachable. Please wait 15 seconds and refresh.")
else:
    # --- ALIGNMENT ---
    idx = main_df.index
    def sync(sym):
        d = data_map.get(sym, pd.DataFrame())
        return d['Close'].reindex(idx).ffill().bfill() if not d.empty else pd.Series(0, index=idx)

    p_c, b_c, e_c = sync("PAXG-USD"), sync("BTC-USD"), sync("ETH-USD")
    pb_ratio, be_ratio = p_c / (b_c + 1e-9), b_c / (e_c + 1e-9)

    # --- CALCULATIONS ---
    # 1. EMAs
    main_df['EMA_30'] = main_df['Close'].ewm(span=ema_fast, adjust=False).mean()
    main_df['EMA_72'] = main_df['Close'].ewm(span=ema_slow, adjust=False).mean()
    main_df['Spread'] = main_df['EMA_30'] - main_df['EMA_72']
    
    def get_z(ser, win):
        return (ser - ser.rolling(win).mean()) / (ser.rolling(win).std() + 1e-9)
    
    # 2. Raw EMA Z-Score (Kept raw for precision)
    main_df['Z_EMA'] = get_z(main_df['Spread'], z_win)
    
    # 3. Smoothed Inter-market Z-Scores
    # We calculate the Z-score first, then apply the 30 EMA smoothing
    z_gold_raw = get_z(pb_ratio, z_win)
    main_df['Z_GOLD_BTC'] = z_gold_raw.ewm(span=smooth_period, adjust=False).mean()
    
    z_eth_raw = get_z(be_ratio, z_win)
    main_df['Z_BTC_ETH'] = z_eth_raw.ewm(span=smooth_period, adjust=False).mean()

    # 4. Revelation P/D logic
    rng = (main_df['High'] - main_df['Low']) + 1e-9
    pd_v = 1.0 - ((2 * main_df['Close'] - (main_df['High'] + main_df['Low'])) / rng)
    u, d = (main_df['High'] - main_df['High'].shift(1)), (main_df['Low'].shift(1) - main_df['Low'])
    
    h_vals, phases = [], []
    for up, dw, val in zip(u, d, pd_v):
        if up > dw and up > 0: phases.append("Dp" if val > 1.0 else "rP"); h_vals.append(val)
        elif dw > up and dw > 0: phases.append("rD" if val > 1.0 else "Ad"); h_vals.append(-val)
        else: phases.append("Neutral"); h_vals.append(0)
    main_df['H_Val'], main_df['Phase'] = h_vals, phases

    # --- PLOTTING ---
    st.title(f"⚖️ Revelation Engine: {primary_ticker}")
    
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
        row_heights=[0.35, 0.12, 0.16, 0.16, 0.16],
        subplot_titles=(
            "1. Price Action", 
            "2. Revelation P/D Scale", 
            "3. EMA Divergence Z-Score (Raw)", 
            f"4. GOLD/BTC Z-Score ({smooth_period} EMA Smooth)", 
            f"5. BTC/ETH Z-Score ({smooth_period} EMA Smooth)"
        )
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=main_df.index, open=main_df['Open'], high=main_df['High'], low=main_df['Low'], close=main_df['Close'], 
        name="Market Price", increasing_line_color='#26a69a', decreasing_line_color='#ef5350', line_width=1.5
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['EMA_30'], line=dict(color='orange', width=2), name="EMA 30"), row=1, col=1)
    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['EMA_72'], line=dict(color='cyan', width=2), name="EMA 72"), row=1, col=1)

    # Revelation
    c_map = {"rD":'#FF0000', "Ad":'#FFA500', "rP":'#00FF00', "Dp":'#006400', "Neutral":'#808080'}
    fig.add_trace(go.Bar(x=main_df.index, y=main_df['H_Val'], marker_color=[c_map.get(p, 'gray') for p in main_df['Phase']], name="P/D Intensity"), row=2, col=1)

    # Subplot 3: Raw Z-EMA
    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['Z_EMA'], line=dict(color='#FFFF00', width=2), name="Z-EMA (Raw)"), row=3, col=1)
    
    # Subplots 4 & 5: Smoothed Z-Scores
    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['Z_GOLD_BTC'], line=dict(color='#FF00FF', width=2.5), name="Z-Gold (Smoothed)"), row=4, col=1)
    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['Z_BTC_ETH'], line=dict(color='#00BFFF', width=2.5), name="Z-Alts (Smoothed)"), row=5, col=1)

    # Reference Lines
    for r in [3, 4, 5]:
        fig.add_hline(y=2.0, line_dash="dash", line_color="white", row=r, col=1, opacity=0.4)
        fig.add_hline(y=-2.0, line_dash="dash", line_color="white", row=r, col=1, opacity=0.4)
        fig.add_hline(y=0, line_width=1, line_color="gray", row=r, col=1, opacity=0.6)

    fig.update_layout(height=1250, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=True, margin=dict(t=50, b=50, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    # --- DATA TABLE ---
    st.subheader("📋 Output Data Table (Smoothed Stats)")
    cols = ['Close', 'EMA_30', 'EMA_72', 'H_Val', 'Z_EMA', 'Z_GOLD_BTC', 'Z_BTC_ETH', 'Phase']
    st.dataframe(main_df[cols].sort_index(ascending=False).head(100).round(4), use_container_width=True)
