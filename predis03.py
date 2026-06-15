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
st.set_page_config(layout="wide", page_title="Revelation Engine: Full Pro")

# --- SIDEBAR & REFRESH ---
if 'refresh_count' not in st.session_state:
    st.session_state.refresh_count = 0

with st.sidebar:
    st.header("🛡️ Strategy Monitor")
    primary_ticker = st.text_input("Ticker", value="BTC-USD").strip().upper()
    timeframe = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=1)
    lookback = st.selectbox("Data Lookback", ["1d", "3d", "7d", "1mo"], index=2)
    
    st.divider()
    st.subheader("EMA & Stats Settings")
    ema_fast = st.number_input("Fast EMA", value=30)
    ema_slow = st.number_input("Slow EMA", value=72)
    z_win = st.number_input("Z-Score Window", value=20)
    
    st.divider()
    st.subheader("🎚️ Smoothing Controls")
    smooth_period = st.number_input("Global Smooth (EMA/Gold/Alts)", value=3) 
    usdt_smooth_period = st.number_input("USDT/BTC Smooth Only", value=3)
    
    freshness_status = st.empty()
    st.divider()

    if st.button("🔄 REFRESH & SYNC"):
        st.cache_data.clear()
        st.session_state.refresh_count += 1
        st.rerun()

# --- DATA ENGINE ---
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
    except: return pd.DataFrame()

@st.cache_data(ttl=30)
def get_all_data_parallel(main_t, tf, lb, refresh_id):
    tickers = [main_t, "PAXG-USD", "BTC-USD", "ETH-USD", "USDT-USD", "^VIX"]
    results = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_ticker = {executor.submit(fetch_yahoo_direct, s, tf, lb): s for s in tickers}
        for future in future_to_ticker:
            res_df = future.result()
            results[future_to_ticker[future]] = res_df
    return results

# EXECUTE FETCH
data_map = get_all_data_parallel(primary_ticker, timeframe, lookback, st.session_state.refresh_count)
main_df = data_map.get(primary_ticker, pd.DataFrame())

if main_df.empty:
    st.error("❌ Yahoo Finance unreachable. Refresh in 15 seconds.")
else:
    # --- ALIGNMENT ---
    idx = main_df.index
    def sync(sym):
        d = data_map.get(sym, pd.DataFrame())
        return d['Close'].reindex(idx).ffill().bfill() if not d.empty else pd.Series(0, index=idx)

    p_c, b_c, e_c, u_c, v_c = sync("PAXG-USD"), sync("BTC-USD"), sync("ETH-USD"), sync("USDT-USD"), sync("^VIX")
    pb_ratio, be_ratio, ub_ratio = p_c/(b_c+1e-9), b_c/(e_c+1e-9), u_c/(b_c+1e-9)

    # --- CALCULATIONS ---
    # 1. EMAs (30 and 72)
    main_df['EMA_30'] = main_df['Close'].ewm(span=ema_fast, adjust=False).mean()
    main_df['EMA_72'] = main_df['Close'].ewm(span=ema_slow, adjust=False).mean()
    main_df['Spread'] = main_df['EMA_30'] - main_df['EMA_72']
    
    def get_z_smoothed(series, win, smooth):
        raw_z = (series - series.rolling(win).mean()) / (series.rolling(win).std() + 1e-9)
        return raw_z.ewm(span=smooth, adjust=False).mean()
    
    # 2. Z-Scores
    main_df['Z_EMA'] = get_z_smoothed(main_df['Spread'], z_win, smooth_period)
    main_df['Z_GOLD_BTC'] = get_z_smoothed(pb_ratio, z_win, smooth_period)
    main_df['Z_BTC_ETH'] = get_z_smoothed(be_ratio, z_win, smooth_period)
    main_df['Z_USDT'] = get_z_smoothed(ub_ratio, z_win, usdt_smooth_period)
    main_df['Z_VIX'] = get_z_smoothed(v_c, z_win, 3)

    # 3. Revelation P/D logic
    rng = (main_df['High'] - main_df['Low']) + 1e-9
    pd_v = 1.0 - ((2 * main_df['Close'] - (main_df['High'] + main_df['Low'])) / rng)
    u, d = (main_df['High'] - main_df['High'].shift(1)), (main_df['Low'].shift(1) - main_df['Low'])
    h_vals, phases = [], []
    for up_m, dw_m, val in zip(u, d, pd_v):
        if up_m > dw_m and up_m > 0: phases.append("Dp" if val > 1.0 else "rP"); h_vals.append(val)
        elif dw_m > up_m and dw_m > 0: phases.append("rD" if val > 1.0 else "Ad"); h_vals.append(-val)
        else: phases.append("Neutral"); h_vals.append(0)
    main_df['H_Val'], main_df['Phase'] = h_vals, phases

    # --- DROP LIVE CANDLE ---
    main_df = main_df.iloc[:-1]

    # --- SIDEBAR FRESHNESS MONITOR ---
    last_row = main_df.iloc[-1]
    with freshness_status.container():
        st.write("📈 **Setup Freshness Monitor**")
        if last_row['Z_USDT'] >= 1.0:
            if last_row['Z_BTC_ETH'] >= 1.0:
                st.success("🔥 FRESH LEGS: Alts pre-crushed. High recovery potential.")
            else:
                st.warning("⚠️ STALE LEGS: Alts holding up too well. Risk of Alt-dump on BTC bounce.")
        else: st.info("Market Neutral. Monitoring for USDT spike...")

    # --- PLOTTING ---
    st.markdown("<h1 style='text-align: center; color: black;'>⚖️ Revelation Engine Macro</h1>", unsafe_allow_html=True)
    st.metric("Last Confirmed Close (UTC)", main_df.index[-1].strftime('%H:%M:%S'))

    fig = make_subplots(rows=7, cols=1, shared_xaxes=True, vertical_spacing=0.015, 
                        row_heights=[0.25, 0.10, 0.13, 0.13, 0.13, 0.13, 0.13],
                        subplot_titles=("1. Price & EMAs (Confirmed)", "2. P/D Scale", "3. EMA Spread Z", "4. Gold/BTC Z", "5. BTC/ETH Z", "6. USDT/BTC Z", "7. VIX Z"))

    # Subplot 1: Price + EMA 30 + EMA 72
    fig.add_trace(go.Candlestick(x=main_df.index, open=main_df['Open'], high=main_df['High'], low=main_df['Low'], close=main_df['Close'], name="Price", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['EMA_30'], line=dict(color='#FF8C00', width=2), name="EMA 30"), row=1, col=1)
    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['EMA_72'], line=dict(color='#0000FF', width=2), name="EMA 72"), row=1, col=1)

    # Subplot 2: Revelation
    fig.add_trace(go.Bar(x=main_df.index, y=main_df['H_Val'], marker_color=[{"rD":'#FF0000', "Ad":'#FFA500', "rP":'#00FF00', "Dp":'#006400'}.get(p, '#D3D3D3') for p in main_df['Phase']], name="P/D"), row=2, col=1)

    # Subplots 3-7: Z-Scores
    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['Z_EMA'], line=dict(color='#D35400', width=2), name="Z-EMA"), row=3, col=1)
    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['Z_GOLD_BTC'], line=dict(color='#C0392B', width=2), name="Z-Gold"), row=4, col=1)
    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['Z_BTC_ETH'], line=dict(color='#2980B9', width=2), name="Z-Alts"), row=5, col=1)
    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['Z_USDT'], line=dict(color='#000000', width=2.5), name="Z-USDT"), row=6, col=1)
    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['Z_VIX'], line=dict(color='#8E44AD', width=2), name="Z-VIX"), row=7, col=1)

    # Reference Lines
    for r in [3, 4, 5, 6, 7]:
        t = 1.8 if r == 5 else 2.0
        fig.add_hline(y=t, line_dash="dash", line_color="black", row=r, col=1, opacity=0.3)
        fig.add_hline(y=-t, line_dash="dash", line_color="black", row=r, col=1, opacity=0.3)
        fig.add_hline(y=0, line_width=1, line_color="black", row=r, col=1, opacity=0.1)

    fig.update_layout(height=1600, template="plotly_white", xaxis_rangeslider_visible=False, showlegend=True, font=dict(color="black"), margin=dict(t=80, b=50, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    # --- DATA TABLE ---
    st.subheader("📋 Output Data Table (Confirmed Values)")
    cols = ['Close', 'EMA_30', 'EMA_72', 'H_Val', 'Z_EMA', 'Z_GOLD_BTC', 'Z_BTC_ETH', 'Z_USDT', 'Z_VIX', 'Phase']
    st.dataframe(main_df[cols].sort_index(ascending=False).head(100).round(4), use_container_width=True)
