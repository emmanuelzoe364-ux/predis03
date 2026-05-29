import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# --- INITIAL SETTINGS ---
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="Revelation Engine")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Primary Ticker", value="BTC-USD").strip().upper()
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "1d"], index=3)
    # Warning: 5m/15m data fails if period is too long
    period = st.selectbox("Data Period", ["7d", "1mo", "3mo", "1y"], index=1)
    
    st.divider()
    ema_fast_val = st.number_input("Fast EMA", value=30)
    ema_slow_val = st.number_input("Slow EMA", value=72)
    z_win_val = st.number_input("Z-Score Window", value=20)

    if st.button("🔄 Force Refresh"):
        st.cache_data.clear()
        st.rerun()

# --- DATA ENGINE ---
@st.cache_data(ttl=300)
def fetch_and_align(main_t, tf, prd):
    try:
        assets = {"Main": main_t, "PAXG": "PAXG-USD", "BTC": "BTC-USD", "ETH": "ETH-USD"}
        dfs = {}
        for key, sym in assets.items():
            d = yf.download(sym, period=prd, interval=tf, auto_adjust=True, progress=False)
            if d.empty: return None, f"No data for {sym}."
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
            dfs[key] = d

        # Align all to Main Ticker
        df = dfs["Main"].copy()
        df['PAXG_C'] = dfs["PAXG"]['Close'].reindex(df.index).ffill().bfill()
        df['BTC_C'] = dfs["BTC"]['Close'].reindex(df.index).ffill().bfill()
        df['ETH_C'] = dfs["ETH"]['Close'].reindex(df.index).ffill().bfill()
        return df, None
    except Exception as e:
        return None, str(e)

df_raw, err = fetch_and_align(ticker, timeframe, period)

if err:
    st.error(f"Data Error: {err}")
elif df_raw is not None:
    df = df_raw.copy()

    # --- MATH SECTION ---
    df['EMA_30'] = df['Close'].ewm(span=ema_fast_val, adjust=False).mean()
    df['EMA_72'] = df['Close'].ewm(span=ema_slow_val, adjust=False).mean()
    
    df['Spread'] = df['EMA_30'] - df['EMA_72']
    # Safety: Replace potential Inf with NaN
    df['Z_EMA'] = (df['Spread'] - df['Spread'].rolling(z_win_val).mean()) / (df['Spread'].rolling(z_win_val).std() + 1e-9)
    
    r_pb = (df['PAXG_C'] / (df['BTC_C'] + 1e-9))
    df['Z_GOLD_BTC'] = (r_pb - r_pb.rolling(z_win_val).mean()) / (r_pb.rolling(z_win_val).std() + 1e-9)
    
    r_be = (df['BTC_C'] / (df['ETH_C'] + 1e-9))
    df['Z_BTC_ETH'] = (r_be - r_be.rolling(z_win_val).mean()) / (r_be.rolling(z_win_val).std() + 1e-9)

    # Revelation P/D Scale (-2 to 2)
    candle_rng = (df['High'] - df['Low']) + 1e-9
    pd_scale = 1.0 - ((2 * df['Close'] - (df['High'] + df['Low'])) / candle_rng)
    u, d = (df['High'] - df['High'].shift(1)), (df['Low'].shift(1) - df['Low'])
    
    h_vals, phases = [], []
    for up, dw, val in zip(u, d, pd_scale):
        if up > dw and up > 0:
            phases.append("Dp" if val > 1.0 else "rP")
            h_vals.append(float(val))
        elif dw > up and dw > 0:
            phases.append("rD" if val > 1.0 else "Ad")
            h_vals.append(float(-val))
        else:
            phases.append("Neutral")
            h_vals.append(0.0)
    
    df['H_Val'] = h_vals
    df['Phase'] = phases

    # --- PLOT ---
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.35, 0.15, 0.15, 0.15, 0.20])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_30'], line=dict(color='orange', width=1.5), name="EMA 30"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_72'], line=dict(color='cyan', width=1.5), name="EMA 72"), row=1, col=1)
    
    c_map = {"rD":'#FF0000', "Ad":'#FFA500', "rP":'#00FF00', "Dp":'#006400', "Neutral":'gray'}
    fig.add_trace(go.Bar(x=df.index, y=df['H_Val'], marker_color=[c_map.get(p, 'gray') for p in df['Phase']], name="P/D"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Z_EMA'], line=dict(color='yellow'), name="Z-EMA"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Z_GOLD_BTC'], line=dict(color='magenta'), name="Z-Gold"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Z_BTC_ETH'], line=dict(color='deepskyblue'), name="Z-Alts"), row=5, col=1)

    fig.update_layout(height=1000, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- DATAFRAME SAFETY FIX ---
    st.subheader("📋 Revelation Output Data")
    
    # 1. Select and Clean the Data
    final_df = df[['Close', 'EMA_30', 'EMA_72', 'H_Val', 'Z_EMA', 'Z_GOLD_BTC', 'Z_BTC_ETH', 'Phase']].copy()
    
    # 2. Replace Infinity/NaN to prevent JSON SyntaxErrors
    final_df = final_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 3. Format index and limit rows for browser stability
    final_df.index = final_df.index.strftime('%Y-%m-%d %H:%M')
    
    # 4. Display a limited window (Most recent 500 rows)
    st.dataframe(final_df.iloc[-500:].sort_index(ascending=False), use_container_width=True)
