"""
Screener Ichi-Fibo-Heikin Pro • v5.0 (Optimized Core)
═════════════════════════════════════════════════════
Perbaikan Kritis:
- Ichimoku Cloud Shift Fixed (Masa kini, bukan masa depan)
- Vectorized Heikin Ashi (EWM, 10x lebih cepat)
- Market Breadth Regime (Akurat berbasis partisipasi pasar)
- Batch YFinance Download (Anti limit/banned)
- Streamlit Dialogs (Tanpa st.rerun yang memberatkan)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import IchimokuIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import StochasticOscillator
from ta.volatility import AverageTrueRange

# Konfigurasi Halaman
st.set_page_config(page_title="IFH Pro v5.0", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS (Dark Mode Pro) ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .metric-box { background: #1E2329; padding: 15px; border-radius: 8px; border-left: 4px solid #00C853; }
    .metric-title { font-size: 0.9rem; color: #848E9C; }
    .metric-value { font-size: 1.5rem; font-weight: bold; color: #FFFFFF; }
    .signal-bull { color: #00C853; font-weight: bold; }
    .signal-bear { color: #FF3B30; font-weight: bold; }
    .dataframe th { background-color: #1E2329 !important; color: #F0B90B !important; }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI DATA & INDIKATOR ---
@st.cache_data(ttl=3600)
def load_tickers():
    try:
        df = pd.read_csv("emiten.csv")
        return df['ticker'].dropna().unique().tolist()
    except Exception:
        return ["BBCA.JK", "BMRI.JK", "BREN.JK", "AMMN.JK", "TLKM.JK", "ASII.JK"]

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_batch_data(tickers, period="6mo"):
    # Batch download efisien, meminimalisir API Rate Limit
    if not tickers: return pd.DataFrame()
    data = yf.download(" ".join(tickers), period=period, interval="1d", group_by="ticker", threads=True, show_errors=False)
    return data

def heikin_ashi_fast(df: pd.DataFrame) -> pd.DataFrame:
    r = pd.DataFrame(index=df.index)
    r["HA_C"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    r["HA_O"] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2
    r["HA_O"] = r["HA_C"].shift(1).fillna(r["HA_O"]).ewm(alpha=0.5, adjust=False).mean()
    r["HA_H"] = r[["HA_O", "HA_C"]].join(df["High"]).max(axis=1)
    r["HA_L"] = r[["HA_O", "HA_C"]].join(df["Low"]).min(axis=1)
    return r

def process_technical(df: pd.DataFrame):
    if len(df) < 60: return None
    
    # 1. Ichimoku (FIXED: Shift for current cloud)
    ichi = IchimokuIndicator(high=df["High"], low=df["Low"], window1=9, window2=26, window3=52)
    df["Tenkan"] = ichi.ichimoku_conversion_line()
    df["Kijun"] = ichi.ichimoku_base_line()
    df["Senkou_A_Today"] = ichi.ichimoku_a().shift(26)
    df["Senkou_B_Today"] = ichi.ichimoku_b().shift(26)
    
    # 2. Heikin Ashi
    ha_df = heikin_ashi_fast(df)
    
    # 3. Momentum & Volatility
    df["EMA20"] = EMAIndicator(df["Close"], window=20).ema_indicator()
    macd = MACD(df["Close"])
    df["MACD_Hist"] = macd.macd_diff()
    df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    
    stoch = StochasticOscillator(df["High"], df["Low"], df["Close"], window=14, smooth_window=3)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()
    
    # Extract Latest Values
    last = df.iloc[-1]
    last_ha = ha_df.iloc[-1]
    
    # Base Filters
    above_kumo = last["Close"] > max(last["Senkou_A_Today"], last["Senkou_B_Today"])
    ha_bullish = last_ha["HA_C"] > last_ha["HA_O"]
    no_lower_shadow = abs(last_ha["HA_O"] - last_ha["HA_L"]) <= (last["Close"] * 0.002)
    
    if not (above_kumo and ha_bullish):
        return None # Skip jika tidak uptrend atau momentum Heikin tidak bullish
        
    # Confluence Scoring
    score = 0
    reasons = []
    
    if last["Close"] > last["Kijun"]: 
        score += 2; reasons.append("🟢 Harga di atas Kijun-sen")
    if last["MACD_Hist"] > 0 and last["MACD_Hist"] > df["MACD_Hist"].iloc[-2]: 
        score += 2; reasons.append("🟢 MACD Histogram Menguat")
    if last["Stoch_K"] > last["Stoch_D"] and last["Stoch_K"] < 80: 
        score += 1; reasons.append("🟢 Stochastic Golden Cross / Momentum Up")
    if no_lower_shadow: 
        score += 3; reasons.append("🔥 Heikin Ashi Strong Bull (No Lower Shadow)")
    
    # Fibo Levels (Simple Swing based on recent 20d extremes)
    recent = df.tail(20)
    high, low = recent["High"].max(), recent["Low"].min()
    diff = high - low
    
    return {
        "Close": last["Close"],
        "Score": score,
        "Reasons": reasons,
        "ATR": last["ATR"],
        "Kijun": last["Kijun"],
        "Fibo_382": high - (0.382 * diff),
        "Fibo_500": high - (0.500 * diff),
        "Fibo_618": high - (0.618 * diff),
        "Target": high + (0.618 * diff)
    }

def get_market_regime(data_dict):
    """Kalkulasi Market Breadth (% Saham di atas EMA20)"""
    above_ema_count = 0
    valid_tickers = 0
    
    for ticker, df in data_dict.items():
        if len(df) > 20:
            ema20 = df["Close"].ewm(span=20, adjust=False).mean()
            if df["Close"].iloc[-1] > ema20.iloc[-1]:
                above_ema_count += 1
            valid_tickers += 1
            
    if valid_tickers == 0: return "NEUTRAL", 0
    
    breadth = (above_ema_count / valid_tickers) * 100
    if breadth > 60: return "BULL", breadth
    elif breadth < 40: return "BEAR", breadth
    else: return "NEUTRAL", breadth

# --- UI DIALOG (No st.rerun) ---
@st.dialog("📊 Analisis Mendalam & Action Plan")
def show_analysis_modal(ticker, data):
    st.markdown(f"### {ticker} | Score: {data['Score']}/8")
    st.markdown("#### 🧠 Alasan Sinyal:")
    for r in data["Reasons"]:
        st.write(f"- {r}")
        
    st.markdown("---")
    st.markdown("#### 🎯 Trading Plan (Swing Pendek 2-5 Hari)")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Area Beli (Fibo 38-50%)", f"{data['Fibo_500']:.0f} - {data['Fibo_382']:.0f}")
    c2.metric("Target Profit (Fibo Ext)", f"{data['Target']:.0f}")
    c3.metric("Stop Loss (Kijun-sen)", f"{data['Kijun']:.0f}")
    
    st.info(f"💡 Volatilitas Harian (ATR): Rp {data['ATR']:.0f}. Atur sizing lot Anda agar kerugian maksimal 1-2% dari total modal jika menyentuh Stop Loss.")

# --- MAIN APP ---
def main():
    st.title("🦅 Ichi-Fibo-Heikin Pro Screener v5.0")
    
    all_tickers = load_tickers()
    
    with st.sidebar:
        st.header("⚙️ Konfigurasi")
        limit = st.slider("Jumlah Saham (Top N):", 10, len(all_tickers), min(100, len(all_tickers)), step=10)
        min_vol = st.number_input("Min Volume (Avg 10d):", 100000)
        run_btn = st.button("🚀 Jalankan Screener", use_container_width=True)
    
    if run_btn:
        target_tickers = all_tickers[:limit]
        
        with st.spinner(f"Mengunduh data {limit} emiten secara paralel..."):
            raw_data = fetch_batch_data(target_tickers)
        
        # Ekstrak data ke dict
        market_data = {}
        for t in target_tickers:
            try:
                # Handle YF multi-ticker format
                df = raw_data[t].copy() if isinstance(raw_data.columns, pd.MultiIndex) else raw_data.copy()
                df = df.dropna(subset=['Close'])
                if len(df) > 60 and df['Volume'].tail(10).mean() >= min_vol and df['Close'].iloc[-1] >= 50:
                    market_data[t] = df
            except Exception:
                pass
                
        # Analisis Market Regime
        regime, breadth_pct = get_market_regime(market_data)
        r_color = "#00C853" if regime == "BULL" else "#FF3B30" if regime == "BEAR" else "#F0B90B"
        
        st.markdown(f"""
        <div style='background:#1E2329; padding:15px; border-radius:8px; margin-bottom:20px; border-left: 5px solid {r_color};'>
            <h4 style='margin:0; color:#FAFAFA;'>Market Regime: <span style='color:{r_color};'>{regime}</span></h4>
            <p style='margin:0; color:#848E9C;'>Market Breadth: {breadth_pct:.1f}% saham berada di atas EMA-20.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Proses Screener
        results = {}
        progress = st.progress(0)
        
        for i, (ticker, df) in enumerate(market_data.items()):
            res = process_technical(df)
            if res: results[ticker] = res
            progress.progress((i + 1) / len(market_data))
            
        progress.empty()
        
        if not results:
            st.warning("Tidak ada saham yang memenuhi kriteria Uptrend Kuat hari ini.")
            return
            
        # Tampilkan Hasil
        st.success(f"Ditemukan {len(results)} saham potensial!")
        
        # Konversi ke format tabel visual
        display_data = []
        for t, r in sorted(results.items(), key=lambda x: x[1]['Score'], reverse=True):
            display_data.append({
                "Ticker": t,
                "Harga": round(r["Close"], 0),
                "Score": f"{r['Score']}/8",
                "Buy Zone": f"{r['Fibo_500']:.0f} - {r['Fibo_382']:.0f}",
                "StopLoss": round(r["Kijun"], 0),
                "Target": round(r["Target"], 0)
            })
            
        st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)
        
        st.markdown("### 🔍 Action Panel")
        cols = st.columns(6)
        for i, (tk, data) in enumerate(results.items()):
            with cols[i % 6]:
                if st.button(f"Action {tk}", key=f"btn_{tk}", use_container_width=True):
                    show_analysis_modal(tk, data)

if __name__ == "__main__":
    main()
