import streamlit as st
import pandas as pd
import yfinance as yf
from ta.trend import IchimokuIndicator
import time

st.set_page_config(page_title="Screener Ichi-Fibo-Heikin Pro", layout="wide")
st.title("Screener Saham: Ichi-Fibo-Heikin (Pro Version) 🚀")

# ==========================================
# FUNGSI BANTUAN
# ==========================================

# 1. Menghitung Heikin Ashi (SUDAH FIX ERROR SERIES)
def calculate_heikin_ashi(df):
    ha_df = df.copy()
    
    # Peras (squeeze) semua kolom agar jadi 1D murni
    O = df['Open'].squeeze()
    H = df['High'].squeeze()
    L = df['Low'].squeeze()
    C = df['Close'].squeeze()
    
    ha_df['HA_Close'] = (O + H + L + C) / 4
    
    ha_open = [(O.iloc[0] + C.iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha_df['HA_Close'].iloc[i-1]) / 2)
        
    ha_df['HA_Open'] = ha_open
    return ha_df

# 2. Mencari Swing High & Swing Low Terdekat (Metode Fraktal)
def get_recent_swings(df, days_lookback, window=5):
    # Pakai .squeeze() agar data 2D menjadi 1D murni (mencegah error Pandas)
    high_data = df['High'].squeeze().tail(days_lookback)
    low_data = df['Low'].squeeze().tail(days_lookback)
    
    swing_high = high_data.max() # Fallback
    swing_low = low_data.min()   # Fallback
    
    # Mencari puncak terdekat (Pivot High)
    for i in range(len(high_data)-1-window, window, -1):
        if high_data.iloc[i] == max(high_data.iloc[i-window:i+window+1]):
            swing_high = high_data.iloc[i]
            break 
            
    # Mencari lembah terdekat (Pivot Low)
    for i in range(len(low_data)-1-window, window, -1):
        if low_data.iloc[i] == min(low_data.iloc[i-window:i+window+1]):
            swing_low = low_data.iloc[i]
            break 
            
    # Kembalikan sebagai angka float murni
    return float(swing_high), float(swing_low)

# ==========================================
# BACA FILE CSV
# ==========================================
@st.cache_data 
def load_emiten():
    try:
        df_emiten = pd.read_csv('emiten.csv')
        if 'ticker' in df_emiten.columns:
            return df_emiten['ticker'].astype(str).tolist()
        else:
            return df_emiten.iloc[:, 0].astype(str).tolist()
    except FileNotFoundError:
        return []

daftar_saham = load_emiten()

if not daftar_saham:
    st.error("File 'emiten.csv' tidak ditemukan. Pastikan file ada di folder yang sama!")
    st.stop()

# ==========================================
# PENGATURAN UI
# ==========================================
st.sidebar.header("Pengaturan")
mode = st.sidebar.radio("Pilih Mode:", ["Analisis Satuan", "Screener Massal"])
days = st.sidebar.slider("Fokus trend berapa hari terakhir (Untuk Fibo)?", 30, 365, 120)

# ==========================================
# MODE 1: ANALISIS SATUAN
# ==========================================
if mode == "Analisis Satuan":
    st.header("Analisis Spesifik")
    ticker = st.selectbox("Pilih Saham:", daftar_saham)
    
    if st.button("Analisis Saham Ini"):
        with st.spinner(f"Mengambil data {ticker} (Download riwayat 2 tahun)..."):
            df = yf.download(ticker, period="730d", progress=False)
            
            if df.empty or len(df) < 60:
                st.error("Data saham tidak cukup di Yahoo Finance.")
            else:
                try:
                    harga_terkini = float(yf.Ticker(ticker).fast_info['last_price'])
                except:
                    harga_terkini = float(df['Close'].squeeze().iloc[-1])

                ichi = IchimokuIndicator(high=df['High'].squeeze(), low=df['Low'].squeeze())
                df['Tenkan'] = ichi.ichimoku_conversion_line()
                df['Kijun'] = ichi.ichimoku_base_line()
                df['Senkou_A'] = ichi.ichimoku_a()
                df['Senkou_B'] = ichi.ichimoku_b()
                
                ha_df = calculate_heikin_ashi(df)
                
                terakhir = df.iloc[-1]
                di_atas_awan = harga_terkini > max(float(terakhir['Senkou_A'].squeeze()), float(terakhir['Senkou_B'].squeeze()))
                tenkan_cross_kijun = float(terakhir['Tenkan'].squeeze()) > float(terakhir['Kijun'].squeeze())
                
                swing_high, swing_low = get_recent_swings(df, days)
                selisih = swing_high - swing_low
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("1. Ichimoku")
                    st.write(f"Harga Terkini: **{harga_terkini:.0f}**")
                    if di_atas_awan and tenkan_cross_kijun:
                        st.success("🟢 UPTREND KUAT!")
                    else:
                        st.warning("🟡 Belum Uptrend Sempurna.")
                
                with col2:
                    st.subheader("2. Fibo Plan (Pivot)")
                    st.write(f"Swing High: {swing_high:.0f} | Swing Low: {swing_low:.0f}")
                    st.info(f"🎯 Entry: {swing_high - (selisih*0.5):.0f} - {swing_high - (selisih*0.382):.0f}")
                    st.error(f"🛑 Cutloss: < {swing_high - (selisih*0.618):.0f}")
                    st.success(f"🚀 Target: {swing_high:.0f}")
                
                with col3:
                    st.subheader("3. Heikin Ashi")
                    # Ekstra aman menggunakan squeeze() saat membandingkan
                    if float(ha_df['HA_Close'].squeeze().iloc[-1]) > float(ha_df['HA_Open'].squeeze().iloc[-1]):
                        st.success("📈 CANDLE HIJAU - HOLD!")
                    else:
                        st.error("📉 CANDLE MERAH - WAIT/SELL!")

# ==========================================
# MODE 2: SCREENER MASSAL
# ==========================================
elif mode == "Screener Massal":
    st.header("Screener Ichimoku Massal")
    st.write("Menggunakan kalkulasi 2 tahun (Akurat) & Fibo Fraktal.")
    
    if st.button("Mulai Screening"):
        hasil_screener = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(daftar_saham):
            status_text.text(f"Mengecek {ticker}... ({i+1}/{len(daftar_saham)})")
            
            try:
                df = yf.download(ticker, period="730d", progress=False)
                
                if not df.empty and len(df) > 60: 
                    try:
                        harga_terkini = float(yf.Ticker(ticker).fast_info['last_price'])
                    except:
                        harga_terkini = float(df['Close'].squeeze().iloc[-1])

                    ichi = IchimokuIndicator(high=df['High'].squeeze(), low=df['Low'].squeeze())
                    tenkan = float(ichi.ichimoku_conversion_line().squeeze().iloc[-1])
                    kijun = float(ichi.ichimoku_base_line().squeeze().iloc[-1])
                    senkou_a = float(ichi.ichimoku_a().squeeze().iloc[-1])
                    senkou_b = float(ichi.ichimoku_b().squeeze().iloc[-1])
                    
                    if (harga_terkini > max(senkou_a, senkou_b)) and (tenkan > kijun):
                        
                        swing_high, swing_low = get_recent_swings(df, days)
                        selisih = swing_high - swing_low
                        
                        entry_bawah = swing_high - (selisih * 0.5)
                        entry_atas = swing_high - (selisih * 0.382)
                        cutloss = swing_high - (selisih * 0.618)
                        
                        ha_df = calculate_heikin_ashi(df)
                        
                        # Ekstra aman menggunakan squeeze() saat membandingkan
                        if float(ha_df['HA_Close'].squeeze().iloc[-1]) > float(ha_df['HA_Open'].squeeze().iloc[-1]):
                            ha_status = "🟢 HOLD"
                        else:
                            ha_status = "🔴 WAIT / SELL"
                        
                        hasil_screener.append({
                            "Ticker": ticker,
                            "Harga": round(harga_terkini, 0),
                            "Area Entry": f"{round(entry_bawah, 0)} - {round(entry_atas, 0)}",
                            "Target": round(swing_high, 0),
                            "Cutloss (<)": round(cutloss, 0),
                            "Sinyal HA": ha_status
                        })
            except Exception:
                pass 
            
            time.sleep(0.3) 
            progress_bar.progress((i + 1) / len(daftar_saham))
            
        status_text.text("Screening Selesai!")
        
        if hasil_screener:
            st.success(f"🎉 Ketemu! Ada {len(hasil_screener)} saham Uptrend Ichimoku.")
            df_hasil = pd.DataFrame(hasil_screener)
            df_hasil.index = df_hasil.index + 1 
            st.dataframe(df_hasil, width='stretch') 
        else:
            st.error("Tidak ada saham yang memenuhi kriteria Uptrend saat ini.")
