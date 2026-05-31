import streamlit as st
import pandas as pd
import yfinance as yf
from ta.trend import IchimokuIndicator

# Konfigurasi halaman web agar lebih lebar
st.set_page_config(page_title="Screener Ichi-Fibo-Heikin", layout="wide")
st.title("Screener Saham: Ichi-Fibo-Heikin 🚀")

# ==========================================
# 1. BACA FILE CSV UNTUK DAFTAR EMITEN
# ==========================================
@st.cache_data # Cache agar tidak loading CSV berulang kali
def load_emiten():
    try:
        df_emiten = pd.read_csv('emiten.csv')
        # Mengecek apakah ada kolom bernama 'ticker'. Jika tidak ada, ambil kolom pertama.
        if 'ticker' in df_emiten.columns:
            return df_emiten['ticker'].astype(str).tolist()
        else:
            return df_emiten.iloc[:, 0].astype(str).tolist()
    except FileNotFoundError:
        return []

daftar_saham = load_emiten()

if not daftar_saham:
    st.error("File 'emiten.csv' tidak ditemukan atau kosong. Pastikan file ada di folder yang sama!")
    st.stop()

# ==========================================
# 2. PENGATURAN SIDEBAR KIRI
# ==========================================
st.sidebar.header("Pengaturan")
mode = st.sidebar.radio("Pilih Mode:", ["Analisis Satuan", "Screener Massal (Semua Saham)"])
days = st.sidebar.slider("Ambil data berapa hari ke belakang?", 60, 365, 120)

# ==========================================
# MODE 1: ANALISIS SATUAN
# ==========================================
if mode == "Analisis Satuan":
    st.header("Analisis Spesifik")
    
    # Dropdown memilih saham dari daftar CSV
    ticker = st.selectbox("Pilih Saham:", daftar_saham)
    
    if st.button("Analisis Saham Ini"):
        with st.spinner(f"Mengambil data {ticker}..."):
            df = yf.download(ticker, period=f"{days}d", progress=False)
            
            if df.empty:
                st.error("Data saham tidak tersedia di Yahoo Finance.")
            else:
                # -- Hitung Ichimoku --
                ichi = IchimokuIndicator(high=df['High'].squeeze(), low=df['Low'].squeeze())
                df['Tenkan'] = ichi.ichimoku_conversion_line()
                df['Kijun'] = ichi.ichimoku_base_line()
                df['Senkou_A'] = ichi.ichimoku_a()
                df['Senkou_B'] = ichi.ichimoku_b()
                
                terakhir = df.iloc[-1]
                close_price = terakhir['Close'].squeeze()
                di_atas_awan = close_price > max(terakhir['Senkou_A'].squeeze(), terakhir['Senkou_B'].squeeze())
                tenkan_cross_kijun = terakhir['Tenkan'].squeeze() > terakhir['Kijun'].squeeze()
                
                # Menampilkan hasil dalam 3 kolom agar rapi
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("1. Ichimoku")
                    if di_atas_awan and tenkan_cross_kijun:
                        st.success("🟢 UPTREND KUAT!")
                    else:
                        st.warning("🟡 Belum Uptrend Sempurna.")
                
                with col2:
                    st.subheader("2. Fibo Plan")
                    swing_high = df['High'].squeeze().max()
                    swing_low = df['Low'].squeeze().min()
                    selisih = swing_high - swing_low
                    st.info(f"🎯 Entry: {swing_high - (selisih*0.5):.0f} - {swing_high - (selisih*0.382):.0f}")
                    st.error(f"🛑 Cutloss: < {swing_high - (selisih*0.618):.0f}")
                    st.success(f"🚀 Target: {swing_high:.0f}")
                
                with col3:
                    st.subheader("3. Heikin Ashi")
                    ha_close = (df['Open'].squeeze() + df['High'].squeeze() + df['Low'].squeeze() + df['Close'].squeeze()) / 4
                    if len(ha_close) > 1 and ha_close.iloc[-1] > ha_close.iloc[-2]:
                        st.success("📈 CANDLE HIJAU - HOLD!")
                    else:
                        st.error("📉 CANDLE MERAH - WAIT/SELL!")

# ==========================================
# MODE 2: SCREENER MASSAL (LOOPING CSV)
# ==========================================
elif mode == "Screener Massal (Semua Saham)":
    st.header("Screener Ichimoku Massal")
    st.write(f"Mencari saham dengan sinyal *Buy* kuat dari total **{len(daftar_saham)} saham** di file CSV.")
    st.warning("Proses ini memakan waktu beberapa saat. Silakan tunggu bilah progres hingga penuh.")
    
    if st.button("Mulai Screening"):
        hasil_screener = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(daftar_saham):
            status_text.text(f"Mengecek {ticker}... ({i+1}/{len(daftar_saham)})")
            
            try:
                # Ambil data diem-diem
                df = yf.download(ticker, period=f"{days}d", progress=False)
                
                # Syarat minimal hari perdagangan untuk Ichimoku
                if not df.empty and len(df) > 52: 
                    ichi = IchimokuIndicator(high=df['High'].squeeze(), low=df['Low'].squeeze())
                    tenkan = ichi.ichimoku_conversion_line().iloc[-1]
                    kijun = ichi.ichimoku_base_line().iloc[-1]
                    senkou_a = ichi.ichimoku_a().iloc[-1]
                    senkou_b = ichi.ichimoku_b().iloc[-1]
                    close_price = df['Close'].squeeze().iloc[-1]
                    
                    # LOGIKA SCREENER: Harga di atas awan Kumo & Tenkan di atas Kijun
                    if (close_price > max(senkou_a, senkou_b)) and (tenkan > kijun):
                        
                        # Hitung Fibo
                        high_max = df['High'].squeeze().max()
                        low_min = df['Low'].squeeze().min()
                        selisih = high_max - low_min
                        
                        entry_bawah = high_max - (selisih * 0.5)
                        entry_atas = high_max - (selisih * 0.382)
                        cutloss = high_max - (selisih * 0.618)
                        target = high_max 
                        
                        # Hitung Heikin Ashi sederhana
                        ha_close = (df['Open'].squeeze() + df['High'].squeeze() + df['Low'].squeeze() + df['Close'].squeeze()) / 4
                        if ha_close.iloc[-1] > ha_close.iloc[-2]:
                            ha_status = "🟢 HOLD"
                        else:
                            ha_status = "🔴 WAIT / SELL"
                        
                        hasil_screener.append({
                            "Ticker": ticker,
                            "Harga": round(close_price, 0),
                            "Area Entry": f"{round(entry_bawah, 0)} - {round(entry_atas, 0)}",
                            "Target": round(target, 0),
                            "Stop Loss (<)": round(cutloss, 0),
                            "Sinyal HA": ha_status
                        })
            except Exception:
                pass # Abaikan saham yang datanya tidak lengkap/error
            
            # Update Progress Bar
            progress_bar.progress((i + 1) / len(daftar_saham))
            
        status_text.text("Screening Selesai!")
        
        # Tampilkan Tabel
        if hasil_screener:
            st.success(f"🎉 Ketemu! Ada {len(hasil_screener)} saham yang masuk kriteria Uptrend Ichimoku.")
            df_hasil = pd.DataFrame(hasil_screener)
            # Menampilkan index dari 1
            df_hasil.index = df_hasil.index + 1 
            st.dataframe(df_hasil, use_container_width=True)
        else:
            st.error("Tidak ada saham yang memenuhi kriteria Uptrend saat ini.")
