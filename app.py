import streamlit as st
import pandas as pd
import yfinance as yf
from ta.trend import IchimokuIndicator

st.set_page_config(page_title="Screener Saham", layout="wide")
st.title("Screener Saham: Ichi-Fibo-Heikin 🚀")

# 1. BACA FILE CSV UNTUK DAFTAR EMITEN
@st.cache_data # Gunakan cache agar CSV tidak dibaca ulang setiap ada perubahan di layar
def load_emiten():
    try:
        df_emiten = pd.read_csv('emiten.csv')
        # Asumsi nama kolomnya adalah 'ticker' sesuai standar filemu
        return df_emiten['ticker'].tolist()
    except FileNotFoundError:
        return []

daftar_saham = load_emiten()

if not daftar_saham:
    st.error("File 'emiten.csv' tidak ditemukan atau kosong. Pastikan file ada di folder yang sama!")
    st.stop()

# 2. PILIHAN MODE DI SIDEBAR
st.sidebar.header("Pengaturan")
mode = st.sidebar.radio("Pilih Mode:", ["Analisis Satuan", "Screener Massal (Semua Saham)"])
days = st.sidebar.slider("Ambil data berapa hari?", 60, 365, 120)

# ==========================================
# MODE 1: ANALISIS SATUAN (DARI DROPDOWN CSV)
# ==========================================
if mode == "Analisis Satuan":
    st.header("Analisis Spesifik")
    
    # Dropdown mengambil data dari list CSV
    ticker = st.selectbox("Pilih Saham dari CSV:", daftar_saham)
    
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
                
                with col3:
                    st.subheader("3. Heikin Ashi")
                    ha_close = (df['Open'].squeeze() + df['High'].squeeze() + df['Low'].squeeze() + df['Close'].squeeze()) / 4
                    # Pengecekan simpel Heikin Ashi candle terakhir vs hari sebelumnya
                    if ha_close.iloc[-1] > ha_close.iloc[-2]:
                        st.success("📈 CANDLE HIJAU - HOLD!")
                    else:
                        st.error("📉 CANDLE MERAH - SELL!")

# ==========================================
# MODE 2: SCREENER MASSAL (LOOPING SEMUA CSV)
# ==========================================
elif mode == "Screener Massal (Semua Saham)":
    st.header("Screener Ichimoku Massal")
    st.write(f"Mencari saham dengan sinyal *Buy* kuat dari total **{len(daftar_saham)} saham** di file CSV.")
    st.warning("Proses ini bisa memakan waktu beberapa menit tergantung koneksi internetmu.")
    
    if st.button("Mulai Screening"):
        hasil_screener = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(daftar_saham):
            # Update teks progress
            status_text.text(f"Mengecek {ticker}... ({i+1}/{len(daftar_saham)})")
            
            try:
                # Ambil data diem-diem tanpa output terminal
                df = yf.download(ticker, period=f"{days}d", progress=False)
                
                if not df.empty and len(df) > 52: # Pastikan data cukup untuk Ichimoku
                    ichi = IchimokuIndicator(high=df['High'].squeeze(), low=df['Low'].squeeze())
                    tenkan = ichi.ichimoku_conversion_line().iloc[-1]
                    kijun = ichi.ichimoku_base_line().iloc[-1]
                    senkou_a = ichi.ichimoku_a().iloc[-1]
                    senkou_b = ichi.ichimoku_b().iloc[-1]
                    close_price = df['Close'].squeeze().iloc[-1]
                    
                    # LOGIKA SCREENER: Harga di atas awan Kumo & Tenkan di atas Kijun
                    if (close_price > max(senkou_a, senkou_b)) and (tenkan > kijun):
                        
                        # Hitung Fibo untuk yang lolos screening saja
                        high_max = df['High'].squeeze().max()
                        low_min = df['Low'].squeeze().min()
                        selisih = high_max - low_min
                        entry_bawah = high_max - (selisih * 0.5)
                        entry_atas = high_max - (selisih * 0.382)
                        
                        hasil_screener.append({
                            "Ticker": ticker,
                            "Harga Close": round(close_price, 0),
                            "Area Beli": f"{round(entry_bawah, 0)} - {round(entry_atas, 0)}"
                        })
            except Exception:
                pass # Kalau ada saham error/delisting, lewati saja
            
            # Update Bar
            progress_bar.progress((i + 1) / len(daftar_saham))
            
        status_text.text("Screening Selesai!")
        
        # Tampilkan Hasil dalam bentuk Tabel
        if hasil_screener:
            st.success(f"🎉 Ketemu! Ada {len(hasil_screener)} saham yang masuk kriteria Uptrend Ichimoku.")
            st.dataframe(pd.DataFrame(hasil_screener), use_container_width=True)
        else:
            st.error("Tidak ada saham yang memenuhi kriteria Uptrend saat ini.")
