"""
Screener Ichi-Fibo-Heikin Pro  •  v3.0
────────────────────────────────────────
Tambahan v3:
  • Trading Plan Engine — scoring multi-faktor dari posisi harga saat ini
  • Sinyal: STRONG BUY / BUY / WAIT / SELL / STRONG SELL
  • Action plan: entry, cutloss, target, RR, sizing saran
  • Confidence score (0-100) berdasarkan konfluensi sinyal
  • Zone map visual — harga ada di zone mana sekarang
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from ta.trend import IchimokuIndicator

try:
    from scipy.signal import argrelextrema
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ══════════════════════════════════════════════
# KONFIGURASI HALAMAN
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Screener Ichi-Fibo-Heikin Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; }

    /* Trading plan card */
    .plan-card {
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        font-family: monospace;
    }
    .plan-strong-buy  { background:#0d3326; border-left:5px solid #00e676; }
    .plan-buy         { background:#0a2e1a; border-left:5px solid #69f0ae; }
    .plan-wait        { background:#2a2700; border-left:5px solid #ffd600; }
    .plan-sell        { background:#2e1a0a; border-left:5px solid #ff9100; }
    .plan-strong-sell { background:#3e0d0d; border-left:5px solid #ff1744; }

    /* Confidence bar */
    .conf-wrap { background:#1e1e2e; border-radius:8px; height:18px; width:100%; margin:6px 0 12px; }
    .conf-fill  { height:18px; border-radius:8px; transition:width .4s; }

    /* Zone badge */
    .zone-badge {
        display:inline-block;
        padding:3px 10px;
        border-radius:20px;
        font-size:0.78rem;
        font-weight:600;
        margin:2px 3px;
    }
    .zone-active  { background:#00e676; color:#000; }
    .zone-current { background:#ffd600; color:#000; outline:2px solid white; }
    .zone-inactive{ background:#2a2a3e; color:#aaa; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Screener Ichi-Fibo-Heikin Pro v3.0")


# ══════════════════════════════════════════════
# DATACLASS SINYAL TRADING
# ══════════════════════════════════════════════
@dataclass
class TradingSignal:
    sinyal: str           # STRONG BUY / BUY / WAIT / SELL / STRONG SELL
    confidence: int       # 0-100
    zona: str             # posisi price sekarang
    alasan: list[str] = field(default_factory=list)
    aksi: list[str]  = field(default_factory=list)

    # Plan konkret
    entry: Optional[float] = None
    cutloss: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    rr: Optional[float] = None
    sizing_pct: Optional[int] = None   # % modal yang disarankan


# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════
def to_series(x) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return x.squeeze().astype(float)


def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    O = to_series(df["Open"])
    H = to_series(df["High"])
    L = to_series(df["Low"])
    C = to_series(df["Close"])
    ha_close = (O + H + L + C) / 4
    ha_open  = np.zeros(len(df))
    ha_open[0] = (O.iloc[0] + C.iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2
    r = df.copy()
    r["HA_Close"] = ha_close.values
    r["HA_Open"]  = ha_open
    r["HA_High"]  = np.maximum(H.values, np.maximum(ha_open, ha_close.values))
    r["HA_Low"]   = np.minimum(L.values, np.minimum(ha_open, ha_close.values))
    return r


def get_recent_swings(df: pd.DataFrame, days_lookback: int, window: int = 5):
    high = to_series(df["High"]).tail(days_lookback)
    low  = to_series(df["Low"]).tail(days_lookback)
    sh   = float(high.max())
    sl   = float(low.min())
    if HAS_SCIPY:
        order = max(window, 3)
        idx_h = argrelextrema(high.values, np.greater_equal, order=order)[0]
        idx_l = argrelextrema(low.values,  np.less_equal,    order=order)[0]
        if len(idx_h): sh = float(high.iloc[idx_h[-1]])
        if len(idx_l): sl = float(low.iloc[idx_l[-1]])
    else:
        for i in range(len(high)-1-window, window, -1):
            if high.iloc[i] == high.iloc[i-window:i+window+1].max():
                sh = float(high.iloc[i]); break
        for i in range(len(low)-1-window, window, -1):
            if low.iloc[i] == low.iloc[i-window:i+window+1].min():
                sl = float(low.iloc[i]); break
    return sh, sl


# ══════════════════════════════════════════════
# DOWNLOAD & CACHE
# ══════════════════════════════════════════════
@st.cache_data(ttl=900, show_spinner=False)
def download_data(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period="730d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 60:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna(subset=["Open","High","Low","Close","Volume"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def get_last_price(ticker: str, fallback: float) -> float:
    try:
        return float(yf.Ticker(ticker).fast_info["last_price"])
    except Exception:
        return fallback


# ══════════════════════════════════════════════
# CORE ANALYSER
# ══════════════════════════════════════════════
def analyse_ticker(ticker: str, days: int, require_ha_green: bool = False) -> Optional[dict]:
    df = download_data(ticker)
    if df.empty:
        return None

    close_last = float(to_series(df["Close"]).iloc[-1])
    harga      = get_last_price(ticker, close_last)

    # ── Ichimoku ──
    ichi     = IchimokuIndicator(high=to_series(df["High"]), low=to_series(df["Low"]))
    tenkan   = float(to_series(ichi.ichimoku_conversion_line()).iloc[-1])
    kijun    = float(to_series(ichi.ichimoku_base_line()).iloc[-1])
    senkou_a = float(to_series(ichi.ichimoku_a()).iloc[-1])
    senkou_b = float(to_series(ichi.ichimoku_b()).iloc[-1])
    chikou   = float(to_series(df["Close"]).iloc[-27])  # approx chikou 26 lag
    awan_atas  = max(senkou_a, senkou_b)
    awan_bawah = min(senkou_a, senkou_b)

    di_atas_awan      = harga > awan_atas
    di_dalam_awan     = awan_bawah <= harga <= awan_atas
    tenkan_atas_kijun = tenkan > kijun
    chikou_bullish    = harga > chikou   # harga sekarang > close 26 bar lalu

    # ── Fibonacci ──
    sh, sl  = get_recent_swings(df, days)
    diff    = sh - sl
    levels  = {
        "swing_high" : sh,
        "target_2"   : sh + diff * 0.618,
        "target_1"   : sh,
        "fib_236"    : sh - diff * 0.236,
        "entry_atas" : sh - diff * 0.382,
        "entry_bawah": sh - diff * 0.500,
        "cutloss"    : sh - diff * 0.618,
        "swing_low"  : sl,
    }

    # ── Heikin Ashi ──
    ha_df   = calculate_heikin_ashi(df)
    ha_green = float(ha_df["HA_Close"].iloc[-1]) > float(ha_df["HA_Open"].iloc[-1])
    ha_seq   = int(
        (ha_df["HA_Close"] > ha_df["HA_Open"]).iloc[::-1].cumprod().sum()
    )
    # Apakah lower shadow HA terakhir kecil (momentum kuat)?
    ha_last    = ha_df.iloc[-1]
    ha_no_shadow = (float(ha_last["HA_Low"]) >= float(ha_last["HA_Open"]) * 0.998) if ha_green else False

    # ── Volume ──
    vol_s   = to_series(df["Volume"])
    adv20   = float(vol_s.iloc[-21:-1].mean())
    vol_rel = round(float(vol_s.iloc[-1]) / adv20, 2) if adv20 > 0 else 0.0

    # ── RSI sederhana (14) ──
    close_s = to_series(df["Close"])
    delta   = close_s.diff()
    gain    = delta.clip(lower=0).rolling(14).mean()
    loss    = (-delta.clip(upper=0)).rolling(14).mean()
    rsi     = float(100 - 100 / (1 + gain.iloc[-1] / loss.iloc[-1])) if float(loss.iloc[-1]) != 0 else 50.0

    if require_ha_green and not ha_green:
        return None

    return dict(
        ticker=ticker, harga=harga,
        tenkan=tenkan, kijun=kijun, senkou_a=senkou_a, senkou_b=senkou_b,
        awan_atas=awan_atas, awan_bawah=awan_bawah,
        di_atas_awan=di_atas_awan, di_dalam_awan=di_dalam_awan,
        tenkan_atas_kijun=tenkan_atas_kijun, chikou_bullish=chikou_bullish,
        levels=levels,
        sh=sh, sl=sl, diff=diff,
        ha_green=ha_green, ha_seq=ha_seq, ha_no_shadow=ha_no_shadow,
        vol_rel=vol_rel, rsi=rsi,
    )


# ══════════════════════════════════════════════
# TRADING PLAN ENGINE  ← INTI BARU
# ══════════════════════════════════════════════
def build_trading_plan(h: dict) -> TradingSignal:
    """
    Baca posisi harga vs semua level → hasilkan sinyal + action plan.
    Scoring berbasis poin konfluensi (max 10 poin → confidence 100).
    """
    harga  = h["harga"]
    levels = h["levels"]
    score  = 0          # positif = bullish, negatif = bearish
    alasan : list[str] = []
    aksi   : list[str] = []

    # ────────────────────────────────────────
    # A. ICHIMOKU (max +4 / min -4)
    # ────────────────────────────────────────
    if h["di_atas_awan"]:
        score += 2
        alasan.append("✅ Harga di atas awan Ichimoku (bullish zone)")
    elif h["di_dalam_awan"]:
        score -= 1
        alasan.append("⚠️ Harga di dalam awan — zona kabut, volatil")
    else:
        score -= 2
        alasan.append("❌ Harga di bawah awan — bearish zone")

    if h["tenkan_atas_kijun"]:
        score += 1
        alasan.append("✅ Tenkan > Kijun — momentum naik")
    else:
        score -= 1
        alasan.append("⚠️ Tenkan ≤ Kijun — momentum lemah / sideways")

    if h["chikou_bullish"]:
        score += 1
        alasan.append("✅ Chikou span bullish")
    else:
        score -= 1
        alasan.append("⚠️ Chikou span belum konfirmasi")

    # ────────────────────────────────────────
    # B. POSISI vs FIBONACCI (max +3 / min -3)
    # ────────────────────────────────────────
    in_entry_zone = levels["entry_bawah"] <= harga <= levels["entry_atas"]
    below_cutloss = harga < levels["cutloss"]
    above_target  = harga >= levels["target_1"]
    near_sh       = harga > levels["fib_236"]   # sudah di atas 76.4% retracement

    if in_entry_zone:
        score += 3
        alasan.append("🎯 Harga di ZONA ENTRY Fibonacci (38.2%–50%) — sweet spot!")
    elif below_cutloss:
        score -= 3
        alasan.append("🛑 Harga di bawah cutloss Fibonacci — danger zone")
    elif above_target:
        score -= 1
        alasan.append("⚠️ Harga sudah di / atas Swing High — potensi resistance")
    elif near_sh:
        score += 1
        alasan.append("🔼 Harga di atas 76.4% retracement — mendekati target")
    else:
        alasan.append("📍 Harga di luar zona entry optimal saat ini")

    # ────────────────────────────────────────
    # C. HEIKIN ASHI (max +2 / min -2)
    # ────────────────────────────────────────
    if h["ha_green"]:
        score += 1
        alasan.append(f"🕯️ Heikin Ashi hijau {h['ha_seq']} candle berturut-turut")
        if h["ha_no_shadow"]: 
            score += 1
            alasan.append("💪 Candle HA tanpa shadow bawah — momentum sangat kuat")
    else:
        score -= 1
        alasan.append(f"🔴 Heikin Ashi merah {h['ha_seq']} candle — momentum turun")
        if h["ha_seq"] >= 3:
            score -= 1
            alasan.append("⛔ HA merah ≥3 candle berturut — tekanan jual kuat")

    # ────────────────────────────────────────
    # D. VOLUME & RSI (max +2 / min -2)
    # ────────────────────────────────────────
    if h["vol_rel"] >= 1.5:
        score += 1
        alasan.append(f"📊 Volume tinggi {h['vol_rel']:.1f}×ADV20 — konfirmasi pergerakan")
    elif h["vol_rel"] < 0.5:
        score -= 1
        alasan.append(f"📉 Volume rendah {h['vol_rel']:.1f}×ADV20 — pergerakan lemah")

    if 40 <= h["rsi"] <= 60:
        alasan.append(f"📈 RSI netral ({h['rsi']:.0f}) — ruang gerak masih ada")
    elif h["rsi"] > 70:
        score -= 1
        alasan.append(f"🔥 RSI overbought ({h['rsi']:.0f}) — hati-hati reversal")
    elif h["rsi"] < 30:
        score += 1
        alasan.append(f"💧 RSI oversold ({h['rsi']:.0f}) — potensi bouncing")

    # ────────────────────────────────────────
    # KONVERSI SCORE → SINYAL
    # max bullish score = 10, max bearish = -10
    # ────────────────────────────────────────
    confidence = min(100, max(0, int((score + 10) * 5)))   # scale ke 0-100

    if score >= 7:
        sinyal = "STRONG BUY"
        sizing = 100
    elif score >= 4:
        sinyal = "BUY"
        sizing = 75
    elif score >= 1:
        sinyal = "SPECULATIVE BUY"
        sizing = 40
    elif score >= -2:
        sinyal = "WAIT"
        sizing = 0
    elif score >= -5:
        sinyal = "SELL / REDUCE"
        sizing = 0
    else:
        sinyal = "STRONG SELL"
        sizing = 0

    # ────────────────────────────────────────
    # ZONA HARGA SEKARANG
    # ────────────────────────────────────────
    if harga >= levels["target_2"]:
        zona = "Di atas Target 2 (161.8%)"
    elif harga >= levels["target_1"]:
        zona = "Di atas Target 1 / Swing High"
    elif harga >= levels["entry_atas"]:
        zona = "Antara Swing High & Entry Atas (23.6%–38.2%)"
    elif harga >= levels["entry_bawah"]:
        zona = "🎯 ZONA ENTRY (38.2%–50%)"
    elif harga >= levels["cutloss"]:
        zona = "Antara Entry Bawah & Cutloss (50%–61.8%)"
    elif harga >= levels["swing_low"]:
        zona = "Di bawah Cutloss — Bearish"
    else:
        zona = "Di bawah Swing Low — Sangat Bearish"

    # ────────────────────────────────────────
    # ACTION PLAN
    # ────────────────────────────────────────
    if "BUY" in sinyal:
        if in_entry_zone:
            aksi.append(f"🟢 ENTRY sekarang di sekitar {harga:,.0f} (ada di zona entry)")
        else:
            aksi.append(f"⏳ TUNGGU harga masuk zona entry: {levels['entry_bawah']:,.0f} – {levels['entry_atas']:,.0f}")
        aksi.append(f"🛑 CUTLOSS jika close di bawah {levels['cutloss']:,.0f}")
        aksi.append(f"🎯 TARGET 1: {levels['target_1']:,.0f}  |  TARGET 2: {levels['target_2']:,.0f}")
        if sizing > 0:
            aksi.append(f"💰 SIZING SARAN: {sizing}% dari alokasi posisi ini")
        rr_val = (levels["target_1"] - harga) / max(harga - levels["cutloss"], 1)
        aksi.append(f"📊 Risk/Reward saat ini ≈ 1 : {rr_val:.1f}")
    elif "WAIT" in sinyal:
        aksi.append(f"⏳ WAIT — belum ada konfluensi cukup")
        aksi.append(f"👁️ Pantau jika harga turun ke zona entry: {levels['entry_bawah']:,.0f} – {levels['entry_atas']:,.0f}")
        aksi.append(f"👁️ Atau pantau jika Ichimoku & HA mulai konfirmasi bullish")
    else:
        aksi.append(f"🔴 JANGAN tambah posisi — sinyal negatif")
        aksi.append(f"📤 Jika masih pegang: pertimbangkan REDUCE atau EXIT")
        aksi.append(f"❗ Waspadai jika harga close di bawah {levels['cutloss']:,.0f}")

    entry_val = harga if in_entry_zone else levels["entry_atas"]
    rr_final  = (levels["target_1"] - entry_val) / max(entry_val - levels["cutloss"], 1)

    return TradingSignal(
        sinyal=sinyal,
        confidence=confidence,
        zona=zona,
        alasan=alasan,
        aksi=aksi,
        entry=levels["entry_bawah"] if not in_entry_zone else harga,
        cutloss=levels["cutloss"],
        target_1=levels["target_1"],
        target_2=levels["target_2"],
        rr=rr_final,
        sizing_pct=sizing,
    )


# ══════════════════════════════════════════════
# UI: RENDER TRADING PLAN
# ══════════════════════════════════════════════
def render_trading_plan(sig: TradingSignal, harga: float):
    css_class = {
        "STRONG BUY"     : "plan-strong-buy",
        "BUY"            : "plan-buy",
        "SPECULATIVE BUY": "plan-buy",
        "WAIT"           : "plan-wait",
        "SELL / REDUCE"  : "plan-sell",
        "STRONG SELL"    : "plan-strong-sell",
    }.get(sig.sinyal, "plan-wait")

    color_map = {
        "STRONG BUY":"#00e676","BUY":"#69f0ae","SPECULATIVE BUY":"#b9f6ca",
        "WAIT":"#ffd600","SELL / REDUCE":"#ff9100","STRONG SELL":"#ff1744"
    }
    bar_color = color_map.get(sig.sinyal, "#ffd600")

    st.markdown(f"""
    <div class="plan-card {css_class}">
      <div style="font-size:1.6rem;font-weight:800;letter-spacing:1px;margin-bottom:4px;">
        {sig.sinyal}
      </div>
      <div style="font-size:0.85rem;color:#aaa;margin-bottom:8px;">Zona harga: {sig.zona}</div>
      <div style="font-size:0.8rem;color:#888;margin-bottom:2px;">Confidence Score</div>
      <div class="conf-wrap">
        <div class="conf-fill" style="width:{sig.confidence}%;background:{bar_color};"></div>
      </div>
      <div style="font-size:1rem;font-weight:700;color:{bar_color};">{sig.confidence}/100</div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🧠 Analisis Konfluensi")
        for a in sig.alasan:
            st.markdown(f"- {a}")

    with col_b:
        st.markdown("#### 🎯 Action Plan")
        for a in sig.aksi:
            st.markdown(f"- {a}")

    # ── Quick summary metrics ──
    st.divider()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Harga", f"{harga:,.0f}")
    m2.metric("Entry Ideal", f"{sig.entry:,.0f}" if sig.entry else "-")
    m3.metric("Cutloss", f"{sig.cutloss:,.0f}" if sig.cutloss else "-")
    m4.metric("Target 1", f"{sig.target_1:,.0f}" if sig.target_1 else "-")
    m5.metric("Risk/Reward", f"1 : {sig.rr:.1f}" if sig.rr else "-")


# ══════════════════════════════════════════════
# ZONE MAP VISUAL
# ══════════════════════════════════════════════
def render_zone_map(h: dict):
    lv  = h["levels"]
    harga = h["harga"]

    zones = [
        ("Target 2 (161.8%)", lv["target_2"],   "#00e676"),
        ("Target 1 / Swing High", lv["target_1"],"#69f0ae"),
        ("Zona Resistance (23.6%–38.2%)", lv["entry_atas"], "#b2dfdb"),
        ("🎯 Zona Entry (38.2%–50%)", lv["entry_bawah"],    "#ffd600"),
        ("Zona Buffer (50%–61.8%)", lv["cutloss"],          "#ff9100"),
        ("⛔ Cutloss (61.8%)", lv["swing_low"],             "#ff1744"),
    ]

    st.markdown("#### 🗺️ Zone Map — Posisi Harga Sekarang")
    for label, level, color in zones:
        is_current = False
        # cek apakah harga berada di sekitar level ini (±2%)
        for i, (_, lv2, _) in enumerate(zones):
            if i < len(zones) - 1:
                upper = zones[i][1]
                lower = zones[i+1][1]
                if lower <= harga <= upper:
                    if label == zones[i][0]:
                        is_current = True
                    break
        badge_class = "zone-current" if is_current else "zone-active" if harga >= level else "zone-inactive"
        arrow = " ◀ POSISI ANDA" if is_current else ""
        st.markdown(
            f'<span class="zone-badge {badge_class}">{label}: {level:,.0f}</span>{arrow}',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════
# BACA EMITEN CSV
# ══════════════════════════════════════════════
@st.cache_data
def load_emiten() -> list[str]:
    try:
        df_e = pd.read_csv("emiten.csv")
        col  = "ticker" if "ticker" in df_e.columns else df_e.columns[0]
        return df_e[col].astype(str).str.strip().tolist()
    except FileNotFoundError:
        return []

daftar_saham = load_emiten()
if not daftar_saham:
    st.error("❌ File **emiten.csv** tidak ditemukan.")
    st.stop()


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Pengaturan")
    mode = st.radio("Mode:", ["Analisis + Trading Plan", "Screener Massal"])
    st.divider()

    days = st.slider("Lookback Fibonacci (hari):", 30, 365, 120, 10)
    require_ha  = st.toggle("Hanya HA Hijau ✅", value=True)
    min_vol_rel = st.slider("Min. Volume Relatif (×ADV20):", 0.0, 5.0, 0.5, 0.1)
    min_score   = st.slider("Min. Confidence Screener:", 0, 100, 50, 5,
                             help="Hanya tampil saham dengan confidence ≥ nilai ini")
    st.divider()
    st.caption(f"Total emiten: **{len(daftar_saham)}**")
    st.caption("🟢 scipy" if HAS_SCIPY else "🟡 fallback pivot")


# ══════════════════════════════════════════════
# MODE 1: ANALISIS + TRADING PLAN
# ══════════════════════════════════════════════
if mode == "Analisis + Trading Plan":
    st.header("🔍 Analisis Spesifik + Trading Plan")

    c_sel, c_btn = st.columns([3, 1])
    with c_sel:
        ticker = st.selectbox("Pilih Saham:", daftar_saham, label_visibility="collapsed")
    with c_btn:
        run = st.button("Analisis ▶", use_container_width=True, type="primary")

    if run:
        with st.spinner(f"Menganalisis {ticker}…"):
            h = analyse_ticker(ticker, days, require_ha_green=False)

        if h is None:
            st.error("Data tidak cukup atau ticker tidak valid.")
        else:
            sig = build_trading_plan(h)

            # ── Banner sinyal utama ──
            st.divider()
            render_trading_plan(sig, h["harga"])

            # ── Tab detail ──
            st.divider()
            tab1, tab2, tab3 = st.tabs(["☁️ Ichimoku Detail", "📐 Fibonacci Detail", "🕯️ Heikin Ashi + Volume"])

            with tab1:
                df_ichi = pd.DataFrame({
                    "Indikator": ["Tenkan-sen", "Kijun-sen", "Senkou A", "Senkou B",
                                  "Awan Atas", "Awan Bawah", "Posisi Harga"],
                    "Nilai": [
                        f"{h['tenkan']:,.0f}", f"{h['kijun']:,.0f}",
                        f"{h['senkou_a']:,.0f}", f"{h['senkou_b']:,.0f}",
                        f"{h['awan_atas']:,.0f}", f"{h['awan_bawah']:,.0f}",
                        "Di atas awan ✅" if h["di_atas_awan"] else
                        "Dalam awan ⚠️" if h["di_dalam_awan"] else "Di bawah awan ❌",
                    ],
                })
                st.dataframe(df_ichi, hide_index=True, use_container_width=True)

            with tab2:
                lv = h["levels"]
                df_fibo = pd.DataFrame({
                    "Level": ["Target 2 (161.8%)", "Target 1 (Swing High)",
                              "Resistance (23.6%)", "Entry Atas (38.2%)",
                              "Entry Bawah (50%)", "Cutloss (61.8%)", "Swing Low"],
                    "Harga": [f"{lv['target_2']:,.0f}", f"{lv['target_1']:,.0f}",
                               f"{lv['fib_236']:,.0f}", f"{lv['entry_atas']:,.0f}",
                               f"{lv['entry_bawah']:,.0f}", f"{lv['cutloss']:,.0f}",
                               f"{lv['swing_low']:,.0f}"],
                    "Status": [
                        "🎯 Target" if h["harga"] < lv["target_2"] else "✅ Tercapai",
                        "🎯 Target" if h["harga"] < lv["target_1"] else "✅ Tercapai",
                        "─", "🟡 Entry Zone", "🟡 Entry Zone",
                        "🛑 Cutloss", "─",
                    ],
                })
                st.dataframe(df_fibo, hide_index=True, use_container_width=True)
                render_zone_map(h)

            with tab3:
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Heikin Ashi",
                           f"{'🟢 Hijau' if h['ha_green'] else '🔴 Merah'} ({h['ha_seq']} candle)")
                mc2.metric("Volume Relatif", f"{h['vol_rel']:.2f}×",
                           delta="Tinggi" if h["vol_rel"] > 1.5 else "Normal")
                mc3.metric("RSI (14)", f"{h['rsi']:.1f}",
                           delta="Overbought" if h["rsi"] > 70 else
                                 "Oversold" if h["rsi"] < 30 else "Normal")


# ══════════════════════════════════════════════
# MODE 2: SCREENER MASSAL
# ══════════════════════════════════════════════
elif mode == "Screener Massal":
    st.header("🚀 Screener Ichimoku Massal + Sinyal")
    st.caption(
        f"Filter aktif: Ichimoku Uptrend · "
        f"{'HA Hijau · ' if require_ha else ''}"
        f"Vol ≥{min_vol_rel}×ADV20 · Confidence ≥{min_score}"
    )

    if st.button("▶ Mulai Screening", type="primary"):
        hasil_screener: list[dict] = []
        errors: list[str] = []

        pbar    = st.progress(0)
        status  = st.empty()
        holder  = st.empty()
        total   = len(daftar_saham)

        for i, ticker in enumerate(daftar_saham):
            status.text(f"⏳ {ticker}… ({i+1}/{total})")
            try:
                h = analyse_ticker(ticker, days, require_ha_green=require_ha)
                if h and h["di_atas_awan"] and h["tenkan_atas_kijun"] and h["vol_rel"] >= min_vol_rel:
                    sig = build_trading_plan(h)
                    if sig.confidence >= min_score:
                        hasil_screener.append({
                            "Ticker"    : h["ticker"],
                            "Sinyal"    : sig.sinyal,
                            "Confidence": sig.confidence,
                            "Harga"     : int(round(h["harga"])),
                            "Entry"     : f"{int(round(h['levels']['entry_bawah']))}–{int(round(h['levels']['entry_atas']))}",
                            "Target 1"  : int(round(h["levels"]["target_1"])),
                            "Cutloss"   : int(round(h["levels"]["cutloss"])),
                            "R/R"       : f"1:{sig.rr:.1f}" if sig.rr else "-",
                            "HA"        : f"🟢 {h['ha_seq']}c" if h["ha_green"] else f"🔴 {h['ha_seq']}c",
                            "Vol"       : f"{h['vol_rel']:.1f}×",
                            "RSI"       : f"{h['rsi']:.0f}",
                        })
                        df_tmp = pd.DataFrame(hasil_screener).sort_values("Confidence", ascending=False)
                        holder.dataframe(df_tmp, hide_index=True, use_container_width=True)
            except Exception as e:
                errors.append(f"{ticker}: {e}")

            pbar.progress((i+1)/total)
            time.sleep(0.15)

        status.text("✅ Selesai!")
        pbar.progress(1.0)

        if hasil_screener:
            df_hasil = pd.DataFrame(hasil_screener).sort_values("Confidence", ascending=False)
            st.success(f"🎉 {len(df_hasil)} saham lolos — diurutkan dari confidence tertinggi")
            st.dataframe(df_hasil, hide_index=True, use_container_width=True)
            csv = df_hasil.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", csv, "hasil_screener.csv", "text/csv")
        else:
            st.warning("Tidak ada saham yang lolos semua filter.")

        if errors:
            with st.expander(f"⚠️ {len(errors)} error"):
                for e in errors: st.text(e)
