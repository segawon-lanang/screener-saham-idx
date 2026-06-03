"""
Screener Ichi-Fibo-Heikin Pro  •  v4.0
═══════════════════════════════════════
v4 upgrades:
  KECEPATAN  : Parallel download via ThreadPoolExecutor (5–8× lebih cepat)
               Batch yf.download multi-ticker sekaligus
               Cache per-ticker + cache batch screener
  AKURASI    : +EMA 20/50 trend filter & cross signal
               +MACD histogram untuk momentum shift
               +ATR-based dynamic sizing (volatility-adjusted)
               +Stochastic RSI untuk timing entry presisi
               +Trend strength: ADX
               +Support/Resistance berbasis Volume Profile (simplified)
               Scoring dibobot (weighted), bukan flat +1/-1
  UI/UX      : Full dark-mode custom CSS (professional trading terminal look)
               Sidebar collapsible dengan badge sinyal
               Gauge chart confidence (SVG)
               Price ladder visual (zone map interaktif)
               Tabel hasil screener dengan color-coded sinyal
               Toast notifikasi saat screener selesai
"""

import time
import concurrent.futures
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from ta.trend import IchimokuIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import StochasticOscillator
from ta.volatility import AverageTrueRange

try:
    from scipy.signal import argrelextrema
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG & DARK TERMINAL CSS
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="IFH Pro Screener",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

/* ── Root overrides ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0b0e17 !important;
    color: #c9d1d9 !important;
}
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d !important;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
.block-container { padding: 1rem 2rem !important; max-width: 1400px; }

/* ── Typography ── */
h1,h2,h3 { font-family:'IBM Plex Sans',sans-serif !important; }
code, .mono { font-family:'IBM Plex Mono',monospace !important; }

/* ── Signal banner ── */
.sig-banner {
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    border: 1px solid;
}
.sig-STRONG-BUY  { background:#071f14; border-color:#00e676; }
.sig-BUY         { background:#081a10; border-color:#4caf50; }
.sig-SPEC-BUY    { background:#0f1f0a; border-color:#8bc34a; }
.sig-WAIT        { background:#1a1600; border-color:#ffc107; }
.sig-SELL        { background:#1a0e00; border-color:#ff9800; }
.sig-STRONG-SELL { background:#1f0707; border-color:#f44336; }

.sig-label {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 2px;
    white-space: nowrap;
}
.sig-meta { flex: 1; }
.sig-zona { font-size:0.78rem; color:#8b949e; margin-top:4px; }

/* ── Confidence bar ── */
.cbar-wrap { background:#161b22; border-radius:6px; height:10px; width:100%; margin:8px 0 4px; }
.cbar-fill  { height:10px; border-radius:6px; }

/* ── Metric cards ── */
.mcard {
    background:#161b22;
    border:1px solid #21262d;
    border-radius:10px;
    padding:1rem 1.2rem;
    text-align:center;
}
.mcard-val { font-size:1.5rem; font-weight:700; font-family:'IBM Plex Mono',monospace; }
.mcard-lbl { font-size:0.72rem; color:#8b949e; text-transform:uppercase; letter-spacing:1px; }

/* ── Price ladder ── */
.ladder { width:100%; border-collapse:collapse; font-family:'IBM Plex Mono',monospace; font-size:0.82rem; }
.ladder tr { border-bottom:1px solid #21262d; }
.ladder td { padding:7px 10px; }
.ldr-current { background:#1c2128; font-weight:700; }
.ldr-pill {
    display:inline-block; padding:2px 10px;
    border-radius:20px; font-size:0.72rem; font-weight:600;
}
.pill-entry  { background:#1b4332; color:#6ee7b7; border:1px solid #059669; }
.pill-target { background:#1e3a5f; color:#93c5fd; border:1px solid #3b82f6; }
.pill-cut    { background:#450a0a; color:#fca5a5; border:1px solid #dc2626; }
.pill-now    { background:#312107; color:#fcd34d; border:1px solid #d97706; }
.pill-gray   { background:#1c2128; color:#8b949e; border:1px solid #30363d; }

/* ── Screener table badges ── */
.badge {
    display:inline-block; padding:2px 9px; border-radius:12px;
    font-size:0.72rem; font-weight:700; font-family:'IBM Plex Mono',monospace;
}
.b-sb  { background:#00e676; color:#000; }
.b-b   { background:#4caf50; color:#fff; }
.b-sp  { background:#8bc34a; color:#000; }
.b-w   { background:#ffc107; color:#000; }
.b-s   { background:#ff9800; color:#000; }
.b-ss  { background:#f44336; color:#fff; }

/* ── Dividers & misc ── */
hr { border-color:#21262d !important; }
[data-testid="metric-container"] {
    background:#161b22; border:1px solid #21262d;
    border-radius:10px; padding:0.8rem 1rem;
}
.stTabs [data-baseweb="tab"] { font-family:'IBM Plex Sans',sans-serif; }
.stAlert { border-radius:8px !important; }
[data-testid="stDataFrame"] { border:1px solid #21262d; border-radius:8px; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# DATACLASS
# ══════════════════════════════════════════════════════════════
@dataclass
class TradingSignal:
    sinyal: str
    css_key: str               # untuk class CSS
    confidence: int            # 0-100
    score_raw: float           # raw weighted score
    zona: str
    alasan: list[str]  = field(default_factory=list)
    aksi:   list[str]  = field(default_factory=list)
    entry:    Optional[float] = None
    cutloss:  Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    rr:       Optional[float] = None
    sizing_pct: Optional[int] = None
    atr_sizing: Optional[str] = None


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def s(x) -> pd.Series:
    """Paksa jadi 1-D float Series."""
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return x.squeeze().astype(float)


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    O,H,L,C = s(df["Open"]), s(df["High"]), s(df["Low"]), s(df["Close"])
    hac = (O+H+L+C)/4
    hao = np.empty(len(df)); hao[0] = (O.iloc[0]+C.iloc[0])/2
    for i in range(1, len(df)):
        hao[i] = (hao[i-1] + hac.iloc[i-1]) / 2
    r = df.copy()
    r["HA_C"] = hac.values; r["HA_O"] = hao
    r["HA_H"] = np.maximum(H.values, np.maximum(hao, hac.values))
    r["HA_L"] = np.minimum(L.values, np.minimum(hao, hac.values))
    return r


def recent_swings(df: pd.DataFrame, days: int, win: int = 5):
    hi = s(df["High"]).tail(days)
    lo = s(df["Low"]).tail(days)
    sh, sl = float(hi.max()), float(lo.min())
    if HAS_SCIPY:
        oi = max(win, 3)
        ih = argrelextrema(hi.values, np.greater_equal, order=oi)[0]
        il = argrelextrema(lo.values, np.less_equal,    order=oi)[0]
        if len(ih): sh = float(hi.iloc[ih[-1]])
        if len(il): sl = float(lo.iloc[il[-1]])
    else:
        for i in range(len(hi)-1-win, win, -1):
            if hi.iloc[i] == hi.iloc[i-win:i+win+1].max(): sh=float(hi.iloc[i]); break
        for i in range(len(lo)-1-win, win, -1):
            if lo.iloc[i] == lo.iloc[i-win:i+win+1].min(): sl=float(lo.iloc[i]); break
    return sh, sl


def rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    g = d.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=period-1, adjust=False).mean()
    return 100 - 100/(1 + g/l.replace(0, 1e-9))


# ══════════════════════════════════════════════════════════════
# LIQUIDITY & HEALTH FILTER
# ══════════════════════════════════════════════════════════════
@dataclass
class HealthCheck:
    lolos: bool
    alasan: str          # kenapa diloloskan / ditolak
    flags: list[str]     # daftar masalah yang ditemukan


def health_check(df: pd.DataFrame, ticker: str,
                 min_price: float      = 50,
                 min_adv_juta: float   = 500,     # Rp juta/hari
                 min_active_days: float = 0.70,   # % hari aktif (ada transaksi)
                 max_flat_streak: int   = 10,      # max hari berturut close = open
                 ) -> HealthCheck:
    """
    Deteksi saham tidur / tidak likuid / zombie / FCA-like.
    Return HealthCheck — caller putuskan mau skip atau warning.
    """
    flags: list[str] = []
    close  = s(df["Close"])
    high_s = s(df["High"])
    low_s  = s(df["Low"])
    vol_s  = s(df["Volume"])
    open_s = s(df["Open"])

    last_60  = df.tail(60)
    close_60 = s(last_60["Close"])
    vol_60   = s(last_60["Volume"])
    high_60  = s(last_60["High"])
    low_60   = s(last_60["Low"])

    # ── 1. Harga terlalu murah (saham gocap / penny) ──
    last_price = float(close.iloc[-1])
    if last_price < min_price:
        flags.append(f"💀 Penny stock ({last_price:.0f} < {min_price:.0f}) — rentan manipulasi")

    # ── 2. Nilai transaksi harian terlalu kecil ──
    # ADV value = rata-rata (close × volume) 20 hari terakhir
    adv_value_juta = float((close_60 * vol_60).tail(20).mean()) / 1_000_000
    if adv_value_juta < min_adv_juta:
        flags.append(f"😴 Likuiditas rendah (ADV Rp{adv_value_juta:.0f}jt < Rp{min_adv_juta:.0f}jt)")

    # ── 3. Frekuensi hari aktif — saham tidur ──
    # Hari aktif = volume > 0 DAN high != low (ada pergerakan)
    active_mask  = (vol_60 > 0) & (high_60 != low_60)
    active_ratio = float(active_mask.mean())
    if active_ratio < min_active_days:
        flags.append(f"😴 Saham tidur ({active_ratio*100:.0f}% hari aktif, min {min_active_days*100:.0f}%)")

    # ── 4. Flat streak — banyak hari close == open (price frozen) ──
    flat_days = int((abs(close_60 - s(last_60["Open"])) < 1).sum())
    if flat_days > max_flat_streak:
        flags.append(f"🧟 Price frozen ({flat_days} hari close≈open dalam 60 hari)")

    # ── 5. Volume anomali ekstrem (pump & dump pattern) ──
    # Spike volume 10× ADV diikuti volume mati
    vol_20  = float(vol_60.tail(20).mean())
    vol_max = float(vol_60.max())
    vol_now = float(vol_60.tail(5).mean())
    if vol_max > vol_20 * 10 and vol_now < vol_20 * 0.3:
        flags.append("⚠️ Pump & dump pattern — volume spike lalu mati")

    # ── 6. Harga sudah turun > 80% dari high 1 tahun (zombie) ──
    high_1y = float(s(df["High"]).tail(252).max())
    if high_1y > 0 and last_price < high_1y * 0.20:
        flags.append(f"🧟 Zombie — harga {last_price:.0f} sudah turun {(1-last_price/high_1y)*100:.0f}% dari high 1 tahun ({high_1y:.0f})")

    # ── 7. Volatilitas terlalu rendah (saham dikunci / tidak bergerak) ──
    # ATR% rata-rata < 0.3% = nyaris tidak bergerak
    atr_pct_avg = float(
        AverageTrueRange(high_60, low_60, close_60, window=14)
        .average_true_range().tail(10).mean()
    ) / last_price * 100 if last_price > 0 else 0
    if atr_pct_avg < 0.3:
        flags.append(f"😴 Volatilitas sangat rendah (ATR {atr_pct_avg:.2f}%) — saham tidak bergerak")

    # ── 8. Bid-ask spread proxy: banyak candle doji (high-low sangat kecil) ──
    doji_ratio = float(((high_60 - low_60) < last_price * 0.003).mean())
    if doji_ratio > 0.5:
        flags.append(f"😴 {doji_ratio*100:.0f}% candle doji — spread sangat sempit / tidak ada minat")

    lolos  = len(flags) == 0
    alasan = "✅ Sehat" if lolos else f"❌ {len(flags)} masalah ditemukan"
    return HealthCheck(lolos=lolos, alasan=alasan, flags=flags)


# ══════════════════════════════════════════════════════════════
# DOWNLOAD  (parallel-capable, cached)
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=900, show_spinner=False)
def download_one(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period="730d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 60: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna(subset=["Open","High","Low","Close","Volume"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def live_price(ticker: str, fallback: float) -> float:
    try:    return float(yf.Ticker(ticker).fast_info["last_price"])
    except: return fallback


def download_batch_parallel(tickers: list[str], max_workers: int = 8) -> dict[str, pd.DataFrame]:
    """Download semua ticker secara paralel."""
    result = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut = {ex.submit(download_one, t): t for t in tickers}
        for f in concurrent.futures.as_completed(fut):
            t = fut[f]
            try:    result[t] = f.result()
            except: result[t] = pd.DataFrame()
    return result


# ══════════════════════════════════════════════════════════════
# FULL ANALYSER  (semua indikator dalam satu pass)
# ══════════════════════════════════════════════════════════════
def analyse(ticker: str, days: int, df: Optional[pd.DataFrame] = None) -> Optional[dict]:
    if df is None:
        df = download_one(ticker)
    if df is None or df.empty:
        return None

    close  = s(df["Close"])
    high   = s(df["High"])
    low    = s(df["Low"])
    volume = s(df["Volume"])
    harga  = live_price(ticker, float(close.iloc[-1]))

    # ── Ichimoku ──
    ichi    = IchimokuIndicator(high=high, low=low)
    tenkan  = float(s(ichi.ichimoku_conversion_line()).iloc[-1])
    kijun   = float(s(ichi.ichimoku_base_line()).iloc[-1])
    sa      = float(s(ichi.ichimoku_a()).iloc[-1])
    sb      = float(s(ichi.ichimoku_b()).iloc[-1])
    awan_hi = max(sa, sb); awan_lo = min(sa, sb)
    chikou_ref  = float(close.iloc[-27]) if len(close) > 27 else float(close.iloc[0])
    di_atas     = harga > awan_hi
    di_dalam    = awan_lo <= harga <= awan_hi
    tk_kj       = tenkan > kijun
    chikou_bull = harga > chikou_ref

    # ── EMA 20 / 50 ──
    ema20 = float(EMAIndicator(close, window=20).ema_indicator().iloc[-1])
    ema50 = float(EMAIndicator(close, window=50).ema_indicator().iloc[-1])
    ema_bull   = ema20 > ema50
    price_ema20 = harga > ema20

    # ── MACD ──
    macd_obj   = MACD(close)
    macd_hist  = float(s(macd_obj.macd_diff()).iloc[-1])
    macd_prev  = float(s(macd_obj.macd_diff()).iloc[-2])
    macd_cross_up = macd_hist > 0 and macd_prev <= 0   # baru cross up
    macd_bull     = macd_hist > 0

    # ── ADX (trend strength) ──
    adx_obj = ADXIndicator(high, low, close, window=14)
    adx_val = float(s(adx_obj.adx()).iloc[-1])
    adx_strong = adx_val > 25

    # ── Stochastic RSI ──
    rsi14 = rsi_series(close, 14)
    stoch_period = 14
    rsi_min = rsi14.rolling(stoch_period).min()
    rsi_max = rsi14.rolling(stoch_period).max()
    stoch_rsi = ((rsi14 - rsi_min) / (rsi_max - rsi_min + 1e-9) * 100)
    srsi_k = float(stoch_rsi.rolling(3).mean().iloc[-1])
    srsi_d = float(stoch_rsi.rolling(3).mean().rolling(3).mean().iloc[-1])
    srsi_oversold   = srsi_k < 20
    srsi_overbought = srsi_k > 80
    srsi_cross_up   = srsi_k > srsi_d and srsi_k < 50

    # ── ATR (volatility sizing) ──
    atr_val = float(AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1])
    atr_pct = atr_val / harga * 100 if harga > 0 else 0

    # ── RSI 14 standard ──
    rsi_now = float(rsi14.iloc[-1])

    # ── Volume ──
    adv20   = float(volume.iloc[-21:-1].mean())
    vol_rel = round(float(volume.iloc[-1]) / adv20, 2) if adv20 > 0 else 0.0

    # ── Volume Profile support (simplified: VWAP area) ──
    vwap = float((close * volume).rolling(20).sum().iloc[-1] /
                 volume.rolling(20).sum().iloc[-1])
    price_above_vwap = harga > vwap

    # ── Fibonacci ──
    sh, sl = recent_swings(df, days)
    diff = sh - sl
    lvl = {
        "target_2"   : sh + diff * 0.618,
        "target_1"   : sh,
        "fib_236"    : sh - diff * 0.236,
        "entry_atas" : sh - diff * 0.382,
        "entry_bawah": sh - diff * 0.500,
        "cutloss"    : sh - diff * 0.618,
        "swing_low"  : sl,
    }
    in_entry  = lvl["entry_bawah"] <= harga <= lvl["entry_atas"]
    below_cut = harga < lvl["cutloss"]
    above_sh  = harga >= lvl["target_1"]
    near_sh   = harga > lvl["fib_236"]

    # ── Heikin Ashi ──
    ha = heikin_ashi(df)
    ha_green   = float(ha["HA_C"].iloc[-1]) > float(ha["HA_O"].iloc[-1])
    ha_seq     = int((ha["HA_C"] > ha["HA_O"]).iloc[::-1].cumprod().sum())
    ha_no_tail = (float(ha["HA_L"].iloc[-1]) >= float(ha["HA_O"].iloc[-1]) * 0.999) if ha_green else False

    # ── Candle Quality Score (CQS) — pembeda akumulasi vs distribusi ──
    # CQS = (close - low) / (high - low) → 1.0 = close di high, 0.0 = close di low
    # Dihitung untuk 3 candle terakhir dan dirata-rata
    cqs_vals = []
    for i in [-1, -2, -3]:
        c_hi = float(high.iloc[i]); c_lo = float(low.iloc[i]); c_cl = float(close.iloc[i])
        rng = c_hi - c_lo
        cqs_vals.append((c_cl - c_lo) / rng if rng > 0 else 0.5)
    cqs = round(float(np.mean(cqs_vals)), 3)          # 0–1

    # ── Distribution trap detection ──
    # Tanda-tanda: volume spike TAPI close dekat low (institusi jual ke retail)
    vol_spike       = vol_rel >= 2.0
    dist_trap       = vol_spike and cqs < 0.4          # volume besar, close rendah
    accum_confirm   = vol_spike and cqs > 0.6          # volume besar, close tinggi → genuine akumulasi

    # ── Upper shadow ratio — candle berbentuk shooting star / bearish engulf ──
    o_last = float(s(df["Open"]).iloc[-1])
    h_last = float(high.iloc[-1]); l_last = float(low.iloc[-1]); c_last = float(close.iloc[-1])
    body   = abs(c_last - o_last)
    upper_shadow = h_last - max(c_last, o_last)
    upper_shadow_ratio = upper_shadow / body if body > 0 else 0.0
    shooting_star = upper_shadow_ratio > 2.0 and vol_rel > 1.0   # upper shadow > 2× body

    # ── OBV trend (on-balance volume) — apakah volume mendukung harga? ──
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    obv_ema5  = obv.ewm(span=5,  adjust=False).mean()
    obv_ema20 = obv.ewm(span=20, adjust=False).mean()
    obv_bull  = float(obv_ema5.iloc[-1]) > float(obv_ema20.iloc[-1])   # OBV trend naik

    # ── Early Bird signals (Ichimoku pre-breakout) ──
    # Senkou A dan B 26 bar ke depan (future cloud)
    sa_series = s(ichi.ichimoku_a())
    sb_series = s(ichi.ichimoku_b())
    # Kumo twist: titik dimana Senkou A dan B akan bersilangan dalam 26 bar ke depan
    # Kita deteksi dari perubahan tanda (sa - sb) dalam window terakhir
    sa_sb_diff = sa_series - sb_series
    # Cari apakah ada perubahan tanda di 5 bar terakhir (twist baru terjadi)
    kumo_twist_recent = bool(
        (sa_sb_diff.iloc[-1] > 0) and (sa_sb_diff.iloc[-6] <= 0)
    ) if len(sa_sb_diff) >= 6 else False
    # Kumo menyempit: selisih SA-SB mengecil (tren menuju twist)
    kumo_width_now  = abs(float(sa_sb_diff.iloc[-1]))
    kumo_width_5ago = abs(float(sa_sb_diff.iloc[-6])) if len(sa_sb_diff) >= 6 else kumo_width_now
    kumo_narrowing  = kumo_width_now < kumo_width_5ago * 0.7   # menyempit >30%
    # Harga mendekati awan dari bawah (dalam 2% dari bawah awan)
    price_near_cloud = (not di_atas) and (harga >= awan_lo * 0.98)
    # TK cross baru terjadi (dalam 5 bar terakhir)
    tk_series    = s(ichi.ichimoku_conversion_line())
    kj_series    = s(ichi.ichimoku_base_line())
    tk_kj_diff   = tk_series - kj_series
    tk_cross_new = bool(
        (tk_kj_diff.iloc[-1] > 0) and (tk_kj_diff.iloc[-6] <= 0)
    ) if len(tk_kj_diff) >= 6 else False
    # Chikou hampir bebas (dalam 3% dari price bar 26 lalu)
    chikou_near_free = (not chikou_bull) and (harga >= chikou_ref * 0.97)

    # Early bird score (0–5): makin tinggi makin early
    early_score = sum([
        kumo_twist_recent,   # awan baru berubah warna
        kumo_narrowing,      # awan menipis
        price_near_cloud,    # harga mendekati awan
        tk_cross_new,        # TK cross baru
        chikou_near_free,    # chikou hampir bebas
    ])

    return dict(
        ticker=ticker, harga=harga,
        # ichimoku
        tenkan=tenkan, kijun=kijun, sa=sa, sb=sb,
        awan_hi=awan_hi, awan_lo=awan_lo,
        di_atas=di_atas, di_dalam=di_dalam,
        tk_kj=tk_kj, chikou_bull=chikou_bull,
        # ema
        ema20=ema20, ema50=ema50, ema_bull=ema_bull, price_ema20=price_ema20,
        # macd
        macd_hist=macd_hist, macd_cross_up=macd_cross_up, macd_bull=macd_bull,
        # adx
        adx=adx_val, adx_strong=adx_strong,
        # srsi
        srsi_k=srsi_k, srsi_d=srsi_d,
        srsi_oversold=srsi_oversold, srsi_overbought=srsi_overbought, srsi_cross_up=srsi_cross_up,
        # atr
        atr=atr_val, atr_pct=atr_pct,
        # rsi & vol
        rsi=rsi_now, vol_rel=vol_rel, vwap=vwap, price_above_vwap=price_above_vwap,
        # fibo
        sh=sh, sl=sl, diff=diff, lvl=lvl,
        in_entry=in_entry, below_cut=below_cut, above_sh=above_sh, near_sh=near_sh,
        # ha
        ha_green=ha_green, ha_seq=ha_seq, ha_no_tail=ha_no_tail,
        # candle quality & distribution
        cqs=cqs, dist_trap=dist_trap, accum_confirm=accum_confirm,
        shooting_star=shooting_star, upper_shadow_ratio=upper_shadow_ratio,
        obv_bull=obv_bull,
        # early bird
        early_score=early_score,
        kumo_twist_recent=kumo_twist_recent, kumo_narrowing=kumo_narrowing,
        price_near_cloud=price_near_cloud, tk_cross_new=tk_cross_new,
        chikou_near_free=chikou_near_free,
    )


# ══════════════════════════════════════════════════════════════
# WEIGHTED SCORING ENGINE  (v4 — bobot presisi)
# ══════════════════════════════════════════════════════════════
#
# Bobot dirancang berdasarkan kontribusi nyata tiap indikator:
#   Ichimoku cloud position  → paling dominan (trend primer)
#   Fibonacci entry zone     → timing entry terbaik
#   MACD + EMA               → konfirmasi momentum
#   HA + ADX                 → strength filter
#   StochRSI                 → presisi timing
#   Volume + VWAP            → konfirmasi institusional

WEIGHTS = {
    "cloud_pos"   : 2.5,   # di atas / dalam / di bawah awan
    "tk_kj"       : 1.0,   # tenkan > kijun
    "chikou"      : 0.8,   # chikou bullish
    "fib_zone"    : 2.5,   # di zona entry / di bawah cutloss
    "ema_trend"   : 1.2,   # ema20 > ema50
    "price_ema"   : 0.8,   # harga > ema20
    "macd"        : 1.5,   # macd hist positif
    "macd_cross"  : 0.5,   # macd cross up (bonus)
    "adx"         : 0.5,   # ADX kuat
    "ha"          : 1.0,   # ha green
    "ha_seq"      : 0.5,   # ha berturut konsisten
    "ha_no_tail"  : 0.3,   # momentum penuh
    "srsi_timing" : 1.0,   # stoch rsi timing
    "volume"      : 0.8,   # volume konfirmasi
    "vwap"        : 0.6,   # di atas VWAP
    "rsi_extreme" : 0.5,   # oversold/overbought
}
MAX_SCORE = sum(WEIGHTS.values())   # ~16.3

def build_plan(h: dict) -> TradingSignal:
    score  = 0.0
    alasan : list[str] = []
    aksi   : list[str] = []
    lvl    = h["lvl"]

    # ── A. Ichimoku cloud ──
    if h["di_atas"]:
        score += WEIGHTS["cloud_pos"]
        alasan.append("✅ Di atas awan Ichimoku — bullish zone primer")
    elif h["di_dalam"]:
        score -= WEIGHTS["cloud_pos"] * 0.4
        alasan.append("⚠️ Dalam awan — zona kabut, volatil")
    else:
        score -= WEIGHTS["cloud_pos"]
        alasan.append("❌ Di bawah awan — bearish zone")

    if h["tk_kj"]:
        score += WEIGHTS["tk_kj"]
        alasan.append("✅ Tenkan > Kijun — momentum naik")
    else:
        score -= WEIGHTS["tk_kj"]
        alasan.append("⚠️ Tenkan ≤ Kijun — momentum lemah")

    if h["chikou_bull"]:
        score += WEIGHTS["chikou"]
        alasan.append("✅ Chikou span konfirmasi bullish")
    else:
        score -= WEIGHTS["chikou"] * 0.5
        alasan.append("⚠️ Chikou belum konfirmasi")

    # ── B. Fibonacci zone ──
    if h["in_entry"]:
        score += WEIGHTS["fib_zone"]
        alasan.append("🎯 Harga di ZONA ENTRY Fibo (38.2%–50%) — sweet spot!")
    elif h["below_cut"]:
        score -= WEIGHTS["fib_zone"]
        alasan.append("🛑 Harga di bawah cutloss Fibonacci — danger!")
    elif h["above_sh"]:
        score -= WEIGHTS["fib_zone"] * 0.4
        alasan.append("⚠️ Di atas Swing High — potensi resistance kuat")
    elif h["near_sh"]:
        score += WEIGHTS["fib_zone"] * 0.4
        alasan.append("🔼 Di atas 76.4% retracement — mendekati target")
    else:
        alasan.append(f"📍 Harga di luar zona entry (entry: {lvl['entry_bawah']:,.0f}–{lvl['entry_atas']:,.0f})")

    # ── C. EMA trend ──
    if h["ema_bull"]:
        score += WEIGHTS["ema_trend"]
        alasan.append(f"✅ EMA20 ({h['ema20']:,.0f}) > EMA50 ({h['ema50']:,.0f}) — uptrend struktural")
    else:
        score -= WEIGHTS["ema_trend"]
        alasan.append(f"❌ EMA20 < EMA50 — struktur downtrend")

    if h["price_ema20"]:
        score += WEIGHTS["price_ema"]
        alasan.append("✅ Harga di atas EMA20 — short-term bullish")
    else:
        score -= WEIGHTS["price_ema"] * 0.5
        alasan.append("⚠️ Harga di bawah EMA20")

    # ── D. MACD ──
    if h["macd_cross_up"]:
        score += WEIGHTS["macd"] + WEIGHTS["macd_cross"]
        alasan.append(f"🚀 MACD baru cross UP — sinyal momentum kuat! (hist:{h['macd_hist']:+.2f})")
    elif h["macd_bull"]:
        score += WEIGHTS["macd"]
        alasan.append(f"✅ MACD histogram positif ({h['macd_hist']:+.2f}) — momentum bullish")
    else:
        score -= WEIGHTS["macd"]
        alasan.append(f"❌ MACD histogram negatif ({h['macd_hist']:+.2f}) — momentum bearish")

    # ── E. ADX trend strength ──
    if h["adx_strong"]:
        score += WEIGHTS["adx"]
        alasan.append(f"💪 ADX {h['adx']:.1f} — trend kuat (>25)")
    else:
        alasan.append(f"➡️ ADX {h['adx']:.1f} — trend lemah/sideways (<25)")

    # ── F. Heikin Ashi ──
    if h["ha_green"]:
        score += WEIGHTS["ha"]
        if h["ha_seq"] >= 5:
            score += WEIGHTS["ha_seq"]
            alasan.append(f"🕯️ HA hijau {h['ha_seq']} candle — momentum sangat konsisten")
        else:
            alasan.append(f"🕯️ HA hijau {h['ha_seq']} candle")
        if h["ha_no_tail"]:
            score += WEIGHTS["ha_no_tail"]
            alasan.append("💎 HA tanpa shadow bawah — tekanan jual nol")
    else:
        score -= WEIGHTS["ha"]
        if h["ha_seq"] >= 3:
            score -= WEIGHTS["ha_seq"]
            alasan.append(f"🔴 HA merah {h['ha_seq']} candle — tekanan jual dominan")
        else:
            alasan.append(f"🔴 HA merah {h['ha_seq']} candle")

    # ── G. Stochastic RSI (timing presisi) ──
    if h["srsi_cross_up"]:
        score += WEIGHTS["srsi_timing"]
        alasan.append(f"⏱️ StochRSI cross up di area rendah (K:{h['srsi_k']:.0f}) — timing entry bagus")
    elif h["srsi_oversold"]:
        score += WEIGHTS["srsi_timing"] * 0.6
        alasan.append(f"💧 StochRSI oversold ({h['srsi_k']:.0f}) — potensi bouncing")
    elif h["srsi_overbought"]:
        score -= WEIGHTS["srsi_timing"]
        alasan.append(f"🔥 StochRSI overbought ({h['srsi_k']:.0f}) — hati-hati reversal")
    else:
        alasan.append(f"📊 StochRSI netral (K:{h['srsi_k']:.0f} D:{h['srsi_d']:.0f})")

    # ── H. Volume & VWAP ──
    if h["accum_confirm"]:
        score += WEIGHTS["volume"] * 1.5
        alasan.append(f"🏦 Volume spike {h['vol_rel']:.1f}× + CQS {h['cqs']:.2f} — GENUINE AKUMULASI smart money!")
    elif h["dist_trap"]:
        score -= WEIGHTS["volume"] * 2.0
        alasan.append(f"🚨 DISTRIBUTION TRAP! Vol {h['vol_rel']:.1f}× tapi CQS {h['cqs']:.2f} — institusi jualan ke retail!")
    elif h["vol_rel"] >= 1.5:
        score += WEIGHTS["volume"]
        alasan.append(f"📊 Volume {h['vol_rel']:.1f}×ADV20 — konfirmasi pergerakan")
    elif h["vol_rel"] < 0.5:
        score -= WEIGHTS["volume"] * 0.6
        alasan.append(f"📉 Volume sepi {h['vol_rel']:.1f}×ADV20 — lemah keyakinan")
    else:
        alasan.append(f"📊 Volume {h['vol_rel']:.1f}×ADV20 — normal")

    if h["shooting_star"]:
        score -= 1.0
        alasan.append(f"⚠️ Shooting star! Upper shadow {h['upper_shadow_ratio']:.1f}× body — potensi reversal")

    if h["obv_bull"]:
        score += WEIGHTS["vwap"] * 0.8
        alasan.append("✅ OBV trend naik — volume mendukung harga")
    else:
        score -= WEIGHTS["vwap"] * 0.3
        alasan.append("⚠️ OBV divergen — volume tidak mendukung kenaikan")

    if h["price_above_vwap"]:
        score += WEIGHTS["vwap"]
        alasan.append(f"✅ Di atas VWAP 20d ({h['vwap']:,.0f}) — buyer kontrol")
    else:
        score -= WEIGHTS["vwap"] * 0.5
        alasan.append(f"⚠️ Di bawah VWAP ({h['vwap']:,.0f})")

    # ── RSI extreme ──
    if h["rsi"] < 30:
        score += WEIGHTS["rsi_extreme"]
        alasan.append(f"💧 RSI oversold ({h['rsi']:.0f}) — area murah historis")
    elif h["rsi"] > 70:
        score -= WEIGHTS["rsi_extreme"]
        alasan.append(f"🔥 RSI overbought ({h['rsi']:.0f}) — area mahal")
    else:
        alasan.append(f"📈 RSI {h['rsi']:.0f} — normal range")

    # ── Normalize score ke 0-100 ──
    confidence = int(min(100, max(0, (score / MAX_SCORE) * 100 + 50)))

    # ── Sinyal ──
    norm = score / MAX_SCORE   # -1 to +1
    if norm >= 0.55:
        sinyal, css = "STRONG BUY",  "STRONG-BUY";  sizing = 100
    elif norm >= 0.30:
        sinyal, css = "BUY",         "BUY";          sizing = 75
    elif norm >= 0.10:
        sinyal, css = "SPEC. BUY",   "SPEC-BUY";     sizing = 40
    elif norm >= -0.10:
        sinyal, css = "WAIT",        "WAIT";          sizing = 0
    elif norm >= -0.35:
        sinyal, css = "SELL",        "SELL";          sizing = 0
    else:
        sinyal, css = "STRONG SELL", "STRONG-SELL";  sizing = 0

    # ── Zona ──
    p = h["harga"]
    if p >= lvl["target_2"]:    zona = "Di atas Target 2 (161.8%)"
    elif p >= lvl["target_1"]:  zona = "Di area Swing High / Target 1"
    elif p >= lvl["entry_atas"]:zona = "Antara SH & Entry Atas (23.6%–38.2%)"
    elif p >= lvl["entry_bawah"]:zona = "🎯 ZONA ENTRY OPTIMAL (38.2%–50%)"
    elif p >= lvl["cutloss"]:   zona = "Antara Entry Bawah & Cutloss (50%–61.8%)"
    elif p >= lvl["swing_low"]: zona = "⛔ Di bawah Cutloss — Bearish"
    else:                        zona = "💀 Di bawah Swing Low — Sangat Bearish"

    # ── ATR-based sizing ──
    atr_pct = h["atr_pct"]
    if atr_pct < 1.5:    atr_label = f"Volatilitas Rendah ({atr_pct:.1f}% ATR) → bisa full sizing"
    elif atr_pct < 3.0:  atr_label = f"Volatilitas Sedang ({atr_pct:.1f}% ATR) → sizing 50-75%"
    else:                atr_label = f"Volatilitas Tinggi ({atr_pct:.1f}% ATR) → hati-hati, sizing 25-40%"

    # ── Action plan — distribution trap override ──
    rr_val = (lvl["target_1"] - p) / max(p - lvl["cutloss"], 1)

    # Hard override: kalau distribution trap terdeteksi → paksa WAIT/SELL
    if h["dist_trap"] and "BUY" in sinyal:
        sinyal = "WAIT"
        css    = "WAIT"
        sizing = 0
        aksi.append("🚨 DISTRIBUTION TRAP terdeteksi — sinyal BUY DIBATALKAN!")
        aksi.append(f"🔍 Volume {h['vol_rel']:.1f}×ADV tapi CQS {h['cqs']:.2f} — close terlalu dekat LOW")
        aksi.append("⏳ Tunggu 1–3 hari konfirmasi: apakah harga lanjut naik atau turun")
        aksi.append(f"❗ Aman entry hanya jika CQS > 0.6 pada volume tinggi berikutnya")
    elif h["shooting_star"] and "BUY" in sinyal:
        aksi.append("⚠️ Ada pola shooting star — pertimbangkan sizing lebih kecil atau tunggu konfirmasi")
        aksi.append(f"🛑 CUTLOSS diperketat: jika close < {lvl['cutloss']:,.0f}")
        if h["in_entry"]:
            aksi.append(f"🟡 Entry dengan sizing 50% saja, sisanya tunggu hari berikutnya")
        else:
            aksi.append(f"⏳ TUNGGU harga masuk zona entry: {lvl['entry_bawah']:,.0f} – {lvl['entry_atas']:,.0f}")
        aksi.append(f"🎯 TARGET 1: {lvl['target_1']:,.0f}   TARGET 2: {lvl['target_2']:,.0f}")
        aksi.append(f"📐 R/R saat ini ≈ 1 : {rr_val:.1f}")
    elif "BUY" in sinyal:
        if h["accum_confirm"]:
            aksi.append("🏦 Akumulasi institusional terkonfirmasi — sinyal lebih kuat dari biasa!")
        if h["in_entry"]:
            aksi.append(f"🟢 ENTRY sekarang — harga {p:,.0f} ada di zona sweet spot")
        else:
            aksi.append(f"⏳ TUNGGU pullback ke zona entry: {lvl['entry_bawah']:,.0f} – {lvl['entry_atas']:,.0f}")
        aksi.append(f"🛑 CUTLOSS jika daily close < {lvl['cutloss']:,.0f}")
        aksi.append(f"🎯 TARGET 1: {lvl['target_1']:,.0f}   TARGET 2: {lvl['target_2']:,.0f}")
        aksi.append(f"📐 R/R saat ini ≈ 1 : {rr_val:.1f}")
        aksi.append(f"💰 {atr_label}")
        if sizing:
            aksi.append(f"🎲 Sizing saran: {sizing}% dari alokasi posisi")
    elif "WAIT" in sinyal:
        if not h["dist_trap"]:
            aksi.append(f"⏳ WAIT — konfluensi belum cukup kuat")
            aksi.append(f"👁️ Set alert di zona entry: {lvl['entry_bawah']:,.0f} – {lvl['entry_atas']:,.0f}")
            aksi.append("👁️ Tunggu MACD cross up + HA hijau + CQS > 0.6 sebagai konfirmasi")
    else:
        aksi.append("🔴 JANGAN masuk posisi baru — sinyal bearish dominan")
        aksi.append("📤 Jika pegang: pertimbangkan REDUCE atau EXIT bertahap")
        aksi.append(f"❗ Level bahaya: close di bawah {lvl['cutloss']:,.0f}")

    return TradingSignal(
        sinyal=sinyal, css_key=css, confidence=confidence,
        score_raw=round(score, 2), zona=zona,
        alasan=alasan, aksi=aksi,
        entry=lvl["entry_bawah"], cutloss=lvl["cutloss"],
        target_1=lvl["target_1"], target_2=lvl["target_2"],
        rr=rr_val, sizing_pct=sizing, atr_sizing=atr_label,
    )


# ══════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════
COLOR_MAP = {
    "STRONG BUY": "#00e676", "BUY": "#4caf50", "SPEC. BUY": "#8bc34a",
    "WAIT": "#ffc107", "SELL": "#ff9800", "STRONG SELL": "#f44336",
}

def render_banner(sig: TradingSignal, harga: float):
    col = COLOR_MAP.get(sig.sinyal, "#ffc107")
    pct = sig.confidence
    st.markdown(f"""
    <div class="sig-banner sig-{sig.css_key}">
      <div>
        <div class="sig-label" style="color:{col}">{sig.sinyal}</div>
        <div class="sig-zona">{sig.zona}</div>
      </div>
      <div class="sig-meta">
        <div style="font-size:0.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px;">
          Confidence Score
        </div>
        <div class="cbar-wrap">
          <div class="cbar-fill" style="width:{pct}%;background:{col};"></div>
        </div>
        <div style="font-size:1.1rem;font-weight:700;color:{col};font-family:'IBM Plex Mono',monospace;">
          {pct}/100 &nbsp;·&nbsp; score raw: {sig.score_raw:+.2f}
        </div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:1.8rem;font-weight:700;font-family:'IBM Plex Mono',monospace;">
          {harga:,.0f}
        </div>
        <div style="font-size:0.72rem;color:#8b949e;">HARGA TERKINI</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_price_ladder(h: dict, sig: TradingSignal):
    p   = h["harga"]
    lvl = h["lvl"]

    rows = [
        ("Target 2 (161.8%)",         lvl["target_2"],    "pill-target"),
        ("Target 1 / Swing High",      lvl["target_1"],    "pill-target"),
        ("Resistance (23.6%)",         lvl["fib_236"],     "pill-gray"),
        ("Entry Atas (38.2%)",         lvl["entry_atas"],  "pill-entry"),
        ("Entry Bawah (50%)",          lvl["entry_bawah"], "pill-entry"),
        ("Cutloss (61.8%)",            lvl["cutloss"],     "pill-cut"),
        ("Swing Low",                  lvl["swing_low"],   "pill-gray"),
    ]

    # sisipkan harga saat ini di posisi yang benar
    ladder_rows = []
    inserted = False
    for label, level, pill in rows:
        if not inserted and p >= level:
            ladder_rows.append(("NOW", p, "pill-now", True))
            inserted = True
        ladder_rows.append((label, level, pill, False))
    if not inserted:
        ladder_rows.append(("NOW", p, "pill-now", True))

    html = '<table class="ladder"><tbody>'
    for label, level, pill, is_now in ladder_rows:
        cls = 'class="ldr-current"' if is_now else ""
        pct_from_now = ((level - p) / p * 100) if not is_now else 0
        pct_str = f'<span style="color:#8b949e;font-size:0.72rem;">{pct_from_now:+.1f}%</span>' if not is_now else ""
        html += f"""
        <tr {cls}>
          <td><span class="ldr-pill {pill}">{label}</span></td>
          <td style="text-align:right;font-weight:{'700' if is_now else '400'}">{level:,.0f}</td>
          <td style="text-align:right;">{pct_str}</td>
        </tr>"""
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)


def score_badge_html(sinyal: str) -> str:
    cls = {
        "STRONG BUY":"b-sb","BUY":"b-b","SPEC. BUY":"b-sp",
        "WAIT":"b-w","SELL":"b-s","STRONG SELL":"b-ss"
    }.get(sinyal, "b-w")
    return f'<span class="badge {cls}">{sinyal}</span>'


# ══════════════════════════════════════════════════════════════
# CHART ENGINE  (Plotly — candlestick + Ichimoku + Fibo + Volume)
# ══════════════════════════════════════════════════════════════
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CHART_THEME = dict(
    bg       = "#0b0e17",
    bg2      = "#161b22",
    grid     = "#21262d",
    text     = "#c9d1d9",
    green    = "#00e676",
    red      = "#f44336",
    tenkan   = "#e91e63",
    kijun    = "#2196f3",
    sa       = "rgba(0,230,118,0.15)",
    sb       = "rgba(244,67,54,0.15)",
    ema20    = "#ff9800",
    ema50    = "#9c27b0",
    fib_col  = "rgba(255,214,0,0.55)",
    vwap     = "#00bcd4",
)

def build_chart(ticker: str, h: dict, df: pd.DataFrame, candle_type: str = "Candle") -> go.Figure:
    """
    Buat chart lengkap:
      Panel atas (70%): candlestick/HA + Ichimoku cloud + EMA20/50 + Fibonacci levels + VWAP
      Panel bawah (30%): volume bar + ADV20 line + volume profile background
    """
    CT = CHART_THEME
    close  = s(df["Close"])
    high_s = s(df["High"])
    low_s  = s(df["Low"])
    open_s = s(df["Open"])
    vol_s  = s(df["Volume"])
    dates  = df.index

    # Ichimoku series (full history)
    ichi   = IchimokuIndicator(high=high_s, low=low_s)
    tk_s   = s(ichi.ichimoku_conversion_line())
    kj_s   = s(ichi.ichimoku_base_line())
    sa_s   = s(ichi.ichimoku_a())
    sb_s   = s(ichi.ichimoku_b())

    # EMA
    ema20_s = EMAIndicator(close, window=20).ema_indicator()
    ema50_s = EMAIndicator(close, window=50).ema_indicator()

    # VWAP 20d rolling
    vwap_s = (close * vol_s).rolling(20).sum() / vol_s.rolling(20).sum()

    # Heikin Ashi
    ha_df  = heikin_ashi(df)

    # ADV20
    adv20_s = vol_s.rolling(20).mean()

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.72, 0.28],
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    # ── Kumo (cloud) ──
    # Isi cloud hijau (SA > SB)
    fig.add_trace(go.Scatter(
        x=list(dates) + list(dates[::-1]),
        y=list(sa_s) + list(sb_s[::-1]),
        fill="toself",
        fillcolor="rgba(0,200,100,0.12)",
        line=dict(width=0),
        name="Kumo Bullish", showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    # Isi cloud merah (SB > SA)
    fig.add_trace(go.Scatter(
        x=list(dates) + list(dates[::-1]),
        y=list(sb_s) + list(sa_s[::-1]),
        fill="toself",
        fillcolor="rgba(244,67,54,0.10)",
        line=dict(width=0),
        name="Kumo Bearish", showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    # Senkou A & B borders
    fig.add_trace(go.Scatter(x=dates, y=sa_s, line=dict(color="rgba(0,200,100,0.5)", width=1),
                             name="Senkou A", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=sb_s, line=dict(color="rgba(244,67,54,0.5)", width=1),
                             name="Senkou B", showlegend=True), row=1, col=1)

    # ── Candlestick atau Heikin Ashi ──
    if candle_type == "Heikin Ashi":
        o_src = ha_df["HA_O"]; c_src = ha_df["HA_C"]
        h_src = ha_df["HA_H"]; l_src = ha_df["HA_L"]
        cname = "Heikin Ashi"
    else:
        o_src = open_s; c_src = close; h_src = high_s; l_src = low_s
        cname = "Candle"

    fig.add_trace(go.Candlestick(
        x=dates, open=o_src, high=h_src, low=l_src, close=c_src,
        name=cname,
        increasing_line_color=CT["green"], increasing_fillcolor=CT["green"],
        decreasing_line_color=CT["red"],   decreasing_fillcolor=CT["red"],
        line=dict(width=1),
    ), row=1, col=1)

    # ── Tenkan & Kijun ──
    fig.add_trace(go.Scatter(x=dates, y=tk_s,
                             line=dict(color=CT["tenkan"], width=1.2, dash="solid"),
                             name="Tenkan"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=kj_s,
                             line=dict(color=CT["kijun"], width=1.5, dash="solid"),
                             name="Kijun"), row=1, col=1)

    # ── EMA 20 / 50 ──
    fig.add_trace(go.Scatter(x=dates, y=ema20_s,
                             line=dict(color=CT["ema20"], width=1, dash="dot"),
                             name="EMA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ema50_s,
                             line=dict(color=CT["ema50"], width=1, dash="dot"),
                             name="EMA50"), row=1, col=1)

    # ── VWAP ──
    fig.add_trace(go.Scatter(x=dates, y=vwap_s,
                             line=dict(color=CT["vwap"], width=1, dash="dashdot"),
                             name="VWAP 20d"), row=1, col=1)

    # ── Fibonacci horizontal lines ──
    lvl = h["lvl"]
    fib_lines = [
        ("Target 2 (161.8%)", lvl["target_2"],    "#42a5f5", "dot"),
        ("Target 1 / SH",     lvl["target_1"],    "#42a5f5", "solid"),
        ("Res. 23.6%",        lvl["fib_236"],     CT["fib_col"], "dash"),
        ("Entry Atas 38.2%",  lvl["entry_atas"],  "#66bb6a", "dash"),
        ("Entry Bwh 50%",     lvl["entry_bawah"], "#66bb6a", "solid"),
        ("Cutloss 61.8%",     lvl["cutloss"],     "#ef5350", "solid"),
        ("Swing Low",         lvl["swing_low"],   "#ef5350", "dot"),
    ]
    x_start = dates[max(0, len(dates)-60)]  # tampilkan dari 60 bar terakhir
    x_end   = dates[-1]
    for label, level, color, dash in fib_lines:
        fig.add_shape(type="line", x0=x_start, x1=x_end, y0=level, y1=level,
                      line=dict(color=color, width=1, dash=dash), row=1, col=1)
        fig.add_annotation(x=x_end, y=level, text=f" {label}: {level:,.0f}",
                           showarrow=False, xanchor="left", font=dict(size=9, color=color),
                           row=1, col=1)

    # ── Harga sekarang line ──
    fig.add_hline(y=h["harga"], line=dict(color="#ffd600", width=1.5, dash="dash"),
                  annotation_text=f" NOW {h['harga']:,.0f}",
                  annotation_font_color="#ffd600", annotation_font_size=10,
                  row=1, col=1)

    # ── Volume bars ──
    vol_colors = [CT["green"] if float(close.iloc[i]) >= float(open_s.iloc[i])
                  else CT["red"] for i in range(len(dates))]
    fig.add_trace(go.Bar(
        x=dates, y=vol_s,
        marker_color=vol_colors, marker_opacity=0.6,
        name="Volume", showlegend=False,
    ), row=2, col=1)

    # ADV20 line
    fig.add_trace(go.Scatter(x=dates, y=adv20_s,
                             line=dict(color="#ffd600", width=1.2, dash="dot"),
                             name="ADV20", showlegend=True), row=2, col=1)

    # ── Layout ──
    fig.update_layout(
        paper_bgcolor=CT["bg"], plot_bgcolor=CT["bg2"],
        font=dict(color=CT["text"], size=11),
        margin=dict(l=60, r=140, t=40, b=20),
        height=640,
        xaxis_rangeslider_visible=False,
        legend=dict(
            bgcolor="rgba(0,0,0,0)", font=dict(size=10),
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
        ),
        title=dict(text=f"{ticker}  ·  {candle_type}", font=dict(size=14), x=0.01),
    )
    for ax in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        fig.update_layout(**{ax: dict(
            gridcolor=CT["grid"], zerolinecolor=CT["grid"],
            tickfont=dict(color=CT["text"], size=10),
        )})
    fig.update_xaxes(showspikes=True, spikecolor=CT["grid"], spikethickness=1)
    fig.update_yaxes(showspikes=True, spikecolor=CT["grid"], spikethickness=1)

    return fig


# ══════════════════════════════════════════════════════════════
# VOLUME PROFILE  (price histogram + POC + VAH + VAL)
# ══════════════════════════════════════════════════════════════

def build_volume_profile(df: pd.DataFrame, h: dict, bins: int = 40) -> go.Figure:
    """
    Volume profile: histogram horizontal volume per price level.
    POC  = Price Of Control (harga dengan volume terbesar)
    VAH  = Value Area High  (70% volume upper bound)
    VAL  = Value Area Low   (70% volume lower bound)
    """
    CT   = CHART_THEME
    close  = s(df["Close"])
    high_s = s(df["High"])
    low_s  = s(df["Low"])
    vol_s  = s(df["Volume"])

    # Distribusi volume ke price bucket (pakai typical price per candle)
    typical  = (high_s + low_s + close) / 3
    price_min, price_max = float(typical.min()), float(typical.max())
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    vol_per_bin = np.zeros(bins)
    for i in range(len(typical)):
        idx = min(int((float(typical.iloc[i]) - price_min) /
                      (price_max - price_min + 1e-9) * bins), bins - 1)
        vol_per_bin[idx] += float(vol_s.iloc[i])

    # POC
    poc_idx = int(np.argmax(vol_per_bin))
    poc     = bin_centers[poc_idx]

    # Value Area (70% dari total volume di sekitar POC)
    total_vol  = vol_per_bin.sum()
    target_vol = total_vol * 0.70
    va_lo_idx, va_hi_idx = poc_idx, poc_idx
    accumulated = vol_per_bin[poc_idx]
    while accumulated < target_vol:
        can_lo = va_lo_idx > 0
        can_hi = va_hi_idx < bins - 1
        if not can_lo and not can_hi:
            break
        add_lo = vol_per_bin[va_lo_idx - 1] if can_lo else -1
        add_hi = vol_per_bin[va_hi_idx + 1] if can_hi else -1
        if add_hi >= add_lo and can_hi:
            va_hi_idx += 1
            accumulated += vol_per_bin[va_hi_idx]
        elif can_lo:
            va_lo_idx -= 1
            accumulated += vol_per_bin[va_lo_idx]
        else:
            break

    vah = bin_centers[va_hi_idx]
    val = bin_centers[va_lo_idx]

    # ── Warna per bar: merah di bawah VAL, hijau di atas VAH, abu di VA, kuning di POC ──
    bar_colors = []
    for i, bc in enumerate(bin_centers):
        if i == poc_idx:                bar_colors.append("#ffd600")
        elif bc >= val and bc <= vah:   bar_colors.append("rgba(100,181,246,0.7)")
        elif bc > vah:                  bar_colors.append("rgba(0,230,118,0.55)")
        else:                           bar_colors.append("rgba(239,83,80,0.55)")

    fig = go.Figure()

    # Volume bars (horizontal)
    fig.add_trace(go.Bar(
        x=vol_per_bin,
        y=bin_centers,
        orientation="h",
        marker_color=bar_colors,
        name="Volume per level",
        hovertemplate="Harga: %{y:,.0f}<br>Volume: %{x:,.0f}<extra></extra>",
    ))

    # POC line
    fig.add_hline(y=poc, line=dict(color="#ffd600", width=2, dash="solid"),
                  annotation_text=f"POC {poc:,.0f}",
                  annotation_font_color="#ffd600", annotation_font_size=10)
    # VAH
    fig.add_hline(y=vah, line=dict(color="#42a5f5", width=1.2, dash="dash"),
                  annotation_text=f"VAH {vah:,.0f}",
                  annotation_font_color="#42a5f5", annotation_font_size=10)
    # VAL
    fig.add_hline(y=val, line=dict(color="#42a5f5", width=1.2, dash="dash"),
                  annotation_text=f"VAL {val:,.0f}",
                  annotation_font_color="#42a5f5", annotation_font_size=10)

    # Harga sekarang
    fig.add_hline(y=h["harga"], line=dict(color="#ff9800", width=2, dash="dot"),
                  annotation_text=f"NOW {h['harga']:,.0f}",
                  annotation_font_color="#ff9800", annotation_font_size=10)

    # Fibonacci key levels
    for label, level, color in [
        ("Entry Atas", h["lvl"]["entry_atas"], "#66bb6a"),
        ("Entry Bwh",  h["lvl"]["entry_bawah"],"#66bb6a"),
        ("Cutloss",    h["lvl"]["cutloss"],     "#ef5350"),
        ("Target 1",   h["lvl"]["target_1"],    "#42a5f5"),
    ]:
        fig.add_hline(y=level, line=dict(color=color, width=1, dash="dot"),
                      annotation_text=f"{label} {level:,.0f}",
                      annotation_font_color=color, annotation_font_size=9)

    fig.update_layout(
        paper_bgcolor=CT["bg"], plot_bgcolor=CT["bg2"],
        font=dict(color=CT["text"], size=11),
        height=500,
        margin=dict(l=60, r=140, t=40, b=20),
        title=dict(text="Volume Profile (70% Value Area)", font=dict(size=13), x=0.01),
        xaxis=dict(title="Volume", gridcolor=CT["grid"], tickfont=dict(color=CT["text"], size=9)),
        yaxis=dict(title="Harga", gridcolor=CT["grid"], tickfont=dict(color=CT["text"], size=9)),
        showlegend=False,
    )
    return fig, poc, vah, val


def render_charts(ticker: str, h: dict):
    """Render chart panel lengkap — dipanggil dari mode Analisis dan Screener inline."""
    df = download_one(ticker)
    if df is None or df.empty:
        st.warning("Data tidak tersedia untuk chart."); return

    # ── State keys unik per ticker ──
    key_ct   = f"_ct_{ticker}"
    key_bars = f"_bars_{ticker}"
    if key_ct   not in st.session_state: st.session_state[key_ct]   = "Candle"
    if key_bars not in st.session_state: st.session_state[key_bars] = 120

    col_ct, col_bars = st.columns([2, 3])
    with col_ct:
        # Pakai index bukan key — tidak trigger re-render ke root
        ct_idx = st.radio(
            "Tipe candle:", ["Candle", "Heikin Ashi"],
            horizontal=True,
            index=["Candle", "Heikin Ashi"].index(st.session_state[key_ct]),
        )
        st.session_state[key_ct] = ct_idx

    with col_bars:
        bar_opts = [60, 90, 120, 180, 252, 365, 500, 730]
        cur_bars = st.session_state[key_bars]
        if cur_bars not in bar_opts: cur_bars = 120
        bars_val = st.select_slider(
            "Tampilkan berapa bar:",
            options=bar_opts,
            value=cur_bars,
        )
        st.session_state[key_bars] = bars_val

    candle_type = st.session_state[key_ct]
    chart_bars  = st.session_state[key_bars]
    df_chart    = df.tail(chart_bars)

    col_chart, col_vp = st.columns([3, 1])

    with col_chart:
        fig = build_chart(ticker, h, df_chart, candle_type)
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    with col_vp:
        fig_vp, poc, vah, val = build_volume_profile(df_chart, h)
        st.plotly_chart(fig_vp, use_container_width=True)

        # Summary volume profile
        harga = h["harga"]
        st.markdown(f"""
        <div style="font-size:0.8rem;line-height:2;">
          <span style="color:#ffd600;">● POC</span>: {poc:,.0f}<br>
          <span style="color:#42a5f5;">▲ VAH</span>: {vah:,.0f}
            {'✅ di bawah' if harga < vah else '⚠️ di atas'}<br>
          <span style="color:#42a5f5;">▼ VAL</span>: {val:,.0f}
            {'✅ di atas' if harga > val else '⚠️ di bawah'}<br>
          <span style="color:#ff9800;">● NOW</span>: {harga:,.0f}<br>
          <br>
          <span style="font-size:0.75rem;color:#8b949e;">
          {"🟢 Harga dalam Value Area — support kuat" if val <= harga <= vah
           else "🔼 Di atas VA — buyer dominan" if harga > vah
           else "🔽 Di bawah VA — seller dominan"}
          </span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD EMITEN  (support kolom sektor opsional)
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_emiten():
    try:
        df_e = pd.read_csv("emiten.csv")
        col  = "ticker" if "ticker" in df_e.columns else df_e.columns[0]
        tickers = df_e[col].astype(str).str.strip().tolist()
        # Kolom sektor opsional — bisa "sector", "sektor", "industry"
        sektor_col = next((c for c in df_e.columns
                           if c.lower() in ["sector","sektor","industry","industri"]), None)
        sektor_map = {}
        if sektor_col:
            sektor_map = dict(zip(df_e[col].astype(str).str.strip(),
                                  df_e[sektor_col].astype(str).str.strip()))
        return tickers, sektor_map
    except FileNotFoundError:
        return [], {}

daftar_saham, sektor_map = load_emiten()
if not daftar_saham:
    st.error("❌ **emiten.csv** tidak ditemukan. Letakkan di folder yang sama.")
    st.stop()

# Fallback sektor map pakai prefix ticker IDX (BBCA→Perbankan, dll)
IDX_SEKTOR_PREFIX = {
    "BB":"Perbankan","BT":"Perbankan","BN":"Perbankan","BS":"Perbankan",
    "TL":"Telekomunikasi","XL":"Telekomunikasi","IS":"Telekomunikasi",
    "AS":"Asuransi","PR":"Properti","LP":"Properti","PW":"Properti",
    "CN":"Consumer","MY":"Consumer","UN":"Consumer","CL":"Consumer",
    "IN":"Industri","KL":"Kimia","TM":"Tambang","AN":"Tambang","IT":"Tambang",
    "MD":"Media","EM":"Energi","PG":"Energi","AR":"Agribisnis","SA":"Agribisnis",
}
def get_sektor(ticker: str) -> str:
    if ticker in sektor_map:
        return sektor_map[ticker]
    prefix = ticker[:2] if len(ticker) >= 2 else ticker
    return IDX_SEKTOR_PREFIX.get(prefix, "Lainnya")


# ══════════════════════════════════════════════════════════════
# SENTIMENT ENGINE  (Anthropic API — news summary per ticker)
# ══════════════════════════════════════════════════════════════
import json, re

@st.cache_data(ttl=3600, show_spinner=False)   # cache 1 jam
def get_sentiment(ticker: str) -> dict:
    """
    Panggil Claude API dengan web_search untuk cari berita terbaru
    lalu hasilkan sentiment score + ringkasan.
    """
    prompt = f"""Kamu adalah analis fundamental saham Indonesia.
Cari berita terbaru (maks 7 hari terakhir) tentang saham {ticker} di Bursa Efek Indonesia.
Fokus pada: kinerja keuangan, aksi korporasi, sentimen analis, isu regulasi, berita industri.

Balas HANYA dalam format JSON ini (tanpa markdown, tanpa penjelasan):
{{
  "score": <integer -2 hingga +2>,
  "label": "<VERY POSITIVE|POSITIVE|NEUTRAL|NEGATIVE|VERY NEGATIVE>",
  "ringkasan": "<1-2 kalimat ringkasan berita utama>",
  "berita": ["<judul berita 1>", "<judul berita 2>", "<judul berita 3>"],
  "sumber": "<nama media/sumber>"
}}

score: +2=sangat positif, +1=positif, 0=netral, -1=negatif, -2=sangat negatif
Kalau tidak ada berita relevan, score=0, label=NEUTRAL, ringkasan="Tidak ada berita signifikan ditemukan."
"""
    try:
        resp = st.session_state.get("_api_cache_" + ticker)
        if resp:
            return resp

        import urllib.request
        body = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 500,
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "messages": [{"role": "user", "content": prompt}]
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())

        # Ambil text dari response
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        # Parse JSON — strip markdown fence kalau ada
        text = re.sub(r"```json|```", "", text).strip()
        result = json.loads(text)
        st.session_state["_api_cache_" + ticker] = result
        return result

    except Exception as e:
        return {
            "score": 0, "label": "NEUTRAL",
            "ringkasan": f"Gagal mengambil data sentimen: {e}",
            "berita": [], "sumber": "—"
        }


def render_sentiment(ticker: str):
    """Tampilkan sentiment card untuk satu ticker."""
    with st.spinner(f"Menganalisis sentimen berita {ticker}…"):
        sent = get_sentiment(ticker)

    score  = sent.get("score", 0)
    label  = sent.get("label", "NEUTRAL")
    colors = {
        "VERY POSITIVE": "#00e676", "POSITIVE": "#69f0ae",
        "NEUTRAL": "#ffc107",
        "NEGATIVE": "#ff9800", "VERY NEGATIVE": "#f44336",
    }
    icons = {
        "VERY POSITIVE": "🚀", "POSITIVE": "📈",
        "NEUTRAL": "➡️",
        "NEGATIVE": "📉", "VERY NEGATIVE": "🔻",
    }
    col = colors.get(label, "#ffc107")
    ico = icons.get(label, "➡️")

    bar_w = int((score + 2) / 4 * 100)   # -2..+2 → 0..100%

    st.markdown(f"""
    <div style="background:#161b22;border:1px solid {col};border-radius:10px;
                padding:1rem 1.3rem;margin-bottom:0.8rem;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:1.1rem;font-weight:700;color:{col};">{ico} {label}</span>
        <span style="font-size:0.78rem;color:#8b949e;">Score: {score:+d} / 2</span>
      </div>
      <div style="background:#0b0e17;border-radius:4px;height:6px;margin:8px 0;">
        <div style="width:{bar_w}%;height:6px;border-radius:4px;background:{col};"></div>
      </div>
      <div style="font-size:0.88rem;color:#c9d1d9;margin-top:6px;">{sent.get('ringkasan','')}</div>
    </div>
    """, unsafe_allow_html=True)

    berita = sent.get("berita", [])
    if berita:
        st.markdown("**📰 Berita terkait:**")
        for b in berita:
            st.markdown(f"- {b}")

    return score, label


# ══════════════════════════════════════════════════════════════
# SECTOR HEATMAP ENGINE
# ══════════════════════════════════════════════════════════════
def build_sector_heatmap(hasil: list[dict]) -> go.Figure:
    """
    Dari hasil screener, buat heatmap sektor:
    - Sumbu X: sinyal (STRONG BUY → STRONG SELL)
    - Tiap sektor: count saham per sinyal
    - Warna: dominasi sinyal
    """
    if not hasil:
        return None

    df_h = pd.DataFrame(hasil)
    if "Sektor" not in df_h.columns:
        return None

    signal_order = ["STRONG BUY", "BUY", "SPEC. BUY", "WAIT", "SELL", "STRONG SELL"]
    signal_colors = {
        "STRONG BUY": "#00e676", "BUY": "#4caf50", "SPEC. BUY": "#8bc34a",
        "WAIT": "#ffc107", "SELL": "#ff9800", "STRONG SELL": "#f44336",
    }

    sektors = sorted(df_h["Sektor"].unique())
    pivot   = df_h.groupby(["Sektor", "Sinyal"]).size().unstack(fill_value=0)
    for sig in signal_order:
        if sig not in pivot.columns:
            pivot[sig] = 0
    pivot = pivot[signal_order]

    # Hitung "bull score" per sektor: weighted sum
    weights = {"STRONG BUY": 2, "BUY": 1, "SPEC. BUY": 0.5,
               "WAIT": 0, "SELL": -1, "STRONG SELL": -2}
    pivot["_bull_score"] = sum(pivot[s] * w for s, w in weights.items())
    pivot["_total"]      = pivot[signal_order].sum(axis=1)
    pivot = pivot.sort_values("_bull_score", ascending=True)

    sektors_sorted = pivot.index.tolist()
    z_data  = []
    text_data = []
    for sektor in sektors_sorted:
        row = []; trow = []
        for sig in signal_order:
            cnt = int(pivot.loc[sektor, sig]) if sektor in pivot.index else 0
            row.append(cnt); trow.append(str(cnt) if cnt > 0 else "")
        z_data.append(row); text_data.append(trow)

    fig = go.Figure(go.Heatmap(
        z=z_data,
        x=signal_order,
        y=sektors_sorted,
        text=text_data,
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        colorscale=[
            [0.0, "#1f0707"], [0.25, "#2e1a0a"],
            [0.5, "#1a1600"],
            [0.75, "#081a10"], [1.0, "#071f14"],
        ],
        showscale=False,
        hovertemplate="Sektor: %{y}<br>Sinyal: %{x}<br>Jumlah: %{z}<extra></extra>",
    ))

    # Annotasi bull score di kanan
    for i, sektor in enumerate(sektors_sorted):
        bs = float(pivot.loc[sektor, "_bull_score"])
        tot = int(pivot.loc[sektor, "_total"])
        col = "#00e676" if bs > 0 else "#f44336" if bs < 0 else "#ffc107"
        fig.add_annotation(
            x=len(signal_order) - 0.3, y=i,
            text=f"  {bs:+.0f} ({tot})",
            showarrow=False, xanchor="left",
            font=dict(size=10, color=col),
        )

    CT = CHART_THEME
    fig.update_layout(
        paper_bgcolor=CT["bg"], plot_bgcolor=CT["bg2"],
        font=dict(color=CT["text"], size=11),
        height=max(300, len(sektors_sorted) * 42 + 80),
        margin=dict(l=120, r=100, t=50, b=60),
        title=dict(text="Sector Heatmap — distribusi sinyal per sektor", font=dict(size=13), x=0.01),
        xaxis=dict(side="top", tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
    )
    return fig


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📡 IFH Pro")
    st.markdown(f"<span style='color:#8b949e;font-size:0.78rem;'>{len(daftar_saham)} emiten loaded</span>",
                unsafe_allow_html=True)
    st.divider()

    mode = st.radio("Mode", [
        "🔍 Analisis + Plan",
        "🚀 Screener Massal",
        "🐦 Early Bird",
        "🌡️ Sector Heatmap",
    ], label_visibility="collapsed")
    st.divider()

    days = st.slider("Lookback Fibonacci (hari)", 30, 365, 120, 10)
    require_ha  = st.toggle("Filter HA Hijau", value=True)
    min_vol_rel = st.slider("Min. Volume Relatif ×ADV20", 0.0, 3.0, 0.5, 0.1)
    min_conf    = st.slider("Min. Confidence (%)", 0, 100, 50, 5)
    st.divider()
    st.markdown("**🏥 Liquidity Filter**")
    use_health    = st.toggle("Aktifkan filter likuiditas", value=True)
    min_price     = st.number_input("Min. harga (Rp)", value=50, step=10)
    min_adv_juta  = st.number_input("Min. ADV value (Rp juta/hari)", value=500, step=100)
    workers     = st.slider("Parallel workers (screener)", 2, 16, 8, 2,
                             help="Thread paralel untuk download. Lebih tinggi = lebih cepat.")
    st.divider()

    st.caption(f"{'🟢 scipy' if HAS_SCIPY else '🟡 fallback pivot'}")
    st.caption("Ichimoku · EMA · MACD · ADX · StochRSI")
    st.caption("ATR · VWAP · OBV · CQS · Heikin Ashi")
    st.caption("Fibonacci · Early Bird · Dist. Trap")
    st.caption("Sector Heatmap · Sentiment AI")


# ══════════════════════════════════════════════════════════════
# MODE 1: ANALISIS + TRADING PLAN
# ══════════════════════════════════════════════════════════════
if "Analisis" in mode:
    st.markdown("## 🔍 Analisis Spesifik")

    c1, c2 = st.columns([4, 1])
    with c1:
        ticker = st.selectbox("Saham", daftar_saham, label_visibility="collapsed")
    with c2:
        run = st.button("Analisis ▶", use_container_width=True, type="primary")

    if run:
        with st.spinner(f"Menganalisis {ticker}…"):
            h = analyse(ticker, days)

        if not h:
            st.error("Data tidak cukup atau ticker tidak valid.")
        else:
            # Health check
            if use_health:
                df_hc = download_one(ticker)
                hc = health_check(df_hc, ticker, min_price=min_price, min_adv_juta=min_adv_juta)
                if not hc.lolos:
                    st.warning(f"⚠️ **Health Check:** {hc.alasan}")
                    for f in hc.flags:
                        st.markdown(f"- {f}")
                    st.info("Analisis tetap ditampilkan — gunakan dengan ekstra hati-hati.")

            sig = build_plan(h)

            render_banner(sig, h["harga"])

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("#### 🧠 Analisis Konfluensi")
                for a in sig.alasan:
                    st.markdown(f"<div style='font-size:0.88rem;padding:3px 0;'>{a}</div>",
                                unsafe_allow_html=True)
            with col_r:
                st.markdown("#### 🎯 Action Plan")
                for a in sig.aksi:
                    st.markdown(f"<div style='font-size:0.88rem;padding:3px 0;'>{a}</div>",
                                unsafe_allow_html=True)

            st.divider()
            st.markdown("#### 📏 Price Ladder")
            render_price_ladder(h, sig)

            st.divider()
            tab0, tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Chart", "☁️ Ichimoku + EMA", "📐 Indikator Momentum",
                "📊 Volume + HA", "📰 Sentimen Berita"
            ])

            with tab0:
                render_charts(ticker, h)

            with tab1:
                d = {
                    "Indikator": ["Tenkan","Kijun","Senkou A","Senkou B","Awan Atas","Awan Bawah","EMA 20","EMA 50","Posisi"],
                    "Nilai": [f"{h['tenkan']:,.0f}", f"{h['kijun']:,.0f}", f"{h['sa']:,.0f}",
                              f"{h['sb']:,.0f}", f"{h['awan_hi']:,.0f}", f"{h['awan_lo']:,.0f}",
                              f"{h['ema20']:,.0f}", f"{h['ema50']:,.0f}",
                              "✅ Di atas awan" if h["di_atas"] else "⚠️ Dalam awan" if h["di_dalam"] else "❌ Di bawah awan"],
                }
                st.dataframe(pd.DataFrame(d), hide_index=True, use_container_width=True)

            with tab2:
                d2 = {
                    "Indikator": ["MACD Histogram","ADX","RSI (14)","StochRSI K","StochRSI D","ATR (14)","ATR %"],
                    "Nilai": [f"{h['macd_hist']:+.3f}", f"{h['adx']:.1f}", f"{h['rsi']:.1f}",
                              f"{h['srsi_k']:.1f}", f"{h['srsi_d']:.1f}", f"{h['atr']:.2f}", f"{h['atr_pct']:.2f}%"],
                    "Status": [
                        "🚀 Cross UP!" if h["macd_cross_up"] else ("✅ Positif" if h["macd_bull"] else "❌ Negatif"),
                        "💪 Kuat" if h["adx_strong"] else "➡️ Lemah",
                        "🔥 Overbought" if h["rsi"]>70 else ("💧 Oversold" if h["rsi"]<30 else "📊 Normal"),
                        "🔥 Overbought" if h["srsi_overbought"] else ("💧 Oversold" if h["srsi_oversold"] else "📊 Normal"),
                        "─","─","─",
                    ],
                }
                st.dataframe(pd.DataFrame(d2), hide_index=True, use_container_width=True)

            with tab3:
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Volume Rel.", f"{h['vol_rel']:.2f}×", delta="Tinggi" if h["vol_rel"]>1.5 else None)
                m2.metric("VWAP 20d", f"{h['vwap']:,.0f}")
                m3.metric("HA Status", f"{'🟢' if h['ha_green'] else '🔴'} {h['ha_seq']} candle")
                m4.metric("Price vs VWAP", "✅ Di atas" if h["price_above_vwap"] else "⚠️ Di bawah")

                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                cqs_label = "🏦 Akumulasi" if h["accum_confirm"] else ("🚨 Distribusi!" if h["dist_trap"] else "📊 Normal")
                c1.metric("CQS (3 candle)", f"{h['cqs']:.2f}", delta=cqs_label)
                c2.metric("OBV Trend", "✅ Naik" if h["obv_bull"] else "⚠️ Turun")
                c3.metric("Shooting Star", "⚠️ YA" if h["shooting_star"] else "✅ Tidak")
                c4.metric("Early Bird Score", f"{h['early_score']}/5")

            with tab4:
                st.markdown("#### 📰 Sentimen Berita Terkini")
                st.caption("Diambil real-time via AI web search — cache 1 jam")
                score_s, label_s = render_sentiment(ticker)
                # Pengaruh ke sinyal
                st.divider()
                if score_s >= 1 and "BUY" in sig.sinyal:
                    st.success("✅ Sentimen berita MENDUKUNG sinyal teknikal — konfluensi lebih kuat")
                elif score_s <= -1 and "BUY" in sig.sinyal:
                    st.warning("⚠️ Sentimen berita BERTENTANGAN dengan sinyal teknikal — pertimbangkan sizing lebih kecil")
                elif score_s <= -1:
                    st.error("🔴 Sentimen negatif memperkuat sinyal SELL/WAIT")
                else:
                    st.info("➡️ Sentimen netral — keputusan berdasarkan teknikal saja")


# ══════════════════════════════════════════════════════════════
# MODE 4: SECTOR HEATMAP
# ══════════════════════════════════════════════════════════════
elif "Heatmap" in mode:
    st.markdown("## 🌡️ Sector Heatmap")
    st.markdown(
        "<span style='color:#8b949e;font-size:0.85rem;'>"
        "Lihat sektor mana yang paling banyak sinyal BUY — "
        "deteksi rotasi sektor & market breadth secara visual"
        "</span>", unsafe_allow_html=True
    )

    # Gunakan hasil screener kalau sudah ada, atau jalankan scan cepat
    if "heatmap_hasil" not in st.session_state: st.session_state.heatmap_hasil = []
    if "heatmap_raw"   not in st.session_state: st.session_state.heatmap_raw   = {}

    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        run_hm = st.button("🌡️ Scan Semua Sektor", type="primary", use_container_width=True)
    with col_b2:
        if st.button("🔄 Reset", key="hm_reset", use_container_width=True):
            st.session_state.heatmap_hasil = []; st.session_state.heatmap_raw = {}; st.rerun()

    if run_hm:
        st.session_state.heatmap_hasil = []; st.session_state.heatmap_raw = {}
        pbar = st.progress(0, text="Download paralel…")
        status = st.empty(); holder = st.empty()
        total = len(daftar_saham); t0 = time.time()

        # Download paralel
        data_map_hm: dict[str, pd.DataFrame] = {}
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            fut = {ex.submit(download_one, t): t for t in daftar_saham}
            for f in concurrent.futures.as_completed(fut):
                t = fut[f]
                try:    data_map_hm[t] = f.result()
                except: data_map_hm[t] = pd.DataFrame()
                done += 1
                pbar.progress(done/total/2, text=f"⚡ Download: {done}/{total}")

        # Analisis semua (tanpa filter ketat — heatmap butuh semua sinyal)
        for i, ticker in enumerate(daftar_saham):
            df_t = data_map_hm.get(ticker, pd.DataFrame())
            if df_t is None or df_t.empty: continue
            try:
                if use_health:
                    hc = health_check(df_t, ticker, min_price=min_price, min_adv_juta=min_adv_juta)
                    if not hc.lolos: continue
                h = analyse(ticker, days, df=df_t)
                if not h: continue
                sig = build_plan(h)
                sektor = get_sektor(ticker)
                st.session_state.heatmap_hasil.append({
                    "Ticker": ticker, "Sektor": sektor,
                    "Sinyal": sig.sinyal, "Conf.": sig.confidence,
                    "Harga": int(round(h["harga"])),
                })
                st.session_state.heatmap_raw[ticker] = (h, sig)
            except: pass
            pbar.progress(0.5 + (i+1)/total/2, text=f"🧠 Analisis: {i+1}/{total}")

        elapsed = time.time() - t0
        pbar.progress(1.0, text=f"✅ Selesai {elapsed:.1f}s")
        status.empty(); st.rerun()

    if st.session_state.heatmap_hasil:
        hasil_hm = st.session_state.heatmap_hasil
        df_hm    = pd.DataFrame(hasil_hm)
        n_total  = len(df_hm)
        n_bull   = len(df_hm[df_hm["Sinyal"].isin(["STRONG BUY","BUY","SPEC. BUY"])])
        n_bear   = len(df_hm[df_hm["Sinyal"].isin(["SELL","STRONG SELL"])])

        # ── Market breadth summary ──
        breadth_pct = n_bull / n_total * 100 if n_total else 0
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Saham", n_total)
        m2.metric("🟢 Bullish", n_bull, delta=f"{breadth_pct:.0f}% market")
        m3.metric("🔴 Bearish/Sell", n_bear)
        m4.metric("Market Breadth", f"{breadth_pct:.0f}%",
                  delta="Bullish" if breadth_pct > 50 else "Bearish")

        st.divider()

        # ── Heatmap chart ──
        fig_hm = build_sector_heatmap(hasil_hm)
        if fig_hm:
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("Tambahkan kolom 'sector' di emiten.csv untuk heatmap per sektor. Menampilkan ringkasan saja.")

        # ── Top sektor bullish ──
        st.divider()
        if "Sektor" in df_hm.columns:
            bull_df = df_hm[df_hm["Sinyal"].isin(["STRONG BUY","BUY","SPEC. BUY"])]
            top_sektors = bull_df.groupby("Sektor").size().sort_values(ascending=False).head(5)

            col_top, col_list = st.columns([1, 2])
            with col_top:
                st.markdown("#### 🔥 Top Sektor Bullish")
                for sektor, count in top_sektors.items():
                    pct = count / len(bull_df) * 100 if len(bull_df) else 0
                    st.markdown(f"**{sektor}** — {count} saham ({pct:.0f}%)")

            with col_list:
                st.markdown("#### 📋 Saham Bullish per Sektor")
                sel_sektor = st.selectbox(
                    "Pilih sektor:", sorted(df_hm["Sektor"].unique()),
                    key="hm_sektor_sel"
                )
                df_sel = df_hm[
                    (df_hm["Sektor"] == sel_sektor) &
                    (df_hm["Sinyal"].isin(["STRONG BUY","BUY","SPEC. BUY"]))
                ].sort_values("Conf.", ascending=False)
                if not df_sel.empty:
                    st.dataframe(df_sel[["Ticker","Sinyal","Conf.","Harga"]],
                                 hide_index=True, use_container_width=True)
                    # Tombol buka full report
                    st.markdown("**Klik untuk full report:**")
                    if "hm_sel_ticker" not in st.session_state:
                        st.session_state.hm_sel_ticker = None
                    cols_hm = st.columns(min(6, len(df_sel)))
                    for j, (_, row) in enumerate(df_sel.iterrows()):
                        tk = row["Ticker"]
                        if j < len(cols_hm):
                            is_active = st.session_state.hm_sel_ticker == tk
                            if cols_hm[j].button(
                                f"{'▶ ' if is_active else ''}{tk}",
                                key=f"hm_btn_{tk}", use_container_width=True,
                                type="primary" if is_active else "secondary"
                            ):
                                st.session_state.hm_sel_ticker = None if is_active else tk
                                st.rerun()
                else:
                    st.info(f"Tidak ada saham bullish di sektor {sel_sektor}")

        # ── Inline full report dari heatmap ──
        hm_sel = st.session_state.get("hm_sel_ticker")
        if hm_sel and hm_sel in st.session_state.heatmap_raw:
            h, sig = st.session_state.heatmap_raw[hm_sel]
            st.divider()
            st.markdown(f"## 📋 Full Report: `{hm_sel}` [{get_sektor(hm_sel)}]")
            render_banner(sig, h["harga"])
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("#### 🧠 Konfluensi")
                for a in sig.alasan:
                    st.markdown(f"<div style='font-size:0.85rem;padding:2px 0;'>{a}</div>",
                                unsafe_allow_html=True)
            with col_r:
                st.markdown("#### 🎯 Action Plan")
                for a in sig.aksi:
                    st.markdown(f"<div style='font-size:0.85rem;padding:2px 0;'>{a}</div>",
                                unsafe_allow_html=True)
            st.divider()
            render_price_ladder(h, sig)
            st.divider()
            render_charts(hm_sel, h)
            if st.button("✖ Tutup", key="hm_close"):
                st.session_state.hm_sel_ticker = None; st.rerun()

    elif not run_hm:
        st.info("Klik **Scan Semua Sektor** untuk mulai. Semua saham akan dianalisis dan dikelompokkan per sektor.")
        if not sektor_map:
            st.warning(
                "💡 **Tips:** Tambahkan kolom `sector` di **emiten.csv** untuk heatmap yang lebih akurat.\n\n"
                "Contoh format:\n```\nticker,sector\nBBCA.JK,Perbankan\nTLKM.JK,Telekomunikasi\n```\n\n"
                "Tanpa kolom sektor, app akan menebak sektor dari prefix ticker."
            )


# ══════════════════════════════════════════════════════════════
# MODE 2: SCREENER MASSAL (PARALLEL) + INLINE FULL REPORT
# ══════════════════════════════════════════════════════════════
elif "Screener" in mode:
    st.markdown("## 🚀 Screener Massal")
    st.markdown(
        f"<span style='color:#8b949e;font-size:0.85rem;'>"
        f"Filter: Ichimoku Uptrend · {'HA Hijau · ' if require_ha else ''}"
        f"Vol ≥{min_vol_rel}×ADV20 · Confidence ≥{min_conf}%  "
        f"· {workers} parallel workers</span>",
        unsafe_allow_html=True
    )

    # ── session state: simpan hasil screener & ticker yang dipilih ──
    if "screener_hasil"   not in st.session_state: st.session_state.screener_hasil   = []
    if "screener_raw"     not in st.session_state: st.session_state.screener_raw     = {}  # ticker → (h, sig)
    if "selected_ticker"  not in st.session_state: st.session_state.selected_ticker  = None

    col_btn, col_rst = st.columns([2, 1])
    with col_btn:
        run_screen = st.button("▶ Mulai Screening", type="primary", use_container_width=True)
    with col_rst:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.screener_hasil  = []
            st.session_state.screener_raw    = {}
            st.session_state.selected_ticker = None
            st.rerun()

    if run_screen:
        st.session_state.screener_hasil  = []
        st.session_state.screener_raw    = {}
        st.session_state.selected_ticker = None

        errors: list[str] = []
        pbar   = st.progress(0, text="Memulai download paralel…")
        status = st.empty()
        holder = st.empty()
        total  = len(daftar_saham)
        t0     = time.time()

        # ── Fase 1: Parallel download ──
        status.markdown("⚡ **Fase 1/2** — Download paralel semua ticker…")
        data_map: dict[str, pd.DataFrame] = {}
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            fut = {ex.submit(download_one, t): t for t in daftar_saham}
            for f in concurrent.futures.as_completed(fut):
                t = fut[f]
                try:    data_map[t] = f.result()
                except: data_map[t] = pd.DataFrame()
                done += 1
                pbar.progress(done / total / 2, text=f"⚡ Download: {done}/{total}")

        dl_time = time.time() - t0
        status.markdown(f"✅ **Fase 1 selesai** — {total} ticker diunduh dalam **{dl_time:.1f}s**")

        # ── Fase 2: Analisis & scoring ──
        status.markdown("🧠 **Fase 2/2** — Analisis & scoring…")
        for i, ticker in enumerate(daftar_saham):
            df_t = data_map.get(ticker, pd.DataFrame())
            if df_t is None or df_t.empty: continue
            try:
                h = analyse(ticker, days, df=df_t)
                if not h: continue
                # Health check filter
                if use_health:
                    hc = health_check(df_t, ticker, min_price=min_price, min_adv_juta=min_adv_juta)
                    if not hc.lolos: continue
                if require_ha and not h["ha_green"]: continue
                if not (h["di_atas"] and h["tk_kj"]): continue
                if h["vol_rel"] < min_vol_rel: continue
                sig = build_plan(h)
                if sig.confidence < min_conf: continue

                # Simpan raw data untuk full report
                st.session_state.screener_raw[ticker] = (h, sig)

                st.session_state.screener_hasil.append({
                    "Ticker":   ticker,
                    "Sinyal":   sig.sinyal,
                    "Conf.":    sig.confidence,
                    "Harga":    int(round(h["harga"])),
                    "Entry":    f"{int(round(h['lvl']['entry_bawah']))}–{int(round(h['lvl']['entry_atas']))}",
                    "Target 1": int(round(h["lvl"]["target_1"])),
                    "Cutloss":  int(round(h["lvl"]["cutloss"])),
                    "R/R":      f"1:{sig.rr:.1f}" if sig.rr else "─",
                    "MACD":     "🚀" if h["macd_cross_up"] else ("✅" if h["macd_bull"] else "❌"),
                    "ADX":      f"{h['adx']:.0f}",
                    "RSI":      f"{h['rsi']:.0f}",
                    "HA":       f"🟢{h['ha_seq']}c" if h["ha_green"] else f"🔴{h['ha_seq']}c",
                    "Vol":      f"{h['vol_rel']:.1f}×",
                })

                df_tmp = (pd.DataFrame(st.session_state.screener_hasil)
                          .sort_values("Conf.", ascending=False).reset_index(drop=True))
                holder.dataframe(df_tmp, hide_index=True, use_container_width=True)

            except Exception as e:
                errors.append(f"{ticker}: {e}")

            pbar.progress(0.5 + (i+1)/total/2, text=f"🧠 Analisis: {i+1}/{total}")

        elapsed = time.time() - t0
        pbar.progress(1.0, text=f"✅ Selesai dalam {elapsed:.1f}s")
        status.empty()
        holder.empty()

        if errors:
            with st.expander(f"⚠️ {len(errors)} ticker error"):
                for e in errors: st.text(e)

        st.rerun()  # re-render bersih dengan data di session_state

    # ══════════════════════════════════════════════════════════
    # RENDER TABEL HASIL + TOMBOL PER BARIS
    # ══════════════════════════════════════════════════════════
    if st.session_state.screener_hasil:
        df_f = (pd.DataFrame(st.session_state.screener_hasil)
                .sort_values("Conf.", ascending=False).reset_index(drop=True))

        n = len(df_f)
        elapsed_info = ""
        st.success(f"🎉 **{n} saham** lolos filter{elapsed_info} — klik tombol di bawah untuk full report")

        # ── Tabel ringkas ──
        st.dataframe(df_f, hide_index=True, use_container_width=True)

        # ── Download CSV ──
        csv = df_f.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "hasil_screener.csv", "text/csv")

        st.divider()

        # ── Tombol per emiten (grid 6 kolom) ──
        st.markdown("#### 🔍 Klik emiten untuk full report:")

        tickers_sorted = df_f["Ticker"].tolist()
        COLS = 6
        rows_btn = [tickers_sorted[i:i+COLS] for i in range(0, len(tickers_sorted), COLS)]

        for row in rows_btn:
            cols = st.columns(COLS)
            for j, tk in enumerate(row):
                raw = st.session_state.screener_raw.get(tk)
                if raw:
                    _, sig_row = raw
                    col_hex = COLOR_MAP.get(sig_row.sinyal, "#ffc107")
                    # highlight tombol yang sedang aktif
                    is_active = st.session_state.selected_ticker == tk
                    btn_label = f"{'▶ ' if is_active else ''}{tk}"
                    if cols[j].button(btn_label, key=f"btn_{tk}",
                                      use_container_width=True,
                                      type="primary" if is_active else "secondary"):
                        # toggle: klik lagi → tutup
                        if st.session_state.selected_ticker == tk:
                            st.session_state.selected_ticker = None
                        else:
                            st.session_state.selected_ticker = tk
                        st.rerun()

        # ══════════════════════════════════════════════════════
        # INLINE FULL REPORT
        # ══════════════════════════════════════════════════════
        sel = st.session_state.selected_ticker
        if sel and sel in st.session_state.screener_raw:
            h, sig = st.session_state.screener_raw[sel]

            st.divider()
            st.markdown(f"## 📋 Full Report: `{sel}`")

            # ── Banner ──
            render_banner(sig, h["harga"])

            # ── Konfluensi + Action Plan ──
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("#### 🧠 Analisis Konfluensi")
                for a in sig.alasan:
                    st.markdown(f"<div style='font-size:0.88rem;padding:3px 0;'>{a}</div>",
                                unsafe_allow_html=True)
            with col_r:
                st.markdown("#### 🎯 Action Plan")
                for a in sig.aksi:
                    st.markdown(f"<div style='font-size:0.88rem;padding:3px 0;'>{a}</div>",
                                unsafe_allow_html=True)

            st.divider()
            st.markdown("#### 📏 Price Ladder")
            render_price_ladder(h, sig)

            st.divider()
            tab0, tab1, tab2, tab3 = st.tabs(["📈 Chart", "☁️ Ichimoku + EMA", "📐 Momentum", "📊 Volume + HA"])

            with tab0:
                render_charts(sel, h)

            with tab1:
                d = {
                    "Indikator": ["Tenkan","Kijun","Senkou A","Senkou B","Awan Atas","Awan Bawah","EMA 20","EMA 50","Posisi"],
                    "Nilai": [f"{h['tenkan']:,.0f}", f"{h['kijun']:,.0f}", f"{h['sa']:,.0f}",
                              f"{h['sb']:,.0f}", f"{h['awan_hi']:,.0f}", f"{h['awan_lo']:,.0f}",
                              f"{h['ema20']:,.0f}", f"{h['ema50']:,.0f}",
                              "✅ Di atas awan" if h["di_atas"] else "⚠️ Dalam awan" if h["di_dalam"] else "❌ Di bawah awan"],
                }
                st.dataframe(pd.DataFrame(d), hide_index=True, use_container_width=True)

            with tab2:
                d2 = {
                    "Indikator": ["MACD Histogram","ADX","RSI (14)","StochRSI K","StochRSI D","ATR (14)","ATR %"],
                    "Nilai": [f"{h['macd_hist']:+.3f}", f"{h['adx']:.1f}", f"{h['rsi']:.1f}",
                              f"{h['srsi_k']:.1f}", f"{h['srsi_d']:.1f}", f"{h['atr']:.2f}", f"{h['atr_pct']:.2f}%"],
                    "Status": [
                        "🚀 Cross UP!" if h["macd_cross_up"] else ("✅ Positif" if h["macd_bull"] else "❌ Negatif"),
                        "💪 Kuat" if h["adx_strong"] else "➡️ Lemah",
                        "🔥 Overbought" if h["rsi"]>70 else ("💧 Oversold" if h["rsi"]<30 else "📊 Normal"),
                        "🔥 OB" if h["srsi_overbought"] else ("💧 OS" if h["srsi_oversold"] else "📊 Normal"),
                        "─","─","─",
                    ],
                }
                st.dataframe(pd.DataFrame(d2), hide_index=True, use_container_width=True)

            with tab3:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Volume Rel.", f"{h['vol_rel']:.2f}×", delta="Tinggi" if h["vol_rel"]>1.5 else None)
                m2.metric("VWAP 20d", f"{h['vwap']:,.0f}")
                m3.metric("HA Status", f"{'🟢' if h['ha_green'] else '🔴'} {h['ha_seq']} candle")
                m4.metric("Price vs VWAP", "✅ Di atas" if h["price_above_vwap"] else "⚠️ Di bawah")

                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                cqs_label = "🏦 Akumulasi" if h["accum_confirm"] else ("🚨 Distribusi!" if h["dist_trap"] else "📊 Normal")
                c1.metric("CQS (3 candle)", f"{h['cqs']:.2f}", delta=cqs_label)
                c2.metric("OBV Trend", "✅ Naik" if h["obv_bull"] else "⚠️ Turun")
                c3.metric("Shooting Star", "⚠️ YA" if h["shooting_star"] else "✅ Tidak")
                c4.metric("Early Bird Score", f"{h['early_score']}/5")

            # ── Tutup report ──
            if st.button("✖ Tutup Report", key="close_report"):
                st.session_state.selected_ticker = None
                st.rerun()

# ══════════════════════════════════════════════════════════════
# MODE 3: EARLY BIRD — deteksi dini sebelum breakout
# ══════════════════════════════════════════════════════════════
elif "Early Bird" in mode:
    st.markdown("## 🐦 Early Bird Screener")
    st.markdown(
        "<span style='color:#8b949e;font-size:0.85rem;'>"
        "Deteksi saham yang **belum** breakout tapi sinyal Ichimoku mulai terbentuk "
        "— masuk sebelum semua orang sadar"
        "</span>", unsafe_allow_html=True
    )

    # Penjelasan sinyal
    with st.expander("📖 Apa yang dideteksi?", expanded=False):
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("""
**5 sinyal early bird (tiap = 1 poin):**
- 🌀 **Kumo twist baru** — Senkou A baru cross Senkou B, awan ganti warna
- 📉 **Kumo menyempit** — ketebalan awan turun >30% dalam 5 bar
- 🔜 **Harga dekat awan** — dalam 2% dari bawah awan
- ⚡ **TK cross baru** — Tenkan baru motong Kijun dalam 5 bar
- 👁️ **Chikou hampir bebas** — dalam 3% dari price bar 26 lalu
""")
        with col_e2:
            st.markdown("""
**Filter tambahan:**
- Harga belum di atas awan (masih pre-breakout)
- Tidak sedang distribution trap (CQS > 0.35)
- Volume tidak sedang colaps (<0.3×ADV)
- Minimum early score: 2 dari 5 sinyal

**Risk:** Ini sinyal *awal* — belum konfirmasi penuh.
Sizing lebih kecil, cutloss lebih ketat.
""")

    # Session state early bird
    if "eb_hasil"  not in st.session_state: st.session_state.eb_hasil  = []
    if "eb_raw"    not in st.session_state: st.session_state.eb_raw    = {}
    if "eb_sel"    not in st.session_state: st.session_state.eb_sel    = None

    min_eb_score = st.slider("Min. Early Bird Score (dari 5)", 1, 5, 2)
    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        run_eb = st.button("🐦 Mulai Early Bird Scan", type="primary", use_container_width=True)
    with col_b2:
        if st.button("🔄 Reset", key="eb_reset", use_container_width=True):
            st.session_state.eb_hasil = []; st.session_state.eb_raw = {}
            st.session_state.eb_sel = None; st.rerun()

    if run_eb:
        st.session_state.eb_hasil = []; st.session_state.eb_raw = {}; st.session_state.eb_sel = None
        errors_eb: list[str] = []

        pbar   = st.progress(0, text="Download paralel…")
        status = st.empty(); holder = st.empty()
        total  = len(daftar_saham); t0 = time.time()

        # Fase 1: download paralel
        status.markdown("⚡ Download paralel…")
        data_map_eb: dict[str, pd.DataFrame] = {}
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            fut = {ex.submit(download_one, t): t for t in daftar_saham}
            for f in concurrent.futures.as_completed(fut):
                t = fut[f]
                try:    data_map_eb[t] = f.result()
                except: data_map_eb[t] = pd.DataFrame()
                done += 1
                pbar.progress(done / total / 2, text=f"⚡ Download: {done}/{total}")

        # Fase 2: scan early bird
        status.markdown("🐦 Scanning early signals…")
        for i, ticker in enumerate(daftar_saham):
            df_t = data_map_eb.get(ticker, pd.DataFrame())
            if df_t is None or df_t.empty: continue
            try:
                h = analyse(ticker, days, df=df_t)
                if not h: continue
                # Health check
                if use_health:
                    hc = health_check(df_t, ticker, min_price=min_price, min_adv_juta=min_adv_juta)
                    if not hc.lolos: continue

                # Filter: belum breakout + cukup early signals
                if h["di_atas"]: continue              # sudah di atas awan → bukan early bird
                if h["early_score"] < min_eb_score: continue
                if h["dist_trap"]: continue            # distribusi → skip
                if h["vol_rel"] < 0.3: continue        # volume mati → skip

                sig = build_plan(h)
                st.session_state.eb_raw[ticker] = (h, sig)

                # Early bird label per sinyal
                eb_signals = []
                if h["kumo_twist_recent"]: eb_signals.append("🌀Twist")
                if h["kumo_narrowing"]:    eb_signals.append("📉Sempit")
                if h["price_near_cloud"]:  eb_signals.append("🔜Dekat")
                if h["tk_cross_new"]:      eb_signals.append("⚡TKcross")
                if h["chikou_near_free"]:  eb_signals.append("👁️Chikou")

                st.session_state.eb_hasil.append({
                    "Ticker":      ticker,
                    "EB Score":    f"{h['early_score']}/5",
                    "Sinyal":      " ".join(eb_signals),
                    "Harga":       int(round(h["harga"])),
                    "Jarak Awan":  f"{((h['awan_lo'] - h['harga']) / h['harga'] * 100):+.1f}%",
                    "Entry Plan":  f"{int(round(h['lvl']['entry_bawah']))}–{int(round(h['lvl']['entry_atas']))}",
                    "CQS":         f"{h['cqs']:.2f}",
                    "Vol":         f"{h['vol_rel']:.1f}×",
                    "MACD":        "🚀" if h["macd_cross_up"] else ("✅" if h["macd_bull"] else "❌"),
                    "RSI":         f"{h['rsi']:.0f}",
                })

                df_tmp = pd.DataFrame(st.session_state.eb_hasil).sort_values("EB Score", ascending=False)
                holder.dataframe(df_tmp, hide_index=True, use_container_width=True)

            except Exception as e:
                errors_eb.append(f"{ticker}: {e}")

            pbar.progress(0.5 + (i+1)/total/2, text=f"🐦 Scan: {i+1}/{total}")

        elapsed = time.time() - t0
        pbar.progress(1.0, text=f"✅ Selesai {elapsed:.1f}s")
        status.empty(); holder.empty()

        if errors_eb:
            with st.expander(f"⚠️ {len(errors_eb)} error"): 
                for e in errors_eb: st.text(e)
        st.rerun()

    # ── Render hasil early bird ──
    if st.session_state.eb_hasil:
        df_eb = pd.DataFrame(st.session_state.eb_hasil).sort_values("EB Score", ascending=False).reset_index(drop=True)
        st.success(f"🐦 **{len(df_eb)} calon early bird** ditemukan — belum breakout, sinyal mulai terbentuk")

        # Warning box
        st.warning(
            "⚠️ **Early bird = sinyal awal, bukan konfirmasi penuh.** "
            "Gunakan sizing 25–50% dari normal. Cutloss lebih ketat. "
            "Pantau tiap hari sampai harga konfirmasi tembus awan."
        )

        st.dataframe(df_eb, hide_index=True, use_container_width=True)
        csv_eb = df_eb.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV Early Bird", csv_eb, "early_bird.csv", "text/csv")

        st.divider()
        st.markdown("#### 🔍 Klik untuk full report:")

        tickers_eb = df_eb["Ticker"].tolist()
        COLS = 6
        for row in [tickers_eb[i:i+COLS] for i in range(0, len(tickers_eb), COLS)]:
            cols = st.columns(COLS)
            for j, tk in enumerate(row):
                if tk in st.session_state.eb_raw:
                    is_active = st.session_state.eb_sel == tk
                    if cols[j].button(
                        f"{'▶ ' if is_active else ''}{tk}",
                        key=f"eb_btn_{tk}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary"
                    ):
                        st.session_state.eb_sel = None if is_active else tk
                        st.rerun()

        # Inline full report early bird
        eb_sel = st.session_state.eb_sel
        if eb_sel and eb_sel in st.session_state.eb_raw:
            h, sig = st.session_state.eb_raw[eb_sel]
            st.divider()
            st.markdown(f"## 🐦 Early Bird Report: `{eb_sel}`")

            # Early bird status bar
            eb_cols = st.columns(5)
            eb_items = [
                ("🌀 Kumo Twist", h["kumo_twist_recent"]),
                ("📉 Kumo Sempit", h["kumo_narrowing"]),
                ("🔜 Dekat Awan", h["price_near_cloud"]),
                ("⚡ TK Cross", h["tk_cross_new"]),
                ("👁️ Chikou", h["chikou_near_free"]),
            ]
            for col, (label, val) in zip(eb_cols, eb_items):
                col.metric(label, "✅ YA" if val else "❌ Belum")

            st.markdown(f"""
            <div style="background:#0d1a2a;border:1px solid #1565c0;border-radius:10px;
                        padding:1rem 1.4rem;margin:1rem 0;font-size:0.88rem;">
              <strong style="color:#42a5f5;">📋 Trading Plan Early Bird</strong><br><br>
              🎯 <strong>Entry ideal:</strong> {h['lvl']['entry_bawah']:,.0f} – {h['lvl']['entry_atas']:,.0f}
                 (atau saat harga tembus awan {h['awan_lo']:,.0f})<br>
              🛑 <strong>Cutloss:</strong> {h['lvl']['cutloss']:,.0f}
                 (lebih ketat: bisa pakai {h['lvl']['entry_bawah'] * 0.97:,.0f})<br>
              🎯 <strong>Target 1:</strong> {h['lvl']['target_1']:,.0f}
                 &nbsp;|&nbsp; <strong>Target 2:</strong> {h['lvl']['target_2']:,.0f}<br>
              💰 <strong>Sizing:</strong> 25–50% dari normal (sinyal belum konfirmasi penuh)<br>
              👁️ <strong>Trigger masuk:</strong> tunggu harga close di atas awan Ichimoku
                 ({h['awan_lo']:,.0f}) dengan volume >1.5×ADV dan CQS >0.6
            </div>
            """, unsafe_allow_html=True)

            render_banner(sig, h["harga"])

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("#### 🧠 Konfluensi")
                for a in sig.alasan:
                    st.markdown(f"<div style='font-size:0.85rem;padding:2px 0;'>{a}</div>", unsafe_allow_html=True)
            with col_r:
                st.markdown("#### 🎯 Action Plan")
                for a in sig.aksi:
                    st.markdown(f"<div style='font-size:0.85rem;padding:2px 0;'>{a}</div>", unsafe_allow_html=True)

            st.divider()
            st.markdown("#### 📈 Chart")
            render_charts(eb_sel, h)

            st.divider()
            st.markdown("#### 📏 Price Ladder")
            render_price_ladder(h, sig)

            if st.button("✖ Tutup", key="eb_close"):
                st.session_state.eb_sel = None; st.rerun()
