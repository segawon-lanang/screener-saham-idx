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
.block-container { padding: 2.5rem 2rem 1rem !important; max-width: 1400px; }

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
    O, H, L, C = s(df["Open"]), s(df["High"]), s(df["Low"]), s(df["Close"])
    hac = (O + H + L + C) / 4
    # Vectorized HA Open via cumulative approach:
    # hao[i] = (hao[i-1] + hac[i-1]) / 2  →  EWM with alpha=0.5, adjust=False
    hao_init = float((O.iloc[0] + C.iloc[0]) / 2)
    hac_arr  = hac.values
    hao      = np.empty(len(df))
    hao[0]   = hao_init
    # Numba-free tight loop is still O(n) but avoids pandas overhead per element
    for i in range(1, len(df)):
        hao[i] = (hao[i - 1] + hac_arr[i - 1]) * 0.5
    r = df.copy()
    r["HA_C"] = hac.values
    r["HA_O"] = hao
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


def finite_float(value, default: float = 0.0) -> float:
    """Return angka float yang aman untuk scoring dan tampilan."""
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def streak_count(flags: pd.Series, expected: bool = True) -> int:
    """Hitung streak terbaru untuk candle hijau/merah tanpa tertukar."""
    arr = flags.astype(bool).to_numpy()
    count = 0
    for flag in arr[::-1]:
        if flag == expected:
            count += 1
        else:
            break
    return count


def price_too_far_above_entry(h: dict, tolerance_pct: float = 2.0) -> bool:
    """True jika harga sudah terlalu jauh di atas rentang beli."""
    entry_atas = h["lvl"]["entry_atas"]
    return h["harga"] > entry_atas * (1 + tolerance_pct / 100)


# ══════════════════════════════════════════════════════════════
# MARKET REGIME  (IHSG context — filter entry saat pasar bearish)
# ══════════════════════════════════════════════════════════════
@dataclass
class MarketRegime:
    regime: str          # "BULL" / "NEUTRAL" / "BEAR"
    ihsg_trend: str      # "UPTREND" / "SIDEWAYS" / "DOWNTREND"
    ihsg_change_5d: float
    ihsg_change_20d: float
    breadth_pct: float   # % saham di atas EMA20 (dari sample)
    warning: str         # pesan ke user
    multiplier: float    # 1.0=normal, 0.7=hati-hati, 0.4=defensif
    # Data harga IHSG hari ini
    ihsg_last:  float = 0.0
    ihsg_open:  float = 0.0
    ihsg_high:  float = 0.0
    ihsg_low:   float = 0.0
    ihsg_chg1d: float = 0.0   # % change vs previous close
    ihsg_prev:  float = 0.0   # previous close

@st.cache_data(ttl=1800, show_spinner=False)
def get_market_regime() -> MarketRegime:
    """
    Download IHSG (^JKSE) dan hitung regime market saat ini.
    Regime menentukan multiplier sizing dan filter ketat/longgar.
    """
    try:
        df_ihsg = yf.download("^JKSE", period="120d", progress=False, auto_adjust=True)
        if df_ihsg.empty or len(df_ihsg) < 30:
            return MarketRegime("NEUTRAL","SIDEWAYS",0,0,50,"Data IHSG tidak tersedia",1.0)

        if isinstance(df_ihsg.columns, pd.MultiIndex):
            df_ihsg.columns = df_ihsg.columns.get_level_values(0)

        c = s(df_ihsg["Close"])
        o = s(df_ihsg["Open"])
        h_s = s(df_ihsg["High"])
        l_s = s(df_ihsg["Low"])
        ema20_ihsg = float(EMAIndicator(c, window=20).ema_indicator().iloc[-1])
        ema50_ihsg = float(EMAIndicator(c, window=50).ema_indicator().iloc[-1])
        last       = float(c.iloc[-1])
        prev_close = float(c.iloc[-2]) if len(c) >= 2 else last
        chg1d      = (last / prev_close - 1) * 100 if prev_close > 0 else 0.0
        chg5d      = (last / float(c.iloc[-6]) - 1) * 100 if len(c) >= 6 else 0
        chg20d     = (last / float(c.iloc[-21]) - 1) * 100 if len(c) >= 21 else 0
        open_now   = float(o.iloc[-1])
        high_now   = float(h_s.iloc[-1])
        low_now    = float(l_s.iloc[-1])

        # Trend IHSG
        if last > ema20_ihsg and ema20_ihsg > ema50_ihsg and chg20d > 2:
            ihsg_trend = "UPTREND"
        elif last < ema20_ihsg and ema20_ihsg < ema50_ihsg and chg20d < -2:
            ihsg_trend = "DOWNTREND"
        else:
            ihsg_trend = "SIDEWAYS"

        # Regime final
        if ihsg_trend == "UPTREND" and chg5d > -1:
            regime, mult, warn = "BULL",    1.0, "✅ Market BULL — sinyal lebih reliable"
        elif ihsg_trend == "DOWNTREND" or chg5d < -3:
            regime, mult, warn = "BEAR",    0.4, "🚨 Market BEAR — kurangi sizing 60%, prioritas cash"
        else:
            regime, mult, warn = "NEUTRAL", 0.7, "⚠️ Market SIDEWAYS — sizing 70%, selektif"

        return MarketRegime(
            regime=regime, ihsg_trend=ihsg_trend,
            ihsg_change_5d=round(chg5d,2), ihsg_change_20d=round(chg20d,2),
            breadth_pct=50.0,   # diupdate saat screener jalan
            warning=warn, multiplier=mult,
            ihsg_last=round(last, 2),
            ihsg_open=round(open_now, 2),
            ihsg_high=round(high_now, 2),
            ihsg_low=round(low_now, 2),
            ihsg_chg1d=round(chg1d, 2),
            ihsg_prev=round(prev_close, 2),
        )
    except Exception as e:
        return MarketRegime("NEUTRAL","SIDEWAYS",0,0,50,f"Error IHSG: {e}",1.0)


@st.cache_data(ttl=900, show_spinner=False)
def _get_ihsg_cached() -> pd.DataFrame:
    """Download IHSG sekali, cache 15 menit — dipakai RS calculation."""
    try:
        df = yf.download("^JKSE", period="120d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna(subset=["Close","Volume"]) if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def render_market_regime(mr: MarketRegime):
    """Banner IHSG regime — tampil di atas semua mode."""
    col       = {"BULL": "#00e676", "NEUTRAL": "#ffc107", "BEAR": "#f44336"}.get(mr.regime, "#ffc107")
    chg_col   = "#00e676" if mr.ihsg_chg1d >= 0 else "#f44336"
    chg5_col  = "#00e676" if mr.ihsg_change_5d >= 0 else "#f44336"
    chg20_col = "#00e676" if mr.ihsg_change_20d >= 0 else "#f44336"
    arrow     = "▲" if mr.ihsg_chg1d >= 0 else "▼"
    rng       = mr.ihsg_high - mr.ihsg_low
    pos       = max(2, min(98, ((mr.ihsg_last - mr.ihsg_low) / rng * 100) if rng > 0 else 50))

    html = (
        f'<div style="background:#161b22;border:1px solid {col};border-radius:10px;'
        f'padding:0.75rem 1.4rem;margin-top:0.5rem;margin-bottom:1rem;">'

        f'<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.5rem 1.5rem;">'

        f'<div style="display:flex;align-items:center;gap:0.75rem;">'
        f'<span style="color:{col};font-weight:700;font-size:0.9rem;background:{col}18;border:1px solid {col}55;'
        f'border-radius:6px;padding:2px 10px;white-space:nowrap;">{mr.regime}</span>'
        f'<span style="color:#8b949e;font-size:0.82rem;">🌐 IHSG &nbsp;·&nbsp; {mr.ihsg_trend}</span>'
        f'</div>'

        f'<div style="display:flex;align-items:baseline;gap:0.6rem;">'
        f'<span style="font-family:monospace;font-size:1.5rem;font-weight:700;color:#e6edf3;">{mr.ihsg_last:,.2f}</span>'
        f'<span style="font-family:monospace;font-size:1rem;font-weight:600;color:{chg_col};">{arrow} {mr.ihsg_chg1d:+.2f}%</span>'
        f'<span style="font-size:0.78rem;color:#8b949e;">vs prev {mr.ihsg_prev:,.2f}</span>'
        f'</div>'

        f'<span style="font-size:0.82rem;color:{col};white-space:nowrap;">{mr.warning}</span>'
        f'</div>'

        f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:0.4rem 1.6rem;margin-top:0.55rem;">'

        f'<span style="font-size:0.78rem;color:#8b949e;font-family:monospace;">'
        f'O&nbsp;<span style="color:#c9d1d9;">{mr.ihsg_open:,.2f}</span>'
        f'&nbsp;H&nbsp;<span style="color:#00e676;">{mr.ihsg_high:,.2f}</span>'
        f'&nbsp;L&nbsp;<span style="color:#f44336;">{mr.ihsg_low:,.2f}</span>'
        f'</span>'

        f'<span style="color:#30363d;">|</span>'

        f'<span style="font-size:0.78rem;color:#8b949e;">'
        f'5d:&nbsp;<span style="color:{chg5_col};font-weight:600;">{mr.ihsg_change_5d:+.2f}%</span>'
        f'&nbsp;&nbsp;20d:&nbsp;<span style="color:{chg20_col};font-weight:600;">{mr.ihsg_change_20d:+.2f}%</span>'
        f'</span>'

        f'<span style="color:#30363d;">|</span>'

        f'<span style="font-size:0.78rem;color:#8b949e;">'
        f'Sizing:&nbsp;<span style="color:{col};font-weight:600;">{mr.multiplier:.0%}</span>'
        f'</span>'

        f'<div style="flex:1;min-width:160px;max-width:300px;">'
        f'<div style="display:flex;justify-content:space-between;font-size:0.68rem;color:#8b949e;margin-bottom:2px;font-family:monospace;">'
        f'<span>L {mr.ihsg_low:,.0f}</span><span>Day Range</span><span>H {mr.ihsg_high:,.0f}</span>'
        f'</div>'
        f'<div style="background:#21262d;border-radius:4px;height:6px;position:relative;">'
        f'<div style="position:absolute;left:{pos:.1f}%;transform:translateX(-50%);width:10px;height:6px;border-radius:3px;background:{chg_col};"></div>'
        f'</div>'
        f'</div>'

        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# ADAPTIVE FIBONACCI  (lookback menyesuaikan regime & volatilitas)
# ══════════════════════════════════════════════════════════════
def adaptive_lookback(df: pd.DataFrame, base_days: int, mr: MarketRegime) -> int:
    """
    Sesuaikan lookback Fibonacci berdasarkan:
    - Volatilitas saham (ATR%): saham volatile → lookback lebih panjang
    - Market regime: BEAR → lookback lebih pendek (swing lebih dekat)
    - ADX: trending kuat → lookback lebih panjang untuk tangkap swing besar
    """
    close = s(df["Close"])
    high  = s(df["High"])
    low   = s(df["Low"])

    atr_pct = float(
        AverageTrueRange(high, low, close, window=14)
        .average_true_range().iloc[-1]
    ) / float(close.iloc[-1]) * 100

    adx_val = float(
        s(ADXIndicator(high, low, close, window=14).adx()).iloc[-1]
    )

    # Base adjustment
    adj = base_days

    # Volatilitas tinggi → panjangkan lookback (swing lebih ekstrem)
    if atr_pct > 3.0:   adj = int(adj * 1.3)
    elif atr_pct < 1.0: adj = int(adj * 0.8)

    # Trending kuat → panjangkan (swing primer lebih relevan)
    if adx_val > 35:    adj = int(adj * 1.2)

    # Bear market → persingkat (swing baru lebih relevan dari swing lama)
    if mr.regime == "BEAR":  adj = int(adj * 0.7)

    return max(30, min(365, adj))


# ══════════════════════════════════════════════════════════════
# EXIT SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════
@dataclass
class ExitSignal:
    level: str       # "HOLD" / "WATCH" / "PARTIAL_EXIT" / "FULL_EXIT"
    alasan: list[str]
    aksi: list[str]
    urgency: int     # 0=hold, 1=watch, 2=partial, 3=exit sekarang

def build_exit_signal(h: dict, df: pd.DataFrame) -> ExitSignal:
    """
    Deteksi sinyal exit dari posisi yang sudah dipegang.
    Berjalan independen dari entry signal — khusus untuk position management.
    """
    alasan: list[str] = []
    aksi:   list[str] = []
    exit_score = 0    # makin tinggi makin perlu exit

    close  = s(df["Close"])
    high_s = s(df["High"])
    low_s  = s(df["Low"])
    lvl    = h["lvl"]

    # ── 1. HA berubah merah setelah konsisten hijau ──
    ha_df   = heikin_ashi(df)
    ha_c    = ha_df["HA_C"].values
    ha_o    = ha_df["HA_O"].values
    # Hitung berapa candle hijau sebelum yang terakhir
    was_green_streak = 0
    for i in range(len(ha_c)-2, max(0, len(ha_c)-10), -1):
        if ha_c[i] > ha_o[i]: was_green_streak += 1
        else: break
    ha_just_turned_red = (ha_c[-1] < ha_o[-1]) and was_green_streak >= 3
    if ha_just_turned_red:
        exit_score += 2
        alasan.append(f"🔴 HA baru berubah merah setelah {was_green_streak} candle hijau — momentum berbalik")
        aksi.append("📤 Partial exit 50% posisi — lindungi profit")

    # ── 2. TK cross DOWN (Tenkan potong Kijun dari atas ke bawah) ──
    ichi    = IchimokuIndicator(high=high_s, low=low_s)
    tk_s    = s(ichi.ichimoku_conversion_line())
    kj_s    = s(ichi.ichimoku_base_line())
    tk_kj_diff = tk_s - kj_s
    tk_cross_down = (float(tk_kj_diff.iloc[-1]) < 0) and (float(tk_kj_diff.iloc[-2]) >= 0)
    if tk_cross_down:
        exit_score += 2
        alasan.append("⚡ Tenkan cross DOWN Kijun — sinyal exit Ichimoku klasik")
        aksi.append("🛑 Exit atau reduce signifikan — TK cross down adalah sinyal bearish kuat")

    # ── 3. MACD cross DOWN dari zona positif ──
    macd_obj  = MACD(close)
    macd_hist = s(macd_obj.macd_diff())
    macd_cross_down = (float(macd_hist.iloc[-1]) < 0) and (float(macd_hist.iloc[-2]) >= 0)
    if macd_cross_down:
        exit_score += 1
        alasan.append(f"📉 MACD cross DOWN — momentum bullish habis (hist: {float(macd_hist.iloc[-1]):+.3f})")

    # ── 4. Harga menyentuh / melewati Target 1 ──
    if h["harga"] >= lvl["target_1"] * 0.98:
        exit_score += 1
        alasan.append(f"🎯 Harga mendekati Target 1 ({lvl['target_1']:,.0f}) — area take profit")
        aksi.append(f"💰 Take profit 50–70% di sekitar {lvl['target_1']:,.0f}")
        aksi.append(f"🔒 Sisakan 30% untuk kejar Target 2 ({lvl['target_2']:,.0f})")

    # ── 5. Distribution trap setelah posisi masuk ──
    if h["dist_trap"]:
        exit_score += 2
        alasan.append("🚨 Distribution trap terdeteksi — institusi kemungkinan mulai distribusi")
        aksi.append("📤 Pertimbangkan exit 50–75% sekarang")

    # ── 6. Harga turun ke bawah cutloss ──
    if h["harga"] < lvl["cutloss"]:
        exit_score = 10   # override — wajib exit
        alasan.append(f"🛑 Harga {h['harga']:,.0f} di bawah cutloss {lvl['cutloss']:,.0f} — CUTLOSS!")
        aksi.append("❗ CUTLOSS SEKARANG — jangan averaging down, lindungi modal")

    # ── 7. Volume spike merah (selling climax?) ──
    adv20   = float(s(df["Volume"]).iloc[-21:-1].mean())
    vol_now = float(s(df["Volume"]).iloc[-1])
    vol_rel = vol_now / adv20 if adv20 > 0 else 1
    if vol_rel > 3 and not h["ha_green"] and h["cqs"] < 0.4:
        exit_score += 2
        alasan.append(f"📊 Volume {vol_rel:.1f}×ADV + candle merah + CQS {h['cqs']:.2f} — selling pressure besar")

    # ── 8. StochRSI overbought ekstrem ──
    if h["srsi_k"] > 90:
        exit_score += 1
        alasan.append(f"🔥 StochRSI sangat overbought ({h['srsi_k']:.0f}) — potensi reversal tajam")

    # ── Tentukan level exit ──
    if exit_score == 0:
        level   = "HOLD"
        urgency = 0
        alasan.append("✅ Tidak ada sinyal exit — posisi aman, lanjutkan hold")
        aksi.append(f"👁️ Monitor: alert jika close di bawah {lvl['cutloss']:,.0f}")
    elif exit_score <= 1:
        level   = "WATCH"
        urgency = 1
        aksi.insert(0, "👁️ Mulai perketat monitoring — belum perlu exit tapi waspadai")
    elif exit_score <= 3:
        level   = "PARTIAL_EXIT"
        urgency = 2
        if not aksi:
            aksi.insert(0, "📤 Partial exit 30–50% — amankan sebagian profit")
    else:
        level   = "FULL_EXIT"
        urgency = 3
        if not any("CUTLOSS" in a or "exit" in a.lower() for a in aksi):
            aksi.insert(0, "🚨 Pertimbangkan full exit — multiple sinyal negatif sekaligus")

    return ExitSignal(level=level, alasan=alasan, aksi=aksi, urgency=urgency)


def render_exit_signal(ex: ExitSignal):
    color = {"HOLD":"#00e676","WATCH":"#ffc107","PARTIAL_EXIT":"#ff9800","FULL_EXIT":"#f44336"}.get(ex.level,"#ffc107")
    icon  = {"HOLD":"✅","WATCH":"👁️","PARTIAL_EXIT":"📤","FULL_EXIT":"🚨"}.get(ex.level,"⚠️")
    label = ex.level.replace("_"," ")
    st.markdown(f"""
    <div style="background:#161b22;border:1px solid {color};border-radius:10px;
                padding:1rem 1.4rem;margin:0.8rem 0;">
      <div style="font-size:1.1rem;font-weight:700;color:{color};margin-bottom:0.5rem;">
        {icon} EXIT SIGNAL: {label}
      </div>
    </div>
    """, unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Alasan:**")
        for a in ex.alasan: st.markdown(f"- {a}")
    with col_r:
        st.markdown("**Aksi:**")
        for a in ex.aksi: st.markdown(f"- {a}")


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
    except Exception: return fallback


def download_batch_parallel(tickers: list[str], max_workers: int = 8) -> dict[str, pd.DataFrame]:
    """Download semua ticker secara paralel."""
    result = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut = {ex.submit(download_one, t): t for t in tickers}
        for f in concurrent.futures.as_completed(fut):
            t = fut[f]
            try:
                result[t] = f.result()
            except Exception:
                result[t] = pd.DataFrame()
    return result


# ══════════════════════════════════════════════════════════════
# FULL ANALYSER  (semua indikator dalam satu pass)
# ══════════════════════════════════════════════════════════════
def analyse(ticker: str, days: int, df: Optional[pd.DataFrame] = None,
            mr: Optional[MarketRegime] = None) -> Optional[dict]:
    use_live_price = df is None
    if df is None:
        df = download_one(ticker)
    if df is None or df.empty:
        return None

    # Adaptive lookback — sesuaikan dengan volatilitas & regime
    if mr is None:
        mr = get_market_regime()
    adaptive_days = adaptive_lookback(df, days, mr)

    close  = s(df["Close"])
    high   = s(df["High"])
    low    = s(df["Low"])
    volume = s(df["Volume"])
    if len(close) < 80:
        return None
    harga  = live_price(ticker, float(close.iloc[-1])) if use_live_price else float(close.iloc[-1])

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
    srsi_k = finite_float(stoch_rsi.rolling(3).mean().iloc[-1], 50.0)
    srsi_d = finite_float(stoch_rsi.rolling(3).mean().rolling(3).mean().iloc[-1], 50.0)
    srsi_oversold   = srsi_k < 20
    srsi_overbought = srsi_k > 80
    srsi_cross_up   = srsi_k > srsi_d and srsi_k < 50

    # ── ATR (volatility sizing) ──
    atr_val = finite_float(AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1], 0.0)
    atr_pct = atr_val / harga * 100 if harga > 0 else 0

    # ── RSI 14 standard ──
    rsi_now = finite_float(rsi14.iloc[-1], 50.0)

    # ── Volume ──
    adv20   = finite_float(volume.iloc[-21:-1].mean(), 0.0)
    vol_rel = round(float(volume.iloc[-1]) / adv20, 2) if adv20 > 0 else 0.0

    # ── Volume Profile support (simplified: VWAP area) ──
    vwap_denom = finite_float(volume.rolling(20).sum().iloc[-1], 0.0)
    vwap = finite_float((close * volume).rolling(20).sum().iloc[-1] / vwap_denom, harga) if vwap_denom > 0 else harga
    price_above_vwap = harga > vwap

    # ── Fibonacci (adaptive lookback) ──
    sh, sl = recent_swings(df, adaptive_days)
    diff = sh - sl
    if diff <= 0 or not np.isfinite(diff):
        return None
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
    ha_flags   = ha["HA_C"] > ha["HA_O"]
    ha_seq     = streak_count(ha_flags, ha_green)
    ha_red_seq = streak_count(ha_flags, False)
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

    # ── Bollinger Band Squeeze (volatility contraction → explosive move imminent) ──
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_mid_now   = finite_float(bb_mid.iloc[-1], harga)
    bb_width     = finite_float((bb_upper - bb_lower).iloc[-1], 0.0) / bb_mid_now * 100 if bb_mid_now > 0 else 0.0
    bb_width_avg = finite_float((bb_upper - bb_lower).rolling(50).mean().iloc[-1], bb_width) / bb_mid_now * 100 if bb_mid_now > 0 else bb_width
    bb_squeeze   = bb_width < bb_width_avg * 0.75        # bandwidth < 75% dari rata-rata → squeeze
    bb_price_pos = float(close.iloc[-1]) / bb_mid_now if bb_mid_now > 0 else 1.0  # > 1 = di atas mid = bullish
    bb_breakout_up = (float(close.iloc[-1]) > float(bb_upper.iloc[-2])  # close tembus upper band
                      and float(close.iloc[-2]) <= float(bb_upper.iloc[-2]))

    # ── Higher High / Higher Low (market structure) ──
    # Cek 3 swing high dan 3 swing low terakhir dalam window 60 hari
    _hi60 = s(df["High"]).tail(60).values
    _lo60 = s(df["Low"]).tail(60).values
    _cl60 = s(df["Close"]).tail(60).values
    # Cari pivot high/low dengan window 5
    _pw = 5
    _ph_idx = [i for i in range(_pw, len(_hi60) - _pw)
               if _hi60[i] == max(_hi60[i - _pw: i + _pw + 1])]
    _pl_idx = [i for i in range(_pw, len(_lo60) - _pw)
               if _lo60[i] == min(_lo60[i - _pw: i + _pw + 1])]
    hh_hl = False   # Higher High + Higher Low (bullish structure)
    lh_ll = False   # Lower High + Lower Low (bearish structure)
    if len(_ph_idx) >= 2 and len(_pl_idx) >= 2:
        hh = _hi60[_ph_idx[-1]] > _hi60[_ph_idx[-2]]
        hl = _lo60[_pl_idx[-1]] > _lo60[_pl_idx[-2]]
        lh = _hi60[_ph_idx[-1]] < _hi60[_ph_idx[-2]]
        ll = _lo60[_pl_idx[-1]] < _lo60[_pl_idx[-2]]
        hh_hl = hh and hl
        lh_ll = lh and ll

    # ── MACD Histogram Slope (accelerating vs decelerating momentum) ──
    macd_series   = s(macd_obj.macd_diff())
    macd_slope    = float(macd_series.iloc[-1]) - float(macd_series.iloc[-3])  # slope 3 bar
    macd_accel    = macd_slope > 0 and macd_hist > 0   # positif & makin kuat
    macd_decel    = macd_slope < 0 and macd_hist > 0   # positif tapi melemah
    macd_neg_accel = macd_slope < 0 and macd_hist < 0  # negatif & makin lemah

    # ── Multi-day Accumulation Pattern ──
    # Smart money masuk bertahap: 3+ dari 5 hari terakhir volume di atas ADV + close > open
    _v5   = volume.iloc[-5:].values
    _c5   = close.iloc[-5:].values
    _o5   = s(df["Open"]).iloc[-5:].values
    _adv5 = float(volume.iloc[-26:-1].mean())
    accum_days = int(sum(1 for i in range(5)
                         if _v5[i] > _adv5 * 1.1 and _c5[i] > _o5[i]))
    multi_day_accum = accum_days >= 3   # ≥3 dari 5 hari = pola akumulasi kuat

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

    # ── Relative Strength vs IHSG ──
    # Butuh data IHSG — ambil dari cache market regime kalau sudah ada,
    # atau download sekali (di-cache 15 menit)
    rs_5d = rs_20d = rs_60d = 0.0
    beta_60 = 1.0
    price_relative_trend = False   # ratio saham/IHSG sedang naik?
    vol_divergence = False         # volume naik tapi IHSG turun (akumulasi tersembunyi)
    rs_score = 0                   # composite RS score 0-100

    try:
        df_ihsg = _get_ihsg_cached()
        if df_ihsg is not None and not df_ihsg.empty and len(df_ihsg) >= 22:
            ihsg_c = s(df_ihsg["Close"])

            # Align panjang — pakai tanggal yang sama
            min_len = min(len(close), len(ihsg_c))
            c_al    = close.iloc[-min_len:].values
            i_al    = ihsg_c.iloc[-min_len:].values

            def _ret(arr, n):
                if len(arr) > n:
                    return (arr[-1] / arr[-n-1] - 1) * 100
                return 0.0

            ret_s5  = _ret(c_al, 5);  ret_i5  = _ret(i_al, 5)
            ret_s20 = _ret(c_al, 20); ret_i20 = _ret(i_al, 20)
            ret_s60 = _ret(c_al, 60); ret_i60 = _ret(i_al, 60)

            rs_5d  = round(ret_s5  - ret_i5,  2)
            rs_20d = round(ret_s20 - ret_i20, 2)
            rs_60d = round(ret_s60 - ret_i60, 2)

            # Beta 60 hari: cov(saham, IHSG) / var(IHSG)
            if min_len >= 61:
                s_ret = np.diff(c_al[-61:]) / c_al[-62:-1]
                i_ret = np.diff(i_al[-61:]) / i_al[-62:-1]
                var_i = float(np.var(i_ret))
                beta_60 = round(float(np.cov(s_ret, i_ret)[0, 1]) / var_i, 2) if var_i > 0 else 1.0

            # Price Relative Ratio trend (saham/IHSG ratio naik dalam 5 hari?)
            if min_len >= 7:
                pr_now  = c_al[-1]  / i_al[-1]
                pr_5ago = c_al[-6]  / i_al[-6]
                price_relative_trend = pr_now > pr_5ago

            # Volume Divergence: IHSG turun 5 hari tapi volume saham naik
            ihsg_down_5d = ret_i5 < -0.5
            vol_up_5d    = float(volume.iloc[-1]) > float(volume.iloc[-6]) * 1.1 if len(volume) > 6 else False
            vol_divergence = ihsg_down_5d and vol_up_5d and harga >= float(close.iloc[-6]) * 0.99

            # RS composite score 0-100
            # Komponen: RS5d, RS20d, RS60d, price_relative_trend, vol_divergence, beta
            rs_pts = 0
            if rs_5d  > 2:   rs_pts += 25
            elif rs_5d > 0:  rs_pts += 12
            if rs_20d > 3:   rs_pts += 25
            elif rs_20d > 0: rs_pts += 12
            if rs_60d > 5:   rs_pts += 20
            elif rs_60d > 0: rs_pts += 10
            if price_relative_trend: rs_pts += 15
            if vol_divergence:       rs_pts += 15
            rs_score = min(100, rs_pts)

    except Exception:
        pass  # RS gagal tidak boleh blok analisis utama

    # ── Bear Market Rally Detector ──
    # Saham yang RS kuat + volume spike + oversold = kandidat rebound taktis.
    rsi_deep_oversold = rsi_now < 40
    vol_surge_3d      = finite_float(volume.iloc[-3:].mean(), 0.0) > _adv5 * 1.5 if _adv5 > 0 else False
    bear_rally_candidate = (
        rs_score >= 55
        and rsi_deep_oversold
        and vol_surge_3d
        and not dist_trap
        and cqs > 0.45
    )

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
        ha_green=ha_green, ha_seq=ha_seq, ha_red_seq=ha_red_seq, ha_no_tail=ha_no_tail,
        # candle quality & distribution
        cqs=cqs, dist_trap=dist_trap, accum_confirm=accum_confirm,
        shooting_star=shooting_star, upper_shadow_ratio=upper_shadow_ratio,
        obv_bull=obv_bull,
        # early bird
        early_score=early_score,
        kumo_twist_recent=kumo_twist_recent, kumo_narrowing=kumo_narrowing,
        price_near_cloud=price_near_cloud, tk_cross_new=tk_cross_new,
        chikou_near_free=chikou_near_free,
        # relative strength
        rs_5d=rs_5d, rs_20d=rs_20d, rs_60d=rs_60d,
        beta_60=beta_60,
        price_relative_trend=price_relative_trend,
        vol_divergence=vol_divergence,
        rs_score=rs_score,
        # adaptive
        adaptive_days=adaptive_days,
        # v5 additions
        bb_squeeze=bb_squeeze, bb_breakout_up=bb_breakout_up, bb_price_pos=bb_price_pos, bb_width=bb_width,
        hh_hl=hh_hl, lh_ll=lh_ll,
        macd_slope=macd_slope, macd_accel=macd_accel, macd_decel=macd_decel, macd_neg_accel=macd_neg_accel,
        multi_day_accum=multi_day_accum, accum_days=accum_days,
        bear_rally_candidate=bear_rally_candidate,
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
    # ── Trend primer (bobot terbesar — paling prediktif) ──
    "cloud_pos"    : 2.5,   # ichimoku cloud position
    "tk_kj"        : 1.0,   # tenkan > kijun
    "chikou"       : 0.8,   # chikou konfirmasi
    "ema_trend"    : 1.2,   # ema20 > ema50
    "hh_hl"        : 1.5,   # higher high + higher low (market structure)
    # ── Entry timing ──
    "fib_zone"     : 2.5,   # di zona entry fibonacci
    "price_ema"    : 0.8,   # harga > ema20
    "srsi_timing"  : 1.0,   # stochrsi timing
    "bb_squeeze"   : 0.8,   # bollinger squeeze → explosive move imminent
    "bb_breakout"  : 1.0,   # breakout dari upper band
    # ── Momentum confirmation ──
    "macd"         : 1.5,   # macd positif
    "macd_cross"   : 0.5,   # macd cross up
    "macd_accel"   : 0.7,   # histogram makin besar (momentum accelerating)
    "adx"          : 0.5,   # adx kuat
    # ── Candle & volume quality ──
    "ha"           : 1.0,   # ha green
    "ha_seq"       : 0.5,   # ha konsisten
    "ha_no_tail"   : 0.3,   # momentum penuh
    "volume"       : 0.8,   # volume konfirmasi
    "multi_accum"  : 1.2,   # multi-day accumulation pattern
    "vwap"         : 0.6,   # di atas vwap
    "rsi_extreme"  : 0.5,   # rsi oversold/overbought
}
MAX_SCORE = sum(WEIGHTS.values())   # ~21.5

def build_plan(h: dict) -> TradingSignal:
    score  = 0.0
    alasan : list[str] = []
    aksi   : list[str] = []
    lvl    = h["lvl"]
    p      = h["harga"]

    # ═══════════════════════════════════════════════════
    # CONFLUENCE GATE — indikator primer wajib terpenuhi
    # Sinyal BUY tidak bisa muncul jika cloud + EMA + structure semuanya bearish
    # ═══════════════════════════════════════════════════
    _cloud_ok    = h["di_atas"] or h["di_dalam"]
    _ema_ok      = h["ema_bull"] or h["price_ema20"]
    _structure_ok = h["hh_hl"] or h["tk_kj"]
    hard_bearish = (not _cloud_ok) and (not _ema_ok) and (not _structure_ok)

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

    # ── B. Market Structure (HH/HL) ──
    if h.get("hh_hl"):
        score += WEIGHTS["hh_hl"]
        alasan.append("📈 Higher High + Higher Low — struktur uptrend terkonfirmasi")
    elif h.get("lh_ll"):
        score -= WEIGHTS["hh_hl"]
        alasan.append("📉 Lower High + Lower Low — struktur downtrend, hindari entry baru")
    else:
        alasan.append("➡️ Struktur harga sideways — belum ada trend jelas")

    # ── C. Fibonacci zone ──
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

    # ── D. EMA trend ──
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

    # ── E. MACD + slope ──
    if h["macd_cross_up"]:
        score += WEIGHTS["macd"] + WEIGHTS["macd_cross"]
        alasan.append(f"🚀 MACD baru cross UP — sinyal momentum kuat! (hist:{h['macd_hist']:+.2f})")
    elif h["macd_bull"]:
        score += WEIGHTS["macd"]
        if h.get("macd_accel"):
            score += WEIGHTS["macd_accel"]
            alasan.append(f"🚀 MACD histogram positif & ACCELERATING (slope:{h.get('macd_slope',0):+.4f}) — momentum makin kuat!")
        elif h.get("macd_decel"):
            alasan.append(f"⚠️ MACD positif tapi MELAMBAT — momentum mulai melemah (hist:{h['macd_hist']:+.2f})")
        else:
            alasan.append(f"✅ MACD histogram positif ({h['macd_hist']:+.2f}) — momentum bullish")
    else:
        score -= WEIGHTS["macd"]
        if h.get("macd_neg_accel"):
            score -= WEIGHTS["macd_accel"] * 0.5
            alasan.append(f"❌ MACD negatif & makin lemah — selling pressure intensif ({h['macd_hist']:+.2f})")
        else:
            alasan.append(f"❌ MACD histogram negatif ({h['macd_hist']:+.2f}) — momentum bearish")

    # ── F. ADX ──
    if h["adx_strong"]:
        score += WEIGHTS["adx"]
        alasan.append(f"💪 ADX {h['adx']:.1f} — trend kuat (>25)")
    else:
        alasan.append(f"➡️ ADX {h['adx']:.1f} — trend lemah/sideways (<25)")

    # ── G. Bollinger Band Squeeze ──
    if h.get("bb_breakout_up"):
        score += WEIGHTS["bb_breakout"]
        alasan.append(f"💥 BB Breakout UP! Harga tembus upper band — momentum eksplosif!")
    elif h.get("bb_squeeze"):
        score += WEIGHTS["bb_squeeze"]
        alasan.append(f"🔥 Bollinger Band Squeeze (width {h.get('bb_width',0):.1f}%) — volatilitas terkompresi, breakout imminent!")
    else:
        bb_pos = h.get("bb_price_pos", 1.0)
        if bb_pos > 1.02:
            alasan.append(f"📊 Harga di upper BB zone ({bb_pos:.2f}× mid) — bullish tapi waspadai overbought")
        elif bb_pos < 0.98:
            alasan.append(f"📊 Harga di lower BB zone ({bb_pos:.2f}× mid) — area potensial reversal")

    # ── H. Heikin Ashi ──
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
        red_seq = h.get("ha_red_seq", h["ha_seq"])
        if red_seq >= 3:
            score -= WEIGHTS["ha_seq"]
            alasan.append(f"🔴 HA merah {red_seq} candle — tekanan jual dominan")
        else:
            alasan.append(f"🔴 HA merah {red_seq} candle")

    # ── I. Stochastic RSI ──
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

    # ── J. Volume & VWAP ──
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

    # Multi-day accumulation
    if h.get("multi_day_accum"):
        score += WEIGHTS["multi_accum"]
        alasan.append(f"🏦 Akumulasi multi-hari: {h.get('accum_days',0)}/5 hari bullish volume — smart money masuk bertahap!")
    elif h.get("accum_days", 0) >= 2:
        score += WEIGHTS["multi_accum"] * 0.4
        alasan.append(f"📊 {h.get('accum_days',0)}/5 hari akumulasi — pola mulai terbentuk")

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

    # RSI
    if h["rsi"] < 30:
        score += WEIGHTS["rsi_extreme"]
        alasan.append(f"💧 RSI oversold ({h['rsi']:.0f}) — area murah historis")
    elif h["rsi"] > 70:
        score -= WEIGHTS["rsi_extreme"]
        alasan.append(f"🔥 RSI overbought ({h['rsi']:.0f}) — area mahal")
    else:
        alasan.append(f"📈 RSI {h['rsi']:.0f} — normal range")

    # ── K. Relative Strength vs IHSG ──
    rs_bonus = 0.0
    rs_5d    = h.get("rs_5d", 0)
    rs_20d   = h.get("rs_20d", 0)
    beta     = h.get("beta_60", 1.0)
    vd       = h.get("vol_divergence", False)
    prt      = h.get("price_relative_trend", False)

    if rs_5d > 3 and rs_20d > 3:
        rs_bonus += 1.5
        alasan.append(f"🚀 RS sangat kuat: +{rs_5d:.1f}% vs IHSG 5d, +{rs_20d:.1f}% vs IHSG 20d")
    elif rs_5d > 0 and rs_20d > 0:
        rs_bonus += 0.8
        alasan.append(f"✅ RS positif: +{rs_5d:.1f}% vs IHSG 5d, +{rs_20d:.1f}% vs IHSG 20d")
    elif rs_5d < -3:
        rs_bonus -= 1.0
        alasan.append(f"⚠️ RS negatif: {rs_5d:.1f}% vs IHSG 5d — underperform market")

    if beta < 0.5:
        rs_bonus += 0.5
        alasan.append(f"🛡️ Beta rendah ({beta:.2f}) — defensif, tidak terseret IHSG")
    elif beta > 1.5:
        rs_bonus -= 0.3
        alasan.append(f"⚡ Beta tinggi ({beta:.2f}) — amplify gerakan IHSG (dua arah)")

    if vd:
        rs_bonus += 1.0
        alasan.append("🏦 Volume divergence! Volume naik saat IHSG turun — akumulasi tersembunyi")

    if prt:
        rs_bonus += 0.5
        alasan.append("📈 Price Relative Ratio naik — saham outperform IHSG dalam 5 hari terakhir")

    score += rs_bonus

    # ═══════════════════════════════════════════════════
    # NORMALIZE + SINYAL
    # ═══════════════════════════════════════════════════
    confidence = int(min(100, max(0, (score / MAX_SCORE) * 100 + 50)))
    norm       = score / MAX_SCORE   # roughly -1 to +1

    # Hard bearish gate: kalau 3 indikator primer semuanya merah → max WAIT
    if hard_bearish and norm >= 0.10:
        norm = 0.09   # paksa ke batas atas WAIT

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
    if p >= lvl["target_2"]:      zona = "Di atas Target 2 (161.8%)"
    elif p >= lvl["target_1"]:    zona = "Di area Swing High / Target 1"
    elif p >= lvl["entry_atas"]:  zona = "Antara SH & Entry Atas (23.6%–38.2%)"
    elif p >= lvl["entry_bawah"]: zona = "🎯 ZONA ENTRY OPTIMAL (38.2%–50%)"
    elif p >= lvl["cutloss"]:     zona = "Antara Entry Bawah & Cutloss (50%–61.8%)"
    elif p >= lvl["swing_low"]:   zona = "⛔ Di bawah Cutloss — Bearish"
    else:                          zona = "💀 Di bawah Swing Low — Sangat Bearish"

    # ── ATR-based sizing label ──
    atr_pct = h["atr_pct"]
    if atr_pct < 1.5:    atr_label = f"Volatilitas Rendah ({atr_pct:.1f}% ATR) → bisa full sizing"
    elif atr_pct < 3.0:  atr_label = f"Volatilitas Sedang ({atr_pct:.1f}% ATR) → sizing 50-75%"
    else:                atr_label = f"Volatilitas Tinggi ({atr_pct:.1f}% ATR) → hati-hati, sizing 25-40%"

    # ── R/R ──
    # Untuk saham yang belum berada di zona entry, R/R sebaiknya dihitung dari
    # rencana eksekusi, bukan harga saat ini. Kalau harga sudah lewat entry atas,
    # gunakan harga saat ini agar risiko chasing tetap terlihat.
    if h["below_cut"]:
        entry_ref = None
    elif h["in_entry"]:
        entry_ref = p
    else:
        entry_ref = lvl["entry_bawah"]

    if entry_ref is None or entry_ref <= lvl["cutloss"] or lvl["target_1"] <= entry_ref:
        rr_val = 0.0
    else:
        rr_val = (lvl["target_1"] - entry_ref) / (entry_ref - lvl["cutloss"])

    # ── R/R GATE: sinyal BUY wajib punya R/R ≥ 1.5 ──
    if "BUY" in sinyal and (h["below_cut"] or rr_val < 1.5):
        # Turunkan sinyal satu level
        if sinyal == "STRONG BUY":
            sinyal, css, sizing = "BUY", "BUY", 75
        elif sinyal == "BUY":
            sinyal, css, sizing = "SPEC. BUY", "SPEC-BUY", 40
        else:
            sinyal, css, sizing = "WAIT", "WAIT", 0
        if h["below_cut"]:
            alasan.append("🛑 Harga sudah di bawah cutloss — semua sinyal BUY dibatalkan")
            sinyal, css, sizing = "WAIT", "WAIT", 0
        else:
            alasan.append(f"⚠️ R/R {rr_val:.1f}:1 terlalu rendah (min 1.5:1) — sinyal diturunkan satu level")

    # ═══════════════════════════════════════════════════
    # ACTION PLAN
    # ═══════════════════════════════════════════════════

    # Override 1: Distribution trap
    if h["dist_trap"] and "BUY" in sinyal:
        sinyal = "WAIT"; css = "WAIT"; sizing = 0
        aksi.append("🚨 DISTRIBUTION TRAP terdeteksi — sinyal BUY DIBATALKAN!")
        aksi.append(f"🔍 Volume {h['vol_rel']:.1f}×ADV tapi CQS {h['cqs']:.2f} — close terlalu dekat LOW")
        aksi.append("⏳ Tunggu 1–3 hari: entry aman jika CQS > 0.6 pada volume tinggi berikutnya")

    # Override 2: Bear Rally Candidate — khusus saat pasar bearish
    elif h.get("bear_rally_candidate") and "BUY" in sinyal:
        alasan.append("🔥 BEAR RALLY CANDIDATE — RS kuat + oversold + volume surge saat pasar merah!")
        aksi.append("🎯 BEAR RALLY PLAY: Entry agresif karena justru ini yang paling kencang saat IHSG rebound")
        aksi.append(f"🟢 Entry zona: {lvl['entry_bawah']:,.0f} – {lvl['entry_atas']:,.0f}")
        aksi.append(f"🛑 CUTLOSS KETAT: daily close < {lvl['cutloss']:,.0f} (jangan toleransi loss besar di bear market)")
        aksi.append(f"🎯 TARGET KONSERVATIF: {lvl['target_1']:,.0f} (ambil profit di T1, jangan rakus)")
        aksi.append(f"📐 R/R: 1:{rr_val:.1f} (berdasarkan entry rencana)")
        aksi.append(f"💰 {atr_label}")

    elif h["shooting_star"] and "BUY" in sinyal:
        aksi.append("⚠️ Ada pola shooting star — sizing lebih kecil, tunggu konfirmasi")
        aksi.append(f"🛑 CUTLOSS diperketat: jika close < {lvl['cutloss']:,.0f}")
        aksi.append(f"🟡 Entry sizing 50% saja: {lvl['entry_bawah']:,.0f} – {lvl['entry_atas']:,.0f}")
        aksi.append(f"🎯 TARGET 1: {lvl['target_1']:,.0f}   TARGET 2: {lvl['target_2']:,.0f}")
        aksi.append(f"📐 R/R ≈ 1:{rr_val:.1f} (berdasarkan entry rencana)")

    elif "BUY" in sinyal:
        if h["accum_confirm"] or h.get("multi_day_accum"):
            aksi.append("🏦 Akumulasi institusional terkonfirmasi — sinyal lebih kuat dari biasa!")
        if h.get("bb_squeeze"):
            aksi.append("🔥 BB Squeeze aktif — posisikan sebelum breakout untuk R/R terbaik!")
        if h["in_entry"]:
            aksi.append(f"🟢 ENTRY sekarang — harga {p:,.0f} di zona sweet spot")
        else:
            aksi.append(f"⏳ TUNGGU pullback ke zona entry: {lvl['entry_bawah']:,.0f} – {lvl['entry_atas']:,.0f}")
        aksi.append(f"🛑 CUTLOSS jika daily close < {lvl['cutloss']:,.0f}")
        aksi.append(f"🎯 TARGET 1: {lvl['target_1']:,.0f}   TARGET 2: {lvl['target_2']:,.0f}")
        aksi.append(f"📐 R/R ≈ 1:{rr_val:.1f} (berdasarkan entry rencana)")
        aksi.append(f"💰 {atr_label}")
        if sizing:
            aksi.append(f"🎲 Sizing saran: {sizing}% dari alokasi posisi")

    elif "WAIT" in sinyal:
        if not h["dist_trap"]:
            aksi.append("⏳ WAIT — konfluensi belum cukup kuat")
            aksi.append(f"👁️ Set alert di zona entry: {lvl['entry_bawah']:,.0f} – {lvl['entry_atas']:,.0f}")
            aksi.append("👁️ Tunggu: MACD cross up + HA hijau + CQS > 0.6 + volume konfirmasi")
            if h.get("bb_squeeze"):
                aksi.append("🔥 BB Squeeze terdeteksi — pantau ketat, breakout bisa terjadi tiba-tiba!")
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


def apply_regime_to_plan(sig: TradingSignal, mr: MarketRegime) -> TradingSignal:
    """
    Post-process: sesuaikan sizing dengan market regime multiplier.
    Bear market → sizing dikurangi otomatis, tambah warning di aksi.
    """
    if mr.regime == "BEAR" and sig.sizing_pct and sig.sizing_pct > 0:
        orig = sig.sizing_pct
        adjusted = max(0, int(orig * mr.multiplier))
        sig.sizing_pct = adjusted
        if adjusted == 0:
            sig.aksi.insert(0, f"🚨 Market BEAR — sizing 0% (skip entry meski sinyal bagus)")
        else:
            sig.aksi.insert(0, f"🌐 Market BEAR: sizing dikurangi {orig}% → {adjusted}% (×{mr.multiplier:.0%} regime multiplier)")
    elif mr.regime == "NEUTRAL" and sig.sizing_pct and sig.sizing_pct > 0:
        orig = sig.sizing_pct
        adjusted = max(0, int(orig * mr.multiplier))
        sig.sizing_pct = adjusted
        sig.aksi.insert(0, f"🌐 Market SIDEWAYS: sizing disesuaikan {orig}% → {adjusted}%")
    return sig


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
        cls      = 'class="ldr-current"' if is_now else ""
        fw       = "700" if is_now else "400"
        pct_from_now = ((level - p) / p * 100) if not is_now else 0
        pct_str  = f'<span style="color:#8b949e;font-size:0.72rem;">{pct_from_now:+.1f}%</span>' if not is_now else ""
        html += (
            f'<tr {cls}>'
            f'<td><span class="ldr-pill {pill}">{label}</span></td>'
            f'<td style="text-align:right;font-weight:{fw}">{level:,.0f}</td>'
            f'<td style="text-align:right;">{pct_str}</td>'
            f'</tr>'
        )
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
    is_green    = close.values >= open_s.values
    vol_colors  = np.where(is_green, CT["green"], CT["red"]).tolist()
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

    # Distribusi volume ke price bucket — vectorized via np.digitize
    typical  = (high_s + low_s + close) / 3
    price_min, price_max = float(typical.min()), float(typical.max())
    bin_edges   = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # np.digitize returns 1-based indices; clip to [0, bins-1]
    indices     = np.clip(np.digitize(typical.values, bin_edges) - 1, 0, bins - 1)
    vol_per_bin = np.bincount(indices, weights=vol_s.values, minlength=bins)

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

    # ── Warna per bar: vectorized ──
    bar_colors = np.where(
        np.arange(bins) == poc_idx, "#ffd600",
        np.where(
            (bin_centers >= val) & (bin_centers <= vah), "rgba(100,181,246,0.7)",
            np.where(bin_centers > vah, "rgba(0,230,118,0.55)", "rgba(239,83,80,0.55)")
        )
    ).tolist()

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


def render_charts(ticker: str, h: dict, ctx: str = "default"):
    """
    Render chart panel lengkap.
    ctx: string unik per call site ('analisis', 'screener', 'eb', 'heatmap')
    — dipakai sebagai bagian key widget supaya stable across reruns.
    """
    df = download_one(ticker)
    if df is None or df.empty:
        st.warning("Data tidak tersedia untuk chart."); return

    # Key stabil: kombinasi ctx + ticker, tidak bergantung posisi di script
    key_ct   = f"_ct_{ctx}_{ticker}"
    key_bars = f"_bars_{ctx}_{ticker}"
    if key_ct   not in st.session_state: st.session_state[key_ct]   = "Candle"
    if key_bars not in st.session_state: st.session_state[key_bars] = 120

    col_ct, col_bars = st.columns([2, 3])
    with col_ct:
        ct_idx = st.radio(
            "Tipe candle:", ["Candle", "Heikin Ashi"],
            horizontal=True,
            index=["Candle", "Heikin Ashi"].index(st.session_state[key_ct]),
            key=key_ct,   # key stabil → Streamlit ingat value-nya
        )
        # Tidak perlu assign manual — Streamlit sudah simpan ke session_state[key_ct]

    with col_bars:
        bar_opts = [60, 90, 120, 180, 252, 365, 500, 730]
        cur_bars = st.session_state[key_bars]
        if cur_bars not in bar_opts: cur_bars = 120
        st.select_slider(
            "Tampilkan berapa bar:",
            options=bar_opts,
            value=cur_bars,
            key=key_bars,  # key stabil
        )

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
        _vah_label = "✅ di bawah" if harga < vah else "⚠️ di atas"
        _val_label = "✅ di atas"  if harga > val else "⚠️ di bawah"
        if val <= harga <= vah:
            _va_msg = "🟢 Harga dalam Value Area — support kuat"
        elif harga > vah:
            _va_msg = "🔼 Di atas VA — buyer dominan"
        else:
            _va_msg = "🔽 Di bawah VA — seller dominan"
        st.markdown(
            f'<div style="font-size:0.8rem;line-height:2;">'
            f'<span style="color:#ffd600;">● POC</span>: {poc:,.0f}<br>'
            f'<span style="color:#42a5f5;">▲ VAH</span>: {vah:,.0f} {_vah_label}<br>'
            f'<span style="color:#42a5f5;">▼ VAL</span>: {val:,.0f} {_val_label}<br>'
            f'<span style="color:#ff9800;">● NOW</span>: {harga:,.0f}<br><br>'
            f'<span style="font-size:0.75rem;color:#8b949e;">{_va_msg}</span>'
            f'</div>',
            unsafe_allow_html=True)


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
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        body = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 600,
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "messages": [{"role": "user", "content": prompt}]
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=25) as r:
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
    st.markdown(
        f"<span style='color:#8b949e;font-size:0.78rem;'>"
        f"v4.1 &nbsp;·&nbsp; {len(daftar_saham)} emiten loaded</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    mode = st.radio("Mode", [
        "🔍 Analisis + Plan",
        "🚀 Screener Massal",
        "🐦 Early Bird",
        "🌡️ Sector Heatmap",
        "💪 RS Hunter",
    ], label_visibility="collapsed")
    st.divider()

    days        = st.slider("Lookback Fibonacci (hari)", 30, 365, 120, 10)
    require_ha  = st.toggle("Filter HA Hijau", value=True)
    min_vol_rel = st.slider("Min. Volume Relatif ×ADV20", 0.0, 3.0, 0.5, 0.1)
    min_conf    = st.slider("Min. Confidence (%)", 0, 100, 50, 5)
    st.divider()
    st.markdown("**🏥 Liquidity Filter**")
    use_health   = st.toggle("Aktifkan filter likuiditas", value=True)
    min_price    = st.number_input("Min. harga (Rp)", value=50, step=10)
    min_adv_juta = st.number_input("Min. ADV value (Rp juta/hari)", value=500, step=100)
    workers      = st.slider("Parallel workers (screener)", 2, 16, 8, 2,
                              help="Thread paralel untuk download. Lebih tinggi = lebih cepat.")
    st.divider()

    # ── IHSG mini status di sidebar ──
    _mr_sb  = get_market_regime()
    _col_sb = {"BULL": "#00e676", "NEUTRAL": "#ffc107", "BEAR": "#f44336"}.get(_mr_sb.regime, "#ffc107")
    _cc_sb  = "#00e676" if _mr_sb.ihsg_chg1d >= 0 else "#f44336"
    _c5_sb  = "#00e676" if _mr_sb.ihsg_change_5d >= 0 else "#f44336"
    _arrow  = "▲" if _mr_sb.ihsg_chg1d >= 0 else "▼"
    _sb_html = (
        f'<div style="background:#0d1117;border:1px solid {_col_sb}44;border-radius:8px;'
        f'padding:0.55rem 0.8rem;margin-bottom:0.6rem;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="font-size:0.72rem;color:#8b949e;">🌐 IHSG</span>'
        f'<span style="font-size:0.7rem;background:{_col_sb}22;color:{_col_sb};'
        f'border-radius:4px;padding:1px 7px;font-weight:700;">{_mr_sb.regime}</span>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-top:4px;">'
        f'<span style="font-family:monospace;font-size:1.05rem;font-weight:700;color:#e6edf3;">{_mr_sb.ihsg_last:,.2f}</span>'
        f'<span style="font-family:monospace;font-size:0.82rem;font-weight:600;color:{_cc_sb};">{_arrow} {_mr_sb.ihsg_chg1d:+.2f}%</span>'
        f'</div>'
        f'<div style="font-size:0.68rem;color:#8b949e;margin-top:3px;font-family:monospace;">'
        f'H&nbsp;<span style="color:#00e676;">{_mr_sb.ihsg_high:,.2f}</span>'
        f'&nbsp;L&nbsp;<span style="color:#f44336;">{_mr_sb.ihsg_low:,.2f}</span>'
        f'&nbsp;·&nbsp;5d&nbsp;<span style="color:{_c5_sb};">{_mr_sb.ihsg_change_5d:+.1f}%</span>'
        f'</div>'
        f'</div>'
    )
    st.markdown(_sb_html, unsafe_allow_html=True)

    # ── Indikator badge list ──
    _scipy_badge = (
        "<span style='background:#1b4332;color:#6ee7b7;border:1px solid #059669;"
        "border-radius:4px;padding:1px 7px;font-size:0.68rem;'>scipy ✓</span>"
        if HAS_SCIPY else
        "<span style='background:#312107;color:#fcd34d;border:1px solid #d97706;"
        "border-radius:4px;padding:1px 7px;font-size:0.68rem;'>scipy ✗ fallback</span>"
    )
    _indicators = [
        ("📈 Trend",    ["Ichimoku", "EMA 20/50", "ADX", "MACD"]),
        ("⏱️ Timing",   ["StochRSI", "Heikin Ashi", "OBV"]),
        ("📐 Fibo",     ["Adaptive Lookback", "Swing H/L", "Entry Zone"]),
        ("📊 Volume",   ["VWAP", "CQS", "Dist. Trap", "Accum."]),
        ("🔍 Screener", ["Early Bird", "RS Hunter", "Heatmap"]),
        ("🤖 AI",       ["Sentiment", "Web Search"]),
    ]
    badges_html = ""
    for group, items in _indicators:
        pills = " ".join(
            f"<span style='background:#161b22;color:#8b949e;border:1px solid #30363d;"
            f"border-radius:3px;padding:1px 5px;font-size:0.65rem;white-space:nowrap;'>{i}</span>"
            for i in items
        )
        badges_html += (
            f"<div style='margin-bottom:5px;'>"
            f"<span style='font-size:0.68rem;color:#6e7681;'>{group}</span><br>"
            f"<div style='display:flex;flex-wrap:wrap;gap:3px;margin-top:3px;'>{pills}</div>"
            f"</div>"
        )

    st.markdown(f"""
    <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;
                padding:0.6rem 0.8rem;font-family:'IBM Plex Mono',monospace;">
      <div style="margin-bottom:6px;">{_scipy_badge}</div>
      {badges_html}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# GLOBAL: fetch market regime SEKALI, render banner SEKALI
# ══════════════════════════════════════════════════════════════
mr = get_market_regime()
render_market_regime(mr)


# ══════════════════════════════════════════════════════════════
# MODE 1: ANALISIS + TRADING PLAN
# ══════════════════════════════════════════════════════════════
if "Analisis" in mode:
    st.markdown("## 🔍 Analisis Spesifik")

    c1, c2 = st.columns([4, 1])
    with c1:
        ticker = st.selectbox("Saham", daftar_saham, label_visibility="collapsed",
                              key="analisis_ticker")
    with c2:
        run = st.button("Analisis ▶", use_container_width=True, type="primary",
                        key="analisis_run")

    # Reset cache kalau ticker berubah
    if st.session_state.get("_analisis_last_ticker") != ticker:
        st.session_state["_analisis_h"]   = None
        st.session_state["_analisis_sig"] = None
        st.session_state["_analisis_hc"]  = None

    if run:
        with st.spinner(f"Menganalisis {ticker}…"):
            h_new = analyse(ticker, days, mr=mr)
        if not h_new:
            st.error("Data tidak cukup atau ticker tidak valid.")
        else:
            sig_new = build_plan(h_new)
            sig_new = apply_regime_to_plan(sig_new, mr)
            hc_new  = None
            if use_health:
                df_hc = download_one(ticker)
                hc_new = health_check(df_hc, ticker,
                                      min_price=min_price, min_adv_juta=min_adv_juta)
            st.session_state["_analisis_h"]            = h_new
            st.session_state["_analisis_sig"]          = sig_new
            st.session_state["_analisis_hc"]           = hc_new
            st.session_state["_analisis_last_ticker"]  = ticker

    # Render dari session_state — tidak hilang saat widget di-klik
    h   = st.session_state.get("_analisis_h")
    sig = st.session_state.get("_analisis_sig")
    hc  = st.session_state.get("_analisis_hc")

    if h and sig:
        if hc and not hc.lolos:
            st.warning(f"⚠️ **Health Check:** {hc.alasan}")
            for fl in hc.flags:
                st.markdown(f"- {fl}")
            st.info("Analisis tetap ditampilkan — gunakan dengan ekstra hati-hati.")

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

        # Adaptive lookback info
        st.caption(f"📐 Lookback Fibonacci adaptive: **{h.get('adaptive_days', days)} hari** "
                   f"(base {days}d, disesuaikan volatilitas & regime {mr.regime})")

        st.divider()
        tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Chart", "☁️ Ichimoku + EMA", "📐 Indikator Momentum",
            "📊 Volume + HA", "🚪 Exit Signal", "💪 Relative Strength", "📰 Sentimen Berita"
        ])

        with tab0:
            render_charts(ticker, h, ctx="analisis")

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
            st.markdown("#### 🚪 Exit Signal — Manajemen Posisi")
            st.caption("Khusus jika kamu sudah pegang saham ini — cek apakah perlu hold, partial exit, atau cut")
            df_exit = download_one(ticker)
            if df_exit is not None and not df_exit.empty:
                ex = build_exit_signal(h, df_exit)
                render_exit_signal(ex)
                # Adaptive exit info
                st.divider()
                st.caption(f"🌐 Regime market saat ini: **{mr.regime}** — "
                           f"{'exit lebih agresif di bear market' if mr.regime == 'BEAR' else 'normal exit rules berlaku'}")
            else:
                st.warning("Data tidak tersedia untuk exit signal.")

        with tab5:
            st.markdown("#### 💪 Relative Strength vs IHSG")
            st.caption("Seberapa kuat saham ini dibanding IHSG — deteksi outperformer sejati")

            rs_c1, rs_c2, rs_c3, rs_c4 = st.columns(4)
            rs_c1.metric("RS 5 Hari",  f"{h.get('rs_5d',0):+.2f}%",
                         delta="Outperform" if h.get('rs_5d',0)>0 else "Underperform")
            rs_c2.metric("RS 20 Hari", f"{h.get('rs_20d',0):+.2f}%",
                         delta="Outperform" if h.get('rs_20d',0)>0 else "Underperform")
            rs_c3.metric("RS 60 Hari", f"{h.get('rs_60d',0):+.2f}%",
                         delta="Outperform" if h.get('rs_60d',0)>0 else "Underperform")
            rs_c4.metric("Beta 60d",   f"{h.get('beta_60',1):.2f}",
                         delta="Defensif" if h.get('beta_60',1)<0.7 else None)

            st.divider()
            rs_c5, rs_c6, rs_c7 = st.columns(3)
            rs_c5.metric("RS Score",          f"{h.get('rs_score',0)}/100")
            rs_c6.metric("Price Relative",    "📈 Naik"  if h.get('price_relative_trend') else "📉 Turun")
            rs_c7.metric("Volume Divergence", "✅ Terdeteksi" if h.get('vol_divergence') else "❌ Tidak")

            if h.get('vol_divergence'):
                st.success("🏦 Volume divergence aktif — volume naik saat IHSG turun = akumulasi tersembunyi")
            if h.get('rs_5d',0) > 2 and h.get('rs_20d',0) > 2:
                st.success("🚀 RS kuat di semua timeframe — kandidat terbang saat IHSG rebound")
            elif h.get('rs_5d',0) < 0 and h.get('rs_20d',0) < 0:
                st.warning("⚠️ RS negatif — saham ini underperform IHSG, hindari dulu")

        with tab6:
            st.markdown("#### 📰 Sentimen Berita Terkini")
            st.caption("Diambil real-time via AI web search — cache 1 jam")
            score_s, label_s = render_sentiment(ticker)
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
                except Exception: data_map_hm[t] = pd.DataFrame()
                done += 1
                pbar.progress(done/total/2, text=f"⚡ Download: {done}/{total}")

        # Analisis semua — PARALLEL (heatmap butuh semua sinyal)
        def _hm_worker(ticker: str) -> Optional[tuple]:
            df_t = data_map_hm.get(ticker, pd.DataFrame())
            if df_t is None or df_t.empty:
                return None
            try:
                if use_health:
                    hc = health_check(df_t, ticker, min_price=min_price, min_adv_juta=min_adv_juta)
                    if not hc.lolos:
                        return None
                h = analyse(ticker, days, df=df_t, mr=mr)
                if not h:
                    return None
                sig = build_plan(h)
                sig = apply_regime_to_plan(sig, mr)
                sektor = get_sektor(ticker)
                return (ticker, h, sig, sektor)
            except Exception:
                return None

        hm_done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            hm_futs = {ex.submit(_hm_worker, t): t for t in daftar_saham}
            for fut in concurrent.futures.as_completed(hm_futs):
                hm_done += 1
                pbar.progress(0.5 + hm_done / total / 2,
                              text=f"🧠 Analisis: {hm_done}/{total}")
                try:
                    result = fut.result()
                except Exception:
                    continue
                if result is None:
                    continue
                ticker_r, h, sig, sektor = result
                st.session_state.heatmap_hasil.append({
                    "Ticker": ticker_r, "Sektor": sektor,
                    "Sinyal": sig.sinyal, "Conf.": sig.confidence,
                    "Harga": int(round(h["harga"])),
                })
                st.session_state.heatmap_raw[ticker_r] = (h, sig)

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
            render_charts(hm_sel, h, ctx="heatmap")
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
        if st.button("🔄 Reset", key="screen_reset", use_container_width=True):
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
                except Exception: data_map[t] = pd.DataFrame()
                done += 1
                pbar.progress(done / total / 2, text=f"⚡ Download: {done}/{total}")

        dl_time = time.time() - t0
        status.markdown(f"✅ **Fase 1 selesai** — {total} ticker diunduh dalam **{dl_time:.1f}s**")

        # ── Fase 2: Analisis & scoring — PARALEL ──
        status.markdown("🧠 **Fase 2/2** — Analisis paralel + scoring…")

        # Ambil close terakhir sebagai fallback harga (tanpa HTTP)
        price_map: dict[str, float] = {}
        for tk, df_tk in data_map.items():
            if df_tk is not None and not df_tk.empty:
                try: price_map[tk] = float(s(df_tk["Close"]).iloc[-1])
                except Exception: pass

        def _analyse_worker(ticker: str) -> Optional[tuple]:
            """Jalankan full pipeline satu ticker — dipanggil dari thread pool."""
            df_t = data_map.get(ticker)
            if df_t is None or df_t.empty:
                return None
            try:
                # Quick volume pre-filter (murah, tanpa kalkulasi indikator)
                vol_t  = s(df_t["Volume"])
                adv20t = float(vol_t.iloc[-21:-1].mean())
                if adv20t > 0 and float(vol_t.iloc[-1]) / adv20t < min_vol_rel * 0.4:
                    return None

                h = analyse(ticker, days, df=df_t, mr=mr)
                if not h:
                    return None

                # Pakai harga dari price_map (close terakhir) — tidak ada HTTP
                if ticker in price_map:
                    h["harga"] = price_map[ticker]

                # Filter logis cepat
                if not (h["di_atas"] and h["tk_kj"]):
                    return None
                if price_too_far_above_entry(h):
                    return None
                if h["vol_rel"] < min_vol_rel:
                    return None
                if require_ha and not h["ha_green"]:
                    return None

                # Health check
                if use_health:
                    hc = health_check(df_t, ticker,
                                      min_price=min_price, min_adv_juta=min_adv_juta)
                    if not hc.lolos:
                        return None

                sig = build_plan(h)
                sig = apply_regime_to_plan(sig, mr)
                if "BUY" not in sig.sinyal or not sig.sizing_pct:
                    return None
                if sig.confidence < min_conf:
                    return None

                return (ticker, h, sig)

            except Exception as e:
                errors.append(f"{ticker}: {e}")
                return None

        # Jalankan semua worker paralel
        done_count  = 0
        total_valid = sum(1 for t in daftar_saham
                         if data_map.get(t) is not None and not data_map[t].empty)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_analyse_worker, t): t for t in daftar_saham
                    if data_map.get(t) is not None and not data_map[t].empty}

            for fut in concurrent.futures.as_completed(futs):
                done_count += 1
                pbar.progress(
                    0.5 + done_count / max(total_valid, 1) / 2,
                    text=f"🧠 {done_count}/{total_valid} — {len(st.session_state.screener_hasil)} lolos"
                )
                try:
                    result = fut.result()
                except Exception as e:
                    errors.append(str(e))
                    continue

                if result is None:
                    continue

                ticker_r, h, sig = result
                st.session_state.screener_raw[ticker_r] = (h, sig)
                st.session_state.screener_hasil.append({
                    "Ticker":   ticker_r,
                    "Sinyal":   sig.sinyal,
                    "Conf.":    sig.confidence,
                    "Harga":    int(round(h["harga"])),
                    "Entry":    f"{int(round(h['lvl']['entry_bawah']))}–{int(round(h['lvl']['entry_atas']))}",
                    "Jarak Entry": f"{(h['harga'] / h['lvl']['entry_atas'] - 1) * 100:+.1f}%",
                    "Target 1": int(round(h["lvl"]["target_1"])),
                    "Cutloss":  int(round(h["lvl"]["cutloss"])),
                    "R/R":      f"1:{sig.rr:.1f}" if sig.rr else "─",
                    "MACD":     "🚀" if h["macd_cross_up"] else ("✅" if h["macd_bull"] else "❌"),
                    "ADX":      f"{h['adx']:.0f}",
                    "RSI":      f"{h['rsi']:.0f}",
                    "HA":       f"🟢{h['ha_seq']}c" if h["ha_green"] else f"🔴{h.get('ha_red_seq', h['ha_seq'])}c",
                    "Vol":      f"{h['vol_rel']:.1f}×",
                })

                # Update tabel setiap 5 hasil baru
                n = len(st.session_state.screener_hasil)
                if n % 5 == 0 or n == 1:
                    df_tmp = (pd.DataFrame(st.session_state.screener_hasil)
                              .sort_values("Conf.", ascending=False)
                              .reset_index(drop=True))
                    holder.dataframe(df_tmp, hide_index=True, use_container_width=True)

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
                render_charts(sel, h, ctx="screener")

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
                except Exception: data_map_eb[t] = pd.DataFrame()
                done += 1
                pbar.progress(done / total / 2, text=f"⚡ Download: {done}/{total}")

        # Fase 2: scan early bird — PARALEL
        status.markdown("🐦 Scanning early signals paralel…")

        # Price fallback dari close terakhir
        eb_price_map = {}
        for tk, df_tk in data_map_eb.items():
            if df_tk is not None and not df_tk.empty:
                try: eb_price_map[tk] = float(s(df_tk["Close"]).iloc[-1])
                except Exception: pass

        def _eb_worker(ticker: str) -> Optional[tuple]:
            df_t = data_map_eb.get(ticker)
            if df_t is None or df_t.empty: return None
            try:
                vol_t  = s(df_t["Volume"])
                adv20t = float(vol_t.iloc[-21:-1].mean())
                if adv20t > 0 and float(vol_t.iloc[-1]) / adv20t < 0.15: return None

                h = analyse(ticker, days, df=df_t, mr=mr)
                if not h: return None
                if ticker in eb_price_map: h["harga"] = eb_price_map[ticker]

                if h["di_atas"]: return None
                if price_too_far_above_entry(h): return None
                if h["early_score"] < min_eb_score: return None
                if h["dist_trap"]: return None
                if h["vol_rel"] < 0.3: return None

                if use_health:
                    hc = health_check(df_t, ticker, min_price=min_price, min_adv_juta=min_adv_juta)
                    if not hc.lolos: return None

                sig = build_plan(h)
                sig = apply_regime_to_plan(sig, mr)
                return (ticker, h, sig)
            except Exception as e:
                errors_eb.append(f"{ticker}: {e}")
                return None

        eb_done = 0
        eb_valid = sum(1 for t in daftar_saham
                       if data_map_eb.get(t) is not None and not data_map_eb[t].empty)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            eb_futs = {ex.submit(_eb_worker, t): t for t in daftar_saham
                       if data_map_eb.get(t) is not None and not data_map_eb[t].empty}

            for fut in concurrent.futures.as_completed(eb_futs):
                eb_done += 1
                pbar.progress(
                    0.5 + eb_done / max(eb_valid, 1) / 2,
                    text=f"🐦 {eb_done}/{eb_valid} — {len(st.session_state.eb_hasil)} early bird"
                )
                try:
                    result = fut.result()
                except Exception: continue
                if result is None: continue

                ticker_r, h, sig = result
                st.session_state.eb_raw[ticker_r] = (h, sig)

                eb_signals = []
                if h["kumo_twist_recent"]: eb_signals.append("🌀Twist")
                if h["kumo_narrowing"]:    eb_signals.append("📉Sempit")
                if h["price_near_cloud"]:  eb_signals.append("🔜Dekat")
                if h["tk_cross_new"]:      eb_signals.append("⚡TKcross")
                if h["chikou_near_free"]:  eb_signals.append("👁️Chikou")

                st.session_state.eb_hasil.append({
                    "Ticker":      ticker_r,
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

                n = len(st.session_state.eb_hasil)
                if n % 5 == 0 or n == 1:
                    df_tmp = (pd.DataFrame(st.session_state.eb_hasil)
                              .sort_values("EB Score", ascending=False))
                    holder.dataframe(df_tmp, hide_index=True, use_container_width=True)

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
            render_charts(eb_sel, h, ctx="earlybird")

            st.divider()
            st.markdown("#### 📏 Price Ladder")
            render_price_ladder(h, sig)

            if st.button("✖ Tutup", key="eb_close"):
                st.session_state.eb_sel = None; st.rerun()


# ══════════════════════════════════════════════════════════════
# MODE 5: RS HUNTER — saham kuat saat IHSG merah
# ══════════════════════════════════════════════════════════════
elif "RS Hunter" in mode:
    st.markdown("## 💪 RS Hunter — Saham Kuat Saat IHSG Merah")


    st.markdown(
        "<span style='color:#8b949e;font-size:0.85rem;'>"
        "Cari saham yang outperform IHSG — kandidat terbang duluan saat market rebound. "
        "Cocok dijalankan saat IHSG sedang merah/turun."
        "</span>", unsafe_allow_html=True
    )

    with st.expander("📖 Logika RS Hunter", expanded=False):
        st.markdown("""
**4 metrik yang diukur:**
- **RS Score** — composite score 0-100 dari RS 5d/20d/60d + price relative + volume divergence
- **Beta 60d** — korelasi saham vs IHSG. Beta rendah (<0.7) = tidak mudah terseret turun
- **Volume Divergence** — volume naik saat IHSG turun = institusi diam-diam akumulasi
- **Price Relative Trend** — rasio saham/IHSG sedang naik dalam 5 hari

**Cara pakai:**
1. Jalankan saat IHSG merah/sedang downtrend
2. Fokus pada RS Score ≥ 60 + vol divergence = TRUE
3. Saham-saham ini yang akan naik duluan dan paling kencang saat IHSG rebound
4. Kombinasikan dengan sinyal teknikal (Ichimoku) untuk timing entry
""")

    if "rs_hasil" not in st.session_state: st.session_state.rs_hasil = []
    if "rs_raw"   not in st.session_state: st.session_state.rs_raw   = {}
    if "rs_sel"   not in st.session_state: st.session_state.rs_sel   = None

    col_b1, col_b2, col_b3 = st.columns([2, 1, 1])
    with col_b1:
        min_rs = st.slider("Min. RS Score", 0, 100, 40, 5, key="rs_min_score")
    with col_b2:
        run_rs = st.button("💪 Mulai RS Hunt", type="primary", use_container_width=True)
    with col_b3:
        if st.button("🔄 Reset", key="rs_reset", use_container_width=True):
            st.session_state.rs_hasil = []; st.session_state.rs_raw = {}
            st.session_state.rs_sel = None; st.rerun()

    if run_rs:
        st.session_state.rs_hasil = []; st.session_state.rs_raw = {}; st.session_state.rs_sel = None
        errors_rs: list[str] = []

        pbar   = st.progress(0, text="Download paralel…")
        status = st.empty(); holder = st.empty()
        total  = len(daftar_saham); t0 = time.time()

        # Fase 1: Download paralel
        status.markdown("⚡ Download paralel…")
        data_map_rs: dict[str, pd.DataFrame] = {}
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            fut = {ex.submit(download_one, t): t for t in daftar_saham}
            for f in concurrent.futures.as_completed(fut):
                t = fut[f]
                try:    data_map_rs[t] = f.result()
                except Exception: data_map_rs[t] = pd.DataFrame()
                done += 1
                pbar.progress(done/total/2, text=f"⚡ Download: {done}/{total}")

        # Price fallback
        rs_price_map = {}
        for tk, df_tk in data_map_rs.items():
            if df_tk is not None and not df_tk.empty:
                try: rs_price_map[tk] = float(s(df_tk["Close"]).iloc[-1])
                except Exception: pass

        # Fase 2: RS scoring paralel
        status.markdown("💪 RS scoring paralel…")

        def _rs_worker(ticker: str) -> Optional[tuple]:
            df_t = data_map_rs.get(ticker)
            if df_t is None or df_t.empty: return None
            try:
                # Quick health check
                if use_health:
                    hc = health_check(df_t, ticker, min_price=min_price, min_adv_juta=min_adv_juta)
                    if not hc.lolos: return None

                h = analyse(ticker, days, df=df_t, mr=mr)
                if not h: return None
                if ticker in rs_price_map: h["harga"] = rs_price_map[ticker]
                if price_too_far_above_entry(h): return None
                if h["dist_trap"]: return None      # distribusi aktif, skip
                if h.get("rs_score", 0) < min_rs: return None

                sig = build_plan(h)
                sig = apply_regime_to_plan(sig, mr)
                return (ticker, h, sig)
            except Exception as e:
                errors_rs.append(f"{ticker}: {e}")
                return None

        rs_done  = 0
        rs_valid = sum(1 for t in daftar_saham
                       if data_map_rs.get(t) is not None and not data_map_rs[t].empty)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            rs_futs = {ex.submit(_rs_worker, t): t for t in daftar_saham
                       if data_map_rs.get(t) is not None and not data_map_rs[t].empty}

            for fut in concurrent.futures.as_completed(rs_futs):
                rs_done += 1
                pbar.progress(0.5 + rs_done/max(rs_valid,1)/2,
                              text=f"💪 {rs_done}/{rs_valid} — {len(st.session_state.rs_hasil)} RS kuat")
                try:
                    result = fut.result()
                except Exception: continue
                if result is None: continue

                ticker_r, h, sig = result
                st.session_state.rs_raw[ticker_r] = (h, sig)
                st.session_state.rs_hasil.append({
                    "Ticker":    ticker_r,
                    "RS Score":  h.get("rs_score", 0),
                    "RS 5d":     f"{h.get('rs_5d',0):+.1f}%",
                    "RS 20d":    f"{h.get('rs_20d',0):+.1f}%",
                    "RS 60d":    f"{h.get('rs_60d',0):+.1f}%",
                    "Beta":      f"{h.get('beta_60',1):.2f}",
                    "Vol Div":   "✅" if h.get("vol_divergence") else "—",
                    "PR Trend":  "📈" if h.get("price_relative_trend") else "—",
                    "Sinyal":    sig.sinyal,
                    "Conf.":     sig.confidence,
                    "Harga":     int(round(h["harga"])),
                    "HA":        f"{'🟢' if h['ha_green'] else '🔴'}{h['ha_seq']}c",
                    "IHSG Regime": mr.regime,
                })

                n = len(st.session_state.rs_hasil)
                if n % 5 == 0 or n == 1:
                    df_tmp = (pd.DataFrame(st.session_state.rs_hasil)
                              .sort_values("RS Score", ascending=False).reset_index(drop=True))
                    holder.dataframe(df_tmp, hide_index=True, use_container_width=True)

        elapsed = time.time() - t0
        pbar.progress(1.0, text=f"✅ Selesai {elapsed:.1f}s")
        status.empty(); holder.empty()

        if errors_rs:
            with st.expander(f"⚠️ {len(errors_rs)} error"):
                for e in errors_rs: st.text(e)
        st.rerun()

    # ── Render hasil RS Hunter ──
    if st.session_state.rs_hasil:
        df_rs = (pd.DataFrame(st.session_state.rs_hasil)
                 .sort_values("RS Score", ascending=False).reset_index(drop=True))

        n_vd  = df_rs["Vol Div"].eq("✅").sum()
        n_prt = df_rs["PR Trend"].eq("📈").sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total kandidat", len(df_rs))
        m2.metric("Vol Divergence", n_vd, delta="akumulasi tersembunyi")
        m3.metric("PR Trend naik",  n_prt)
        m4.metric("Avg RS Score",   f"{df_rs['RS Score'].mean():.0f}/100")

        if mr.regime == "BEAR":
            st.error("🚨 Market BEAR — ini kondisi TERBAIK untuk RS Hunter. "
                     "Saham di list ini yang paling mungkin terbang saat sentiment berbalik.")
        elif mr.regime == "NEUTRAL":
            st.warning("⚠️ Market sideways — RS Hunter tetap valid, fokus RS Score ≥70 + Vol Divergence.")
        else:
            st.info("ℹ️ Market sedang BULL — RS Hunter masih berguna untuk pilih yang terkuat.")

        st.divider()
        st.dataframe(df_rs, hide_index=True, use_container_width=True)
        csv_rs = df_rs.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV RS Hunter", csv_rs, "rs_hunter.csv", "text/csv")

        st.divider()
        st.markdown("#### 🔍 Klik untuk full report:")
        tickers_rs = df_rs["Ticker"].tolist()
        for row_tks in [tickers_rs[i:i+6] for i in range(0, len(tickers_rs), 6)]:
            cols = st.columns(6)
            for j, tk in enumerate(row_tks):
                if tk not in st.session_state.rs_raw: continue
                is_act = st.session_state.rs_sel == tk
                if cols[j].button(f"{'▶ ' if is_act else ''}{tk}",
                                  key=f"rs_btn_{tk}", use_container_width=True,
                                  type="primary" if is_act else "secondary"):
                    st.session_state.rs_sel = None if is_act else tk
                    st.rerun()

        rs_sel = st.session_state.rs_sel
        if rs_sel and rs_sel in st.session_state.rs_raw:
            h, sig = st.session_state.rs_raw[rs_sel]
            st.divider()
            st.markdown(f"## 💪 RS Report: `{rs_sel}`")

            # RS scorecard
            rs_cols = st.columns(6)
            rs_cols[0].metric("RS Score",  f"{h.get('rs_score',0)}/100")
            rs_cols[1].metric("RS 5d",     f"{h.get('rs_5d',0):+.1f}%")
            rs_cols[2].metric("RS 20d",    f"{h.get('rs_20d',0):+.1f}%")
            rs_cols[3].metric("RS 60d",    f"{h.get('rs_60d',0):+.1f}%")
            rs_cols[4].metric("Beta",      f"{h.get('beta_60',1):.2f}")
            rs_cols[5].metric("Vol Div",   "✅" if h.get("vol_divergence") else "❌")

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
            render_charts(rs_sel, h, ctx="rshunter")

            if st.button("✖ Tutup", key="rs_close"):
                st.session_state.rs_sel = None; st.rerun()

    elif not run_rs:
        st.info(
            "Klik **💪 Mulai RS Hunt** untuk scan semua saham dan temukan yang outperform IHSG.\n\n"
            "**Best timing:** jalankan saat IHSG sedang merah 2–5 hari berturut-turut. "
            "Saham dengan RS Score tinggi + Volume Divergence = yang paling likely terbang saat sentiment berbalik."
        )
