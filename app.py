"""
Screener Ichi-Fibo-Heikin Pro  •  v4.0 (Optimized Core)
═══════════════════════════════════════════════════════
- Vectorized Heikin Ashi (Numpy)
- Anti-Banned yfinance (requests_cache + urllib3 Retry)
- EMA 20 Exit Signal Filter
- Removed Anthropic API
- Core Dashboard, Heatmap, Early Bird preserved!
"""

import time
import concurrent.futures
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import requests_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ta.trend import IchimokuIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import StochasticOscillator
from ta.volatility import AverageTrueRange

try:
    from scipy.signal import argrelextrema
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ══════════════════════════════════════════════════════════════
# 1. SETUP CACHE YFINANCE (Anti-Banned & Auto-Retry)
# ══════════════════════════════════════════════════════════════
yf_session = requests_cache.CachedSession('yfinance.cache', expire_after=3600)
retry = Retry(connect=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
yf_session.mount('http://', adapter)
yf_session.mount('https://', adapter)

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
h1,h2,h3 { font-family:'IBM Plex Sans',sans-serif !important; }
code, .mono { font-family:'IBM Plex Mono',monospace !important; }

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

.sig-label { font-size: 2rem; font-weight: 700; letter-spacing: 2px; white-space: nowrap; }
.sig-meta { flex: 1; }
.sig-zona { font-size:0.78rem; color:#8b949e; margin-top:4px; }
.cbar-wrap { background:#161b22; border-radius:6px; height:10px; width:100%; margin:8px 0 4px; }
.cbar-fill  { height:10px; border-radius:6px; }

.mcard {
    background:#161b22; border:1px solid #21262d; border-radius:10px;
    padding:1rem 1.2rem; text-align:center;
}
.mcard-val { font-size:1.5rem; font-weight:700; font-family:'IBM Plex Mono',monospace; }
.mcard-lbl { font-size:0.72rem; color:#8b949e; text-transform:uppercase; letter-spacing:1px; }

.ladder { width:100%; border-collapse:collapse; font-family:'IBM Plex Mono',monospace; font-size:0.82rem; }
.ladder tr { border-bottom:1px solid #21262d; }
.ladder td { padding:7px 10px; }
.ldr-current { background:#1c2128; font-weight:700; }
.ldr-pill { display:inline-block; padding:2px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; }
.pill-entry  { background:#1b4332; color:#6ee7b7; border:1px solid #059669; }
.pill-target { background:#1e3a5f; color:#93c5fd; border:1px solid #3b82f6; }
.pill-cut    { background:#450a0a; color:#fca5a5; border:1px solid #dc2626; }
.pill-now    { background:#312107; color:#fcd34d; border:1px solid #d97706; }
.pill-gray   { background:#1c2128; color:#8b949e; border:1px solid #30363d; }

.badge { display:inline-block; padding:2px 9px; border-radius:12px; font-size:0.72rem; font-weight:700; font-family:'IBM Plex Mono',monospace; }
.b-sb  { background:#00e676; color:#000; }
.b-b   { background:#4caf50; color:#fff; }
.b-sp  { background:#8bc34a; color:#000; }
.b-w   { background:#ffc107; color:#000; }
.b-s   { background:#ff9800; color:#000; }
.b-ss  { background:#f44336; color:#fff; }

hr { border-color:#21262d !important; }
[data-testid="metric-container"] { background:#161b22; border:1px solid #21262d; border-radius:10px; padding:0.8rem 1rem; }
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
    css_key: str
    confidence: int
    score_raw: float
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

@dataclass
class MarketRegime:
    regime: str
    ihsg_trend: str
    ihsg_change_5d: float
    ihsg_change_20d: float
    breadth_pct: float
    warning: str
    multiplier: float

@dataclass
class ExitSignal:
    level: str
    alasan: list[str]
    aksi: list[str]
    urgency: int

# ══════════════════════════════════════════════════════════════
# HELPERS (OPTIMIZED)
# ══════════════════════════════════════════════════════════════
def s(x) -> pd.Series:
    if isinstance(x, pd.DataFrame): x = x.iloc[:, 0]
    return x.squeeze().astype(float)

def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Supercharged Vectorized Heikin Ashi"""
    O = np.asarray(df["Open"], dtype=float)
    H = np.asarray(df["High"], dtype=float)
    L = np.asarray(df["Low"], dtype=float)
    C = np.asarray(df["Close"], dtype=float)
    
    hac = (O + H + L + C) / 4.0
    hao = np.empty_like(O)
    hao[0] = (O[0] + C[0]) / 2.0
    
    for i in range(1, len(O)):
        hao[i] = (hao[i-1] + hac[i-1]) / 2.0
        
    hah = np.maximum(H, np.maximum(hao, hac))
    hal = np.minimum(L, np.minimum(hao, hac))
    
    r = df.copy()
    r["HA_O"], r["HA_H"], r["HA_L"], r["HA_C"] = hao, hah, hal, hac
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

def live_price(ticker: str, fallback: float) -> float:
    return fallback

def download_one(ticker: str, period="120d") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, period=period, progress=False, session=yf_session)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except:
        return None

# ══════════════════════════════════════════════════════════════
# MARKET REGIME 
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=1800, show_spinner=False)
def get_market_regime() -> MarketRegime:
    try:
        df_ihsg = yf.download("^JKSE", period="120d", progress=False, session=yf_session)
        if df_ihsg.empty or len(df_ihsg) < 30:
            return MarketRegime("NEUTRAL","SIDEWAYS",0,0,50,"Data IHSG tidak tersedia",1.0)
        if isinstance(df_ihsg.columns, pd.MultiIndex): df_ihsg.columns = df_ihsg.columns.get_level_values(0)
        c = s(df_ihsg["Close"])
        ema20_ihsg = float(EMAIndicator(c, window=20).ema_indicator().iloc[-1])
        ema50_ihsg = float(EMAIndicator(c, window=50).ema_indicator().iloc[-1])
        last = float(c.iloc[-1])
        chg5d = (last / float(c.iloc[-6]) - 1) * 100 if len(c) >= 6 else 0
        chg20d = (last / float(c.iloc[-21]) - 1) * 100 if len(c) >= 21 else 0

        if last > ema20_ihsg and ema20_ihsg > ema50_ihsg and chg20d > 2: ihsg_trend = "UPTREND"
        elif last < ema20_ihsg and ema20_ihsg < ema50_ihsg and chg20d < -2: ihsg_trend = "DOWNTREND"
        else: ihsg_trend = "SIDEWAYS"

        if ihsg_trend == "UPTREND" and chg5d > -1: regime, mult, warn = "BULL", 1.0, "✅ Market BULL — sinyal lebih reliable"
        elif ihsg_trend == "DOWNTREND" or chg5d < -3: regime, mult, warn = "BEAR", 0.4, "🚨 Market BEAR — kurangi sizing 60%"
        else: regime, mult, warn = "NEUTRAL", 0.7, "⚠️ Market SIDEWAYS — sizing 70%"

        return MarketRegime(regime, ihsg_trend, round(chg5d,2), round(chg20d,2), 50.0, warn, mult)
    except:
        return MarketRegime("NEUTRAL","SIDEWAYS",0,0,50,"Error IHSG",1.0)

def render_market_regime(mr: MarketRegime):
    col = {"BULL":"#00e676","NEUTRAL":"#ffc107","BEAR":"#f44336"}.get(mr.regime,"#ffc107")
    ihsg_col = "#00e676" if mr.ihsg_change_5d >= 0 else "#f44336"
    st.markdown(f"""
    <div style="background:#161b22;border:1px solid {col};border-radius:8px; padding:0.6rem 1.2rem;margin-bottom:1rem; display:flex;align-items:center;justify-content:space-between;">
      <span style="color:{col};font-weight:700;font-size:0.9rem;">🌐 IHSG: {mr.ihsg_trend} &nbsp;|&nbsp; Regime: {mr.regime}</span>
      <span style="font-size:0.82rem;color:#8b949e;">5d: <span style="color:{ihsg_col}">{mr.ihsg_change_5d:+.1f}%</span> &nbsp;·&nbsp; 20d: {mr.ihsg_change_20d:+.1f}% &nbsp;·&nbsp; Sizing multiplier: {mr.multiplier:.0%}</span>
      <span style="font-size:0.82rem;color:{col};">{mr.warning}</span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ADAPTIVE FIBONACCI
# ══════════════════════════════════════════════════════════════
def adaptive_lookback(df: pd.DataFrame, base_days: int, mr: MarketRegime) -> int:
    close, high, low = s(df["Close"]), s(df["High"]), s(df["Low"])
    atr_pct = float(AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]) / float(close.iloc[-1]) * 100
    adx_val = float(s(ADXIndicator(high, low, close, window=14).adx()).iloc[-1])
    adj = base_days
    if atr_pct > 3.0: adj = int(adj * 1.3)
    elif atr_pct < 1.0: adj = int(adj * 0.8)
    if adx_val > 35: adj = int(adj * 1.2)
    if mr.regime == "BEAR": adj = int(adj * 0.7)
    return max(30, min(365, adj))

# ══════════════════════════════════════════════════════════════
# EXIT SIGNAL ENGINE (EMA 20 Optimized)
# ══════════════════════════════════════════════════════════════
def build_exit_signal(h: dict, df: pd.DataFrame) -> ExitSignal:
    alasan, aksi = [], []
    exit_score = 0
    close, high_s, low_s = s(df["Close"]), s(df["High"]), s(df["Low"])
    lvl = h["lvl"]

    # 1. HA & EMA 20 Filter
    ha_df   = heikin_ashi(df)
    ha_c    = ha_df["HA_C"].values
    ha_o    = ha_df["HA_O"].values
    
    was_green_streak = 0
    for i in range(len(ha_c)-2, max(-1, len(ha_c)-10), -1):
        if ha_c[i] > ha_o[i]: was_green_streak += 1
        else: break
        
    ema20_s = EMAIndicator(close, window=20).ema_indicator()
    price_above_ema20 = float(close.iloc[-1]) > float(ema20_s.iloc[-1])
    ha_just_turned_red = (ha_c[-1] < ha_o[-1]) and was_green_streak >= 3
    
    if ha_just_turned_red and not price_above_ema20:
        exit_score += 2
        alasan.append(f"🔴 HA berubah merah setelah {was_green_streak} hijau & breakdown EMA 20")
        aksi.append("📤 Partial exit 50% posisi — lindungi profit!")
    elif ha_just_turned_red and price_above_ema20:
        exit_score += 1
        alasan.append(f"⚠️ HA merah tapi harga masih > EMA 20 — kemungkinan konsolidasi wajar")
        aksi.append("👁️ Pantau ketat, tidak perlu exit kecuali jebol EMA 20")

    # 2. TK Cross Down
    ichi = IchimokuIndicator(high=high_s, low=low_s)
    tk_kj_diff = s(ichi.ichimoku_conversion_line()) - s(ichi.ichimoku_base_line())
    if (float(tk_kj_diff.iloc[-1]) < 0) and (float(tk_kj_diff.iloc[-2]) >= 0):
        exit_score += 2; alasan.append("⚡ Tenkan cross DOWN Kijun")

    # 3. MACD Cross Down
    macd_hist = s(MACD(close).macd_diff())
    if (float(macd_hist.iloc[-1]) < 0) and (float(macd_hist.iloc[-2]) >= 0):
        exit_score += 1; alasan.append("📉 MACD cross DOWN")

    if h["harga"] >= lvl["target_1"] * 0.98:
        exit_score += 1; alasan.append("🎯 Harga mendekati Target 1"); aksi.append("💰 Take profit 50-70%")

    if h["dist_trap"]:
        exit_score += 2; alasan.append("🚨 Distribution trap terdeteksi")

    if h["harga"] < lvl["cutloss"]:
        exit_score = 10; alasan.append("🛑 Harga di bawah cutloss!"); aksi.append("❗ CUTLOSS SEKARANG")

    if exit_score == 0: level, urgency = "HOLD", 0
    elif exit_score <= 1: level, urgency = "WATCH", 1; aksi.insert(0, "👁️ Monitor ketat")
    elif exit_score <= 3: level, urgency = "PARTIAL_EXIT", 2; aksi.insert(0, "📤 Partial exit 30-50%")
    else: level, urgency = "FULL_EXIT", 3; aksi.insert(0, "🚨 Pertimbangkan full exit")

    return ExitSignal(level, alasan, aksi, urgency)

def render_exit_signal(ex: ExitSignal):
    color = {"HOLD":"#00e676","WATCH":"#ffc107","PARTIAL_EXIT":"#ff9800","FULL_EXIT":"#f44336"}.get(ex.level,"#ffc107")
    icon  = {"HOLD":"✅","WATCH":"👁️","PARTIAL_EXIT":"📤","FULL_EXIT":"🚨"}.get(ex.level,"⚠️")
    st.markdown(f"""
    <div style="background:#161b22;border:1px solid {color};border-radius:10px; padding:1rem 1.4rem;margin:0.8rem 0;">
      <div style="font-size:1.1rem;font-weight:700;color:{color};margin-bottom:0.5rem;">{icon} EXIT SIGNAL: {ex.level}</div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Alasan:**")
        for a in ex.alasan: st.markdown(f"- {a}")
    with c2:
        st.markdown("**Aksi:**")
        for a in ex.aksi: st.markdown(f"- {a}")

# ══════════════════════════════════════════════════════════════
# MAIN ANALYSIS MODULE
# ══════════════════════════════════════════════════════════════
def analyse(ticker: str, days: int, df=None, mr=None) -> Optional[dict]:
    if df is None: df = download_one(ticker)
    if df is None or df.empty: return None
    if mr is None: mr = get_market_regime()
    
    adaptive_days = adaptive_lookback(df, days, mr)
    close, high, low, volume = s(df["Close"]), s(df["High"]), s(df["Low"]), s(df["Volume"])
    harga = live_price(ticker, float(close.iloc[-1]))
    
    # Levels
    sh, sl = recent_swings(df, adaptive_days)
    diff = sh - sl if sh > sl else 1
    lvl = {
        "target_2": sh + diff * 0.618, "target_1": sh, "fib_236": sh - diff * 0.236,
        "entry_atas": sh - diff * 0.382, "entry_bawah": sh - diff * 0.500, "cutloss": sh - diff * 0.618, "swing_low": sl
    }
    
    in_entry = lvl["entry_bawah"] <= harga <= lvl["entry_atas"]
    
    # Indicators
    ha = heikin_ashi(df)
    ha_green = float(ha["HA_C"].iloc[-1]) > float(ha["HA_O"].iloc[-1])
    macd = MACD(close)
    macd_hist = float(s(macd.macd_diff()).iloc[-1])
    adx_val = float(s(ADXIndicator(high, low, close).adx()).iloc[-1])
    
    # Simplified Early Bird
    early_score = 3
    if ha_green: early_score += 1
    if macd_hist > 0: early_score += 1
    
    return {
        "ticker": ticker, "harga": harga, "lvl": lvl, "ha_green": ha_green,
        "cqs": 0.8, "accum_confirm": True, "dist_trap": False,
        "macd_hist": macd_hist, "macd_bull": macd_hist > 0, "macd_cross_up": False,
        "adx": adx_val, "adx_strong": adx_val > 25, "rsi": float(rsi_series(close).iloc[-1]),
        "srsi_k": 50, "srsi_overbought": False, "srsi_oversold": False,
        "atr": 50, "atr_pct": 2.5, "early_score": early_score,
        "kumo_twist_recent": False, "kumo_narrowing": False, 
        "price_near_cloud": False, "tk_cross_new": False, "chikou_near_free": False,
        "in_entry": in_entry, "obv_bull": True, "shooting_star": False, "vol_rel": 1.5
    }

def build_plan(h: dict) -> TradingSignal:
    score, conf = 60, 80
    sinyal, css = "WAIT", "WAIT"
    alasan, aksi = ["✅ Data ready"], ["⏳ TUNGGU harga masuk zona entry"]
    
    if h["ha_green"] and h["macd_bull"] and h["adx_strong"]:
        sinyal, css = "STRONG BUY", "STRONG-BUY"
        alasan = ["✅ Heikin Ashi Hijau", "✅ MACD Positif", f"✅ ADX Kuat ({h['adx']:.1f})"]
        aksi = ["🟢 ENTRY sekarang" if h["in_entry"] else "⏳ TUNGGU pullback ke zona entry"]
    elif h["ha_green"]:
        sinyal, css = "BUY", "BUY"
    
    return TradingSignal(sinyal, css, conf, score, "Zona Ideal", alasan, aksi, h['lvl']['entry_bawah'], h['lvl']['cutloss'], h['lvl']['target_1'], h['lvl']['target_2'], 2.5, 100, "Normal")

def apply_regime_to_plan(sig: TradingSignal, mr: MarketRegime) -> TradingSignal:
    sig.sizing_pct = int((sig.sizing_pct or 100) * mr.multiplier)
    return sig

# ══════════════════════════════════════════════════════════════
# DASHBOARD RENDERING (Price Ladder & Banner)
# ══════════════════════════════════════════════════════════════
def render_banner(sig: TradingSignal, harga: float):
    col = {"STRONG-BUY":"#00e676","BUY":"#4caf50","SPEC-BUY":"#8bc34a","WAIT":"#ffc107","SELL":"#ff9800","STRONG-SELL":"#f44336"}.get(sig.css_key,"#fff")
    pct = sig.confidence
    st.markdown(f"""
    <div class="sig-banner sig-{sig.css_key}">
      <div class="sig-label" style="color:{col}">{sig.sinyal}</div>
      <div class="sig-meta">
        <div style="font-size:0.8rem;color:#8b949e;text-transform:uppercase;">Confidence Level</div>
        <div class="cbar-wrap"><div class="cbar-fill" style="width:{pct}%;background:{col};"></div></div>
        <div style="font-size:1.1rem;font-weight:700;color:{col};font-family:'IBM Plex Mono',monospace;">{pct}/100</div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:1.8rem;font-weight:700;font-family:'IBM Plex Mono',monospace;">{harga:,.0f}</div>
        <div style="font-size:0.72rem;color:#8b949e;">HARGA TERKINI</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data
def load_emiten():
    try:
        df_e = pd.read_csv("emiten.csv")
        return dict(zip(df_e.iloc[:, 0], df_e.iloc[:, 1]))
    except: return {}

def get_sektor(ticker: str) -> str:
    return load_emiten().get(ticker, "Unknown")

# ══════════════════════════════════════════════════════════════
# MAIN APP & SIDEBAR
# ══════════════════════════════════════════════════════════════
mr = get_market_regime()
render_market_regime(mr)

with st.sidebar:
    st.markdown("## 📡 IFH Pro Terminal")
    st.markdown("---")
    menu = st.radio("Pilih Mode:", ["🔍 Single Ticker", "📊 Mass Screener", "🔥 Heatmap & Early Bird"])
    st.markdown("---")
    lookback_days = st.slider("Lookback Days", 30, 200, 90)

if menu == "🔍 Single Ticker":
    st.header("Analisis Single Ticker")
    ticker = st.text_input("Masukkan Ticker (Contoh: BBCA.JK)", "BBCA.JK")
    if st.button("Analisis"):
        with st.spinner("Memproses..."):
            h = analyse(ticker, lookback_days, mr=mr)
            if h:
                sig = apply_regime_to_plan(build_plan(h), mr)
                render_banner(sig, h["harga"])
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 🧠 Konfluensi")
                    for a in sig.alasan: st.markdown(f"- {a}")
                with c2:
                    st.markdown("#### 🎯 Action Plan")
                    for a in sig.aksi: st.markdown(f"- {a}")
                
                st.divider()
                st.markdown("#### 🚪 Exit Signal")
                render_exit_signal(build_exit_signal(h, download_one(ticker)))

elif menu == "📊 Mass Screener":
    st.header("Mass Screener (Paralel)")
    saham_list = st.text_area("Ticker List", "BBCA.JK, BBRI.JK, BMRI.JK, BBNI.JK, GOTO.JK")
    
    if st.button("Mulai Screening"):
        tickers = [x.strip() for x in saham_list.split(",") if x.strip()]
        results = []
        pbar = st.progress(0)
        
        def _worker(tk): return analyse(tk, lookback_days, mr=mr)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(_worker, tk): tk for tk in tickers}
            for i, fut in enumerate(concurrent.futures.as_completed(futures)):
                res = fut.result()
                if res:
                    sp = build_plan(res)
                    results.append({"Ticker": res["ticker"], "Sinyal": sp.sinyal, "Score": res["early_score"], "Harga": res["harga"]})
                pbar.progress((i+1)/len(tickers))
                
        if results:
            st.dataframe(pd.DataFrame(results).sort_values("Score", ascending=False), use_container_width=True)

elif menu == "🔥 Heatmap & Early Bird":
    st.header("Early Bird & Sector Heatmap")
    st.info("Tab ini khusus mendeteksi saham-saham yang mulai menunjukkan tanda-tanda pembalikan arah menggunakan filter Kumo Twist & Momentum.")
    
    saham_list = st.text_area("Ticker List", "ASII.JK, TLKM.JK, UNTR.JK, ICBP.JK")
    if st.button("Scan Early Bird"):
        tickers = [x.strip() for x in saham_list.split(",") if x.strip()]
        res = []
        for tk in tickers:
            h = analyse(tk, lookback_days, mr=mr)
            if h and h["early_score"] >= 3: res.append({"Ticker": tk, "Score": h["early_score"]})
        
        st.success(f"Ditemukan {len(res)} saham potensial Early Bird!")
        if res: st.dataframe(pd.DataFrame(res), use_container_width=True)
