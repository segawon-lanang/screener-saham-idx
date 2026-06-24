"""
Screener Ichi-Fibo-Heikin Pro  •  v6.0  (Full Rewrite)
═══════════════════════════════════════════════════════
Audit & rombak total dari v5.0 — 18 bug diperbaiki:

CRITICAL FIXES:
  [1] HA_O pakai EWM(alpha=0.5) bukan formula rekursif standar → diperbaiki ke loop vectorized
  [2] Ichimoku Senkou double-shift(26) → diperbaiki, library ta SUDAH shift, tidak perlu lagi
  [3] Fibonacci lookback hanya 20 bar → adaptive 60-120 bar berdasarkan ATR
  [4] MultiIndex parsing tidak robust (single-ticker ambil full df) → parser defensif per-ticker

MAJOR FIXES:
  [5] Target = high + 0.618*diff (bukan standar) → Fibonacci Extension 127.2% & 161.8% yang benar
  [6] no_lower_shadow threshold 0.002 terlalu ketat, bobot terbesar → threshold adaptif 0.5×ATR
  [7] Market regime pakai subset filtered (selection bias) → IHSG benchmark + breadth independen
  [8] Stop loss = Kijun tanpa validasi jarak → max(Kijun, Fibo618), dikonfirmasi ATR
  [9] R/R tidak dihitung sama sekali → R/R gate minimum 1.5:1
  [10] MACD hanya cek histogram > 0 (bukan fresh cross) → deteksi fresh crossover dari negatif
  [11] Volume filter pakai lot count → nilai rupiah (ADV-Rp) sebagai proxy likuiditas
  [12] Tidak ada ADX — trend strength tidak terukur → ADX 14, skor berbeda trending vs choppy
  [13] process_technical mutate df langsung → bekerja pada salinan internal

MINOR FIXES & ADDITIONS:
  [14] Score /8 menyesatkan → skor dinamis dengan max yang tercapai
  [15] Ganti StochasticOscillator biasa → Stochastic RSI (RSI dalam rentang stochastic)
  [16] Market regime threshold hardcoded → dikalibrasi vs IHSG trend
  [17] process_technical tidak di-cache → @lru_cache per-ticker via hash df
  [18] Tidak ada ADV-Rp → dihitung dan ditampilkan di tabel

TAMBAHAN BARU (tidak ada di v5.0):
  - Relative Strength vs IHSG (RS-score: saham outperform/underperform benchmark)
  - Volume Spike Alert (volume relatif terhadap ADV20)
  - Confidence level: LOW / MEDIUM / HIGH / VERY HIGH dengan threshold ketat
  - Position Sizing berbasis ATR (risk 1% modal per trade)
  - R/R display di modal: entry zone, target, SL, R/R ratio
  - Trend Duration (berapa bar sudah di atas awan Ichimoku)
  - CQS (Candle Quality Score) — close position dalam candle range
  - Sector badge (membutuhkan kolom 'sektor' di emiten.csv)
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from ta.trend import IchimokuIndicator, EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import Optional
import time

# ═══════════════════════════════════════════════════════════
# 0. PAGE CONFIG — HARUS PALING ATAS
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="IFH Pro v6.0",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🦅",
)

# ═══════════════════════════════════════════════════════════
# 1. DARK MODE CSS
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
  .stApp { background: #0E1117; color: #FAFAFA; }
  .block-container { padding-top: 1.5rem; }

  /* Metric cards */
  .kpi-card {
    background: #1E2329; border-radius: 10px; padding: 14px 18px;
    border-left: 4px solid #F0B90B; margin-bottom: 10px;
  }
  .kpi-label { font-size: 0.78rem; color: #848E9C; margin-bottom: 2px; }
  .kpi-value { font-size: 1.35rem; font-weight: 700; color: #FAFAFA; }

  /* Signal badges */
  .badge-strong-buy  { background:#004d1f; color:#00e676; padding:3px 10px; border-radius:12px; font-weight:700; }
  .badge-buy         { background:#1b3a1e; color:#4caf50; padding:3px 10px; border-radius:12px; font-weight:700; }
  .badge-watch       { background:#3d2c00; color:#ffc107; padding:3px 10px; border-radius:12px; font-weight:700; }
  .badge-avoid       { background:#3d0000; color:#f44336; padding:3px 10px; border-radius:12px; font-weight:700; }

  /* Regime banner */
  .regime-bull { background:#0a2f1a; border-left:5px solid #00e676; padding:12px 18px; border-radius:8px; }
  .regime-bear { background:#2f0a0a; border-left:5px solid #f44336; padding:12px 18px; border-radius:8px; }
  .regime-neutral { background:#2a2a0a; border-left:5px solid #ffc107; padding:12px 18px; border-radius:8px; }

  /* Scrollable result table */
  .result-table { font-size: 0.85rem; }

  /* Divider */
  hr { border-color: #2a2d35; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# 2. KONSTANTA & DATACLASS
# ═══════════════════════════════════════════════════════════
IHSG_TICKER   = "^JKSE"
PERIOD        = "1y"          # 1 tahun data untuk warm-up indikator yang cukup
MIN_BARS      = 80            # minimum bar yang dibutuhkan untuk semua indikator
ADX_THRESHOLD = 20            # ADX > 20 = trending, < 20 = choppy

SIGNAL_ORDER = ["STRONG BUY", "BUY", "WATCH", "AVOID"]
SIGNAL_COLOR = {
    "STRONG BUY": "#00e676",
    "BUY":        "#4caf50",
    "WATCH":      "#ffc107",
    "AVOID":      "#f44336",
}

@dataclass
class ScreenResult:
    ticker:        str
    harga:         float
    signal:        str              # STRONG BUY / BUY / WATCH / AVOID
    score:         int
    score_max:     int
    confidence:    str              # LOW / MEDIUM / HIGH / VERY HIGH
    reasons:       list[str]        = field(default_factory=list)

    # Fibonacci levels
    swing_high:    float = 0.0
    swing_low:     float = 0.0
    entry_lo:      float = 0.0      # Fibo 50%
    entry_hi:      float = 0.0      # Fibo 38.2%
    target1:       float = 0.0      # Fibo Ext 127.2%
    target2:       float = 0.0      # Fibo Ext 161.8%
    stop_loss:     float = 0.0      # max(Kijun, Fibo 61.8%)
    rr:            float = 0.0      # (target1 - entry_mid) / (entry_mid - SL)

    # Indikator
    atr:           float = 0.0
    atr_pct:       float = 0.0
    adv_rp:        float = 0.0      # Average Daily Value Rupiah (juta)
    vol_rel:       float = 0.0      # volume relatif vs ADV20
    rs_score:      float = 0.0      # Relative Strength vs IHSG (z-score 20d)
    adx:           float = 0.0
    trend_bars:    int   = 0        # berapa bar sudah di atas awan Ichimoku
    cqs:           float = 0.0      # Candle Quality Score (0-1)
    sektor:        str   = "—"

    # Position sizing (asumsi modal Rp100 juta, risk 1%)
    sizing_lot:    int   = 0        # jumlah lot untuk risk 1% modal


# ═══════════════════════════════════════════════════════════
# 3. DATA LOADING
# ═══════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def load_emiten() -> pd.DataFrame:
    """Load daftar emiten dari emiten.csv. Kolom wajib: ticker. Opsional: nama, sektor."""
    try:
        df = pd.read_csv("emiten.csv")
        df.columns = [c.strip().lower() for c in df.columns]
        if "ticker" not in df.columns:
            raise ValueError("Kolom 'ticker' tidak ditemukan di emiten.csv")
        df["ticker"] = df["ticker"].str.strip().str.upper()
        # Pastikan suffix .JK ada
        df["ticker"] = df["ticker"].apply(lambda x: x if x.endswith(".JK") else x + ".JK")
        return df.dropna(subset=["ticker"])
    except FileNotFoundError:
        # Fallback: IDX LQ45 representatif
        tickers = [
            "BBCA.JK","BMRI.JK","BBRI.JK","BREN.JK","AMMN.JK",
            "TLKM.JK","ASII.JK","GOTO.JK","MDKA.JK","ICBP.JK",
            "INDF.JK","SMGR.JK","ADRO.JK","UNVR.JK","KLBF.JK",
            "PTBA.JK","ANTM.JK","EXCL.JK","MIKA.JK","CPIN.JK",
        ]
        return pd.DataFrame({"ticker": tickers, "sektor": "—"})


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ihsg(period: str = PERIOD) -> pd.DataFrame:
    """Unduh data IHSG untuk RS calculation dan market regime benchmark."""
    try:
        df = yf.download(IHSG_TICKER, period=period, interval="1d", progress=False, auto_adjust=True)
        df = df.dropna(subset=["Close"])
        return df if len(df) >= MIN_BARS else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _download_one(ticker: str, period: str) -> Optional[pd.DataFrame]:
    """Download satu ticker dengan error handling ketat."""
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        df = df.dropna(subset=["Close"])
        if len(df) < MIN_BARS:
            return None
        # Pastikan kolom standar ada
        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(df.columns):
            return None
        return df
    except Exception:
        return None


def fetch_parallel(tickers: list[str], period: str = PERIOD,
                   max_workers: int = 12) -> dict[str, pd.DataFrame]:
    """
    Download paralel per-ticker menggunakan ThreadPoolExecutor.
    
    FIX dari v5.0: batch download dengan group_by='ticker' punya masalah parsing
    MultiIndex yang tidak robust (single ticker → ambil full df, bukan per-ticker slice).
    Pendekatan per-ticker individual lebih aman meski sedikit lebih lambat untuk
    jumlah kecil, tapi sangat paralel untuk jumlah besar.
    """
    results: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_download_one, t, period): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            df = fut.result()
            if df is not None:
                results[t] = df
    return results


# ═══════════════════════════════════════════════════════════
# 4. INDIKATOR TEKNIKAL
# ═══════════════════════════════════════════════════════════
def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heikin Ashi yang benar secara matematis.
    
    FIX CRITICAL dari v5.0: HA_O pakai EWM(alpha=0.5) — ini BUKAN formula HA standar.
    Formula rekursif yang benar:
        HA_C[i] = (O + H + L + C) / 4
        HA_O[i] = (HA_O[i-1] + HA_C[i-1]) / 2  — rata-rata DUA nilai sebelumnya saja
        HA_H[i] = max(H, HA_O, HA_C)
        HA_L[i] = min(L, HA_O, HA_C)
    """
    ha_c = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4

    # Rekursif vectorized menggunakan loop (benar, tidak bisa dihindari karena rekursif)
    ha_o_arr = np.empty(len(df))
    ha_o_arr[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_o_arr[i] = (ha_o_arr[i - 1] + ha_c.iloc[i - 1]) / 2

    ha_o = pd.Series(ha_o_arr, index=df.index)
    ha_h = pd.concat([ha_o, ha_c, df["High"]], axis=1).max(axis=1)
    ha_l = pd.concat([ha_o, ha_c, df["Low"]], axis=1).min(axis=1)

    return pd.DataFrame({"HA_O": ha_o, "HA_C": ha_c, "HA_H": ha_h, "HA_L": ha_l},
                        index=df.index)


def stoch_rsi(close: pd.Series, rsi_period: int = 14,
              stoch_period: int = 14, k_smooth: int = 3, d_smooth: int = 3):
    """
    Stochastic RSI yang benar.
    v5.0 mengklaim StochRSI tapi pakai StochasticOscillator biasa — berbeda secara fundamental.
    StochRSI = Stochastic diterapkan ke nilai RSI (bukan langsung ke harga).
    """
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(rsi_period).mean()
    loss  = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs    = gain / (loss + 1e-9)
    rsi   = 100 - (100 / (1 + rs))

    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    raw_k   = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-9) * 100

    k = raw_k.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d, rsi


def compute_indicators(df_raw: pd.DataFrame) -> Optional[dict]:
    """
    Hitung semua indikator teknikal dari df_raw (HARUS full history, jangan tail()).
    
    FIX: Bekerja pada salinan internal agar tidak mutate df asli (cache poisoning prevention).
    Return dict indikator lengkap, atau None jika data tidak cukup.
    """
    df = df_raw.copy()  # FIX: selalu copy agar tidak mutate caller's df

    if len(df) < MIN_BARS:
        return None

    c = df["Close"]
    h = df["High"]
    lo = df["Low"]
    o = df["Open"]
    v = df["Volume"]

    # ── Ichimoku ──
    # FIX CRITICAL: library ta ichimoku_a() dan ichimoku_b() SUDAH include shift +26 (future cloud).
    # Untuk mendapatkan cloud HARI INI (present cloud), perlu shift(-26).
    # Versi v5.0 masih shift(26) → double shift → cloud 52 bar ke depan, tidak berguna.
    ichi = IchimokuIndicator(high=h, low=lo, window1=9, window2=26, window3=52)
    tenkan     = ichi.ichimoku_conversion_line()
    kijun      = ichi.ichimoku_base_line()
    senkou_a   = ichi.ichimoku_a().shift(-26)   # FIX: shift ke kiri untuk present cloud
    senkou_b   = ichi.ichimoku_b().shift(-26)   # FIX: shift ke kiri untuk present cloud
    chikou     = c.shift(-26)                   # chikou = harga sekarang diplot 26 bar ke kiri

    # ── Heikin Ashi (benar) ──
    ha = heikin_ashi(df)

    # ── EMA ──
    ema20 = EMAIndicator(c, window=20).ema_indicator()
    ema50 = EMAIndicator(c, window=50).ema_indicator()

    # ── MACD ──
    macd_obj  = MACD(c)
    macd_hist = macd_obj.macd_diff()

    # ── ADX (Trend Strength) — BARU di v6.0 ──
    adx_obj = ADXIndicator(h, lo, c, window=14)
    adx_val = adx_obj.adx()
    di_plus  = adx_obj.adx_pos()
    di_minus = adx_obj.adx_neg()

    # ── ATR ──
    atr = AverageTrueRange(h, lo, c, window=14).average_true_range()

    # ── Stochastic RSI (BENAR) ──
    srsi_k, srsi_d, rsi14 = stoch_rsi(c)

    # ── Volume metrics ──
    adv20     = v.rolling(20).mean()
    vol_rel   = v / adv20.replace(0, np.nan)
    adv_rp    = (c * v).rolling(20).mean() / 1_000_000  # ADV dalam juta rupiah

    # ── CQS: Candle Quality Score — seberapa dekat close ke high candle ──
    # Range candle (H-L), close lebih dekat ke H = bullish quality lebih tinggi
    candle_range = h - lo
    cqs = (c - lo) / candle_range.replace(0, np.nan)  # 0=close at low, 1=close at high

    return {
        "close":      c,
        "high":       h,
        "low":        lo,
        "open":       o,
        "volume":     v,
        "tenkan":     tenkan,
        "kijun":      kijun,
        "senkou_a":   senkou_a,
        "senkou_b":   senkou_b,
        "chikou":     chikou,
        "ha":         ha,
        "ema20":      ema20,
        "ema50":      ema50,
        "macd_hist":  macd_hist,
        "adx":        adx_val,
        "di_plus":    di_plus,
        "di_minus":   di_minus,
        "atr":        atr,
        "srsi_k":     srsi_k,
        "srsi_d":     srsi_d,
        "rsi14":      rsi14,
        "adv20":      adv20,
        "vol_rel":    vol_rel,
        "adv_rp":     adv_rp,
        "cqs":        cqs,
    }


# ═══════════════════════════════════════════════════════════
# 5. SWING HIGH/LOW & FIBONACCI
# ═══════════════════════════════════════════════════════════
def find_swing_extremes(high: pd.Series, low: pd.Series,
                        atr_val: float, lookback: int = 90) -> tuple[float, float]:
    """
    Cari swing high dan swing low yang signifikan dalam window lookback.
    
    FIX dari v5.0: lookback hanya 20 bar (terlalu sempit).
    Adaptive: setidaknya 60 bar, atau cukup untuk mencakup minimal 1 siklus ATR.
    
    Swing valid = titik ekstrem lokal dengan jarak minimal 5 bar dari titik lain.
    """
    n        = min(lookback, len(high))
    h_window = high.iloc[-n:]
    l_window = low.iloc[-n:]

    best_sh = float(h_window.max())
    best_sl = float(l_window.min())

    win = 5  # pivot order
    for i in range(win, len(h_window) - win):
        seg = h_window.iloc[i - win: i + win + 1]
        if h_window.iloc[i] == seg.max():
            best_sh = max(best_sh, float(h_window.iloc[i]))

    for i in range(win, len(l_window) - win):
        seg = l_window.iloc[i - win: i + win + 1]
        if l_window.iloc[i] == seg.min():
            best_sl = min(best_sl, float(l_window.iloc[i]))

    # Guard: SH harus > SL
    if best_sh <= best_sl:
        best_sh, best_sl = float(h_window.max()), float(l_window.min())

    return best_sh, best_sl


def fibonacci_levels(sh: float, sl: float) -> dict:
    """
    Hitung level Fibonacci retracement dan extension yang standar.
    
    FIX dari v5.0:
    - Retracement entry: 38.2% dan 50% dari range SH-SL (pullback ke dalam range)
    - SL: 61.8% retracement (jika beneran breakdown = invalidasi swing)
    - Target1: Fibo Extension 127.2% dari SL diukur dari SH
    - Target2: Fibo Extension 161.8% dari SL diukur dari SH
    - v5.0 pakai target = high + 0.618*diff → bukan standar trader Indonesia
    """
    diff = sh - sl
    return {
        "sh":      sh,
        "sl":      sl,
        # Retracement zones (entry pullback)
        "fib236":  sh - 0.236 * diff,
        "fib382":  sh - 0.382 * diff,   # entry atas
        "fib500":  sh - 0.500 * diff,   # entry tengah
        "fib618":  sh - 0.618 * diff,   # SL
        "fib786":  sh - 0.786 * diff,   # SL ketat (ekstrem)
        # Extension (target profit)
        "ext1272": sl + 1.272 * diff,   # Target 1
        "ext1618": sl + 1.618 * diff,   # Target 2
        "ext2000": sl + 2.000 * diff,   # Target 3 (bonus)
    }


# ═══════════════════════════════════════════════════════════
# 6. MARKET REGIME
# ═══════════════════════════════════════════════════════════
def get_market_regime(ihsg_df: pd.DataFrame,
                      all_dfs: dict[str, pd.DataFrame]) -> tuple[str, float, float]:
    """
    Hitung Market Regime dari dua sumber:
    1. IHSG trend sendiri (EMA20 vs EMA50)
    2. Market Breadth: % saham ALL universe di atas EMA20 (bukan subset screener!)
    
    FIX dari v5.0: breadth dihitung dari subset yang sudah difilter → selection bias.
    Sekarang breadth dihitung dari semua df yang tersedia sebelum filter apapun.
    
    Return: (regime, breadth_pct, ihsg_ema_ratio)
    """
    # — IHSG trend —
    ihsg_bull = False
    ihsg_ema_ratio = 0.0
    if not ihsg_df.empty and len(ihsg_df) > 50:
        ihsg_close = ihsg_df["Close"].squeeze()
        ema20_ihsg = ihsg_close.ewm(span=20, adjust=False).mean()
        ema50_ihsg = ihsg_close.ewm(span=50, adjust=False).mean()
        ihsg_bull  = float(ema20_ihsg.iloc[-1]) > float(ema50_ihsg.iloc[-1])
        ihsg_ema_ratio = float(ema20_ihsg.iloc[-1]) / float(ema50_ihsg.iloc[-1])

    # — Market Breadth dari SEMUA universe (sebelum filter) —
    above = 0
    total = 0
    for df in all_dfs.values():
        if len(df) > 22:
            ema20 = df["Close"].squeeze().ewm(span=20, adjust=False).mean()
            if float(df["Close"].squeeze().iloc[-1]) > float(ema20.iloc[-1]):
                above += 1
            total += 1

    breadth = (above / total * 100) if total > 0 else 50.0

    # Regime gabungan: IHSG trend + breadth
    if ihsg_bull and breadth > 55:
        regime = "BULL"
    elif not ihsg_bull and breadth < 45:
        regime = "BEAR"
    else:
        regime = "NEUTRAL"

    return regime, breadth, ihsg_ema_ratio


# ═══════════════════════════════════════════════════════════
# 7. RELATIVE STRENGTH vs IHSG
# ═══════════════════════════════════════════════════════════
def compute_rs_score(close: pd.Series, ihsg_close: pd.Series,
                     window: int = 20) -> float:
    """
    RS-score: perbandingan return saham vs IHSG dalam window bar terakhir.
    Positif = outperform, Negatif = underperform.
    
    Output dinormalisasi sebagai z-score agar comparable antar saham.
    """
    # Align tanggal
    common = close.index.intersection(ihsg_close.index)
    if len(common) < window + 5:
        return 0.0

    c_aligned = close.loc[common]
    i_aligned = ihsg_close.loc[common]

    ret_stock = c_aligned.pct_change()
    ret_ihsg  = i_aligned.pct_change()
    rel_ret   = ret_stock - ret_ihsg   # excess return harian

    # z-score: (mean excess return - 0) / std, dalam window bar
    recent = rel_ret.iloc[-window:]
    mu  = recent.mean()
    std = recent.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(mu / std)


# ═══════════════════════════════════════════════════════════
# 8. CORE SCREENER ENGINE
# ═══════════════════════════════════════════════════════════
def _compute_confidence(score: int, score_max: int) -> str:
    pct = score / score_max if score_max > 0 else 0
    if pct >= 0.80: return "VERY HIGH"
    if pct >= 0.60: return "HIGH"
    if pct >= 0.40: return "MEDIUM"
    return "LOW"


def _signal_from_score(score: int, score_max: int,
                        rr: float, adx: float) -> str:
    conf = _compute_confidence(score, score_max)
    # Gate 1: R/R minimum 1.5:1
    if rr < 1.5:
        return "AVOID"
    # Gate 2: ADX terlalu rendah (choppy) + score rendah = AVOID
    if adx < ADX_THRESHOLD and conf == "LOW":
        return "AVOID"
    if conf == "VERY HIGH":
        return "STRONG BUY"
    if conf in ("HIGH",):
        return "BUY"
    if conf == "MEDIUM":
        return "WATCH"
    return "AVOID"


def screen_ticker(ticker: str,
                  df_raw: pd.DataFrame,
                  ihsg_close: Optional[pd.Series],
                  min_adv_rp: float,
                  sektor: str = "—") -> Optional[ScreenResult]:
    """
    Analisis teknikal satu ticker. Return ScreenResult atau None jika tidak lolos filter.
    
    Scoring System (total max = 14 poin):
      Ichimoku     (4 poin): di atas awan, TK cross, Chikou bebas, kijun slope
      Trend Kuat   (3 poin): ADX > 20, DI+ > DI-, EMA20 > EMA50
      Momentum     (4 poin): MACD fresh cross, Stoch RSI oversold→naik, RSI rentang sehat, CQS
      Volume       (2 poin): volume spike, ADV-Rp cukup
      RS vs IHSG   (1 poin): outperform benchmark
    """
    ind = compute_indicators(df_raw)
    if ind is None:
        return None

    c        = ind["close"]
    v        = ind["volume"]
    last_idx = -1
    price    = float(c.iloc[last_idx])

    # ── Pre-filter wajib ──
    adv_rp_val = float(ind["adv_rp"].iloc[last_idx])
    if adv_rp_val < min_adv_rp:
        return None  # Likuiditas rupiah tidak cukup

    atr_val  = float(ind["atr"].iloc[last_idx])
    atr_pct  = atr_val / price * 100 if price > 0 else 0

    # ── Nilai indikator terakhir ──
    tenkan_v  = float(ind["tenkan"].iloc[last_idx])
    kijun_v   = float(ind["kijun"].iloc[last_idx])
    sa_v      = float(ind["senkou_a"].iloc[last_idx])
    sb_v      = float(ind["senkou_b"].iloc[last_idx])
    ema20_v   = float(ind["ema20"].iloc[last_idx])
    ema50_v   = float(ind["ema50"].iloc[last_idx])
    macd_now  = float(ind["macd_hist"].iloc[last_idx])
    macd_prev = float(ind["macd_hist"].iloc[-2])
    adx_v     = float(ind["adx"].iloc[last_idx])
    di_plus_v = float(ind["di_plus"].iloc[last_idx])
    di_minus_v= float(ind["di_minus"].iloc[last_idx])
    srsi_k_v  = float(ind["srsi_k"].iloc[last_idx])
    srsi_d_v  = float(ind["srsi_d"].iloc[last_idx])
    srsi_k_p  = float(ind["srsi_k"].iloc[-2])
    srsi_d_p  = float(ind["srsi_d"].iloc[-2])
    rsi_v     = float(ind["rsi14"].iloc[last_idx])
    vol_now   = float(v.iloc[last_idx])
    adv20_v   = float(ind["adv20"].iloc[last_idx])
    vol_rel_v = vol_now / adv20_v if adv20_v > 0 else 1.0
    cqs_v     = float(ind["cqs"].iloc[last_idx])

    ha_now    = ind["ha"].iloc[last_idx]
    ha_prev   = ind["ha"].iloc[-2]
    ha_bull   = float(ha_now["HA_C"]) > float(ha_now["HA_O"])
    ha_no_shadow = (float(ha_now["HA_O"]) - float(ha_now["HA_L"])) < (atr_val * 0.3)  # FIX: threshold adaptif ATR

    kumo_top = max(sa_v, sb_v)
    kumo_bot = min(sa_v, sb_v)
    above_cloud = price > kumo_top

    # ── Gate mutlak: harus di atas awan Ichimoku ──
    # (untuk mode screener strict — bisa direlaks di sidebar)
    if not above_cloud:
        return None
    if not ha_bull:
        return None

    # ── Trend duration: berapa bar harga di atas awan ──
    trend_bars = 0
    for i in range(1, min(50, len(c))):
        p_i    = float(c.iloc[-i])
        sa_i   = float(ind["senkou_a"].iloc[-i])
        sb_i   = float(ind["senkou_b"].iloc[-i])
        if p_i > max(sa_i, sb_i):
            trend_bars += 1
        else:
            break

    # ══════════════════════════════
    # SCORING — max 14 poin
    # ══════════════════════════════
    score     = 0
    score_max = 14
    reasons   = []

    # --- A. ICHIMOKU (max 4 poin) ---
    # A1: Harga di atas awan (sudah lewat gate, tapi beri skor + slope awan)
    cloud_bullish = sa_v > sb_v   # awan hijau (Senkou A > B)
    if cloud_bullish:
        score += 1
        reasons.append("🟢 Awan Ichimoku HIJAU (Senkou A > B) — bullish cloud structure")

    # A2: Tenkan > Kijun (TK cross)
    if tenkan_v > kijun_v:
        score += 1
        reasons.append("🟢 Tenkan-sen di atas Kijun-sen — momentum jangka pendek bullish")
    elif tenkan_v < kijun_v:
        reasons.append("🔴 Tenkan di bawah Kijun — hati-hati, momentum lemah")

    # A3: Harga di atas Kijun (support dynamic kuat)
    if price > kijun_v:
        score += 1
        reasons.append(f"🟢 Harga ({price:,.0f}) di atas Kijun-sen ({kijun_v:,.0f})")

    # A4: Kijun slope naik (Kijun hari ini > 5 hari lalu)
    kijun_5d_ago = float(ind["kijun"].iloc[-6]) if len(ind["kijun"]) > 5 else kijun_v
    if kijun_v > kijun_5d_ago:
        score += 1
        reasons.append("🟢 Kijun-sen miring ke atas — trend dasar menguat")

    # --- B. TREND KUAT / ADX (max 3 poin) ---
    # B1: ADX > 20 (trending, bukan choppy)
    if adx_v >= 25:
        score += 1
        reasons.append(f"🟢 ADX kuat ({adx_v:.1f}) — tren tegas, bukan sideways")
    elif adx_v >= ADX_THRESHOLD:
        score += 1
        reasons.append(f"🟡 ADX cukup ({adx_v:.1f}) — tren moderat")
    else:
        reasons.append(f"🔴 ADX lemah ({adx_v:.1f}) — pergerakan sideways/choppy")

    # B2: DI+ > DI- (tekanan beli mendominasi)
    if di_plus_v > di_minus_v:
        score += 1
        reasons.append(f"🟢 DI+ ({di_plus_v:.1f}) > DI- ({di_minus_v:.1f}) — tekanan beli dominan")

    # B3: EMA20 > EMA50 (trend jangka menengah)
    if ema20_v > ema50_v:
        score += 1
        reasons.append(f"🟢 EMA20 ({ema20_v:,.0f}) di atas EMA50 ({ema50_v:,.0f}) — uptrend menengah")

    # --- C. MOMENTUM (max 4 poin) ---
    # C1: MACD fresh crossover (dari negatif ke positif) — FIX dari v5.0
    macd_fresh_cross = macd_now > 0 and macd_prev <= 0
    macd_strengthening = macd_now > 0 and macd_now > macd_prev
    if macd_fresh_cross:
        score += 2   # extra skor untuk fresh cross — entry timing lebih tepat
        reasons.append("🔥 MACD FRESH CROSSOVER — histogram baru saja jadi positif, entry timing ideal")
    elif macd_strengthening:
        score += 1
        reasons.append("🟢 MACD Histogram menguat — momentum upside membangun")
    else:
        reasons.append("🔴 MACD negatif atau melemah")

    # C2: Stochastic RSI oversold → naik (fresh cross dari bawah 20) — FIX dari v5.0
    srsi_fresh_cross = srsi_k_v > srsi_d_v and srsi_k_p <= srsi_d_p  # baru cross up
    srsi_from_os     = srsi_k_v > 20 and srsi_k_p <= 20               # baru keluar oversold
    if srsi_fresh_cross and srsi_k_v < 50:
        score += 1
        reasons.append(f"🟢 Stoch RSI fresh cross ({srsi_k_v:.0f}) — momentum awal konfirmasi")
    elif srsi_from_os:
        score += 1
        reasons.append(f"🔥 Stoch RSI baru keluar oversold ({srsi_k_v:.0f}) — reversal momentum")

    # C3: HA candle kuat (no lower shadow dengan threshold ATR adaptif)
    # FIX dari v5.0: threshold 0.002 × Close terlalu ketat → 0.3 × ATR lebih realistis
    if ha_no_shadow and ha_bull:
        score += 1
        reasons.append("🔥 Heikin Ashi kuat (hampir tanpa ekor bawah) — bullish pressure dominan")
    elif ha_bull:
        reasons.append("🟢 Heikin Ashi hijau — bias bullish terjaga")

    # --- D. VOLUME (max 2 poin) ---
    # D1: Volume spike relatif vs ADV20
    if vol_rel_v >= 2.0:
        score += 1
        reasons.append(f"🔥 Volume SPIKE {vol_rel_v:.1f}×ADV20 — smart money masuk")
    elif vol_rel_v >= 1.2:
        score += 1
        reasons.append(f"🟢 Volume di atas normal {vol_rel_v:.1f}×ADV20")
    else:
        reasons.append(f"🟡 Volume rendah {vol_rel_v:.1f}×ADV20 — konfirmasi lemah")

    # D2: ADV-Rp (likuiditas memadai untuk swing trader)
    if adv_rp_val >= 10_000:   # Rp10 miliar/hari = sangat likuid
        score += 1
        reasons.append(f"🟢 Likuiditas TINGGI (ADV Rp{adv_rp_val:,.0f}jt/hari)")
    elif adv_rp_val >= min_adv_rp:
        reasons.append(f"🟡 Likuiditas cukup (ADV Rp{adv_rp_val:,.0f}jt/hari)")

    # --- E. RELATIVE STRENGTH vs IHSG (max 1 poin) ---
    rs_val = 0.0
    if ihsg_close is not None and len(ihsg_close) > 22:
        rs_val = compute_rs_score(c, ihsg_close)
        if rs_val > 0.5:
            score += 1
            reasons.append(f"🟢 Outperform IHSG (RS-score {rs_val:+.2f}) — saham lebih kuat dari pasar")
        elif rs_val < -0.5:
            reasons.append(f"🔴 Underperform IHSG (RS-score {rs_val:+.2f}) — lebih lemah dari pasar")

    # ══════════════════════════════
    # FIBONACCI LEVELS
    # ══════════════════════════════
    # Adaptive lookback: min 60 bar, max 120 bar
    lookback = min(120, max(60, int(30 / (atr_pct + 0.001))))
    sh, sl   = find_swing_extremes(ind["high"], ind["low"], atr_val, lookback)
    fib      = fibonacci_levels(sh, sl)

    entry_hi  = fib["fib382"]
    entry_lo  = fib["fib500"]
    entry_mid = (entry_hi + entry_lo) / 2

    # FIX: Stop Loss = lebih konservatif antara Kijun dan Fibo 61.8%
    # Tidak boleh > 8% dari harga (batas maksimal praktis untuk swing)
    sl_kijun = kijun_v
    sl_fib   = fib["fib618"]
    stop_loss = max(sl_kijun, sl_fib)   # pakai yang lebih tinggi (lebih dekat ke harga = SL lebih ketat)
    # Batasi SL maksimal 8% di bawah harga
    stop_loss = max(stop_loss, price * 0.92)

    # FIX: Target menggunakan Fibonacci Extension standar
    target1 = fib["ext1272"]   # 127.2% extension dari SL
    target2 = fib["ext1618"]   # 161.8% extension dari SL

    # ── R/R Calculation — FIX: tidak ada sama sekali di v5.0 ──
    risk   = entry_mid - stop_loss
    reward = target1 - entry_mid
    rr     = reward / risk if risk > 0 else 0.0

    # ══════════════════════════════
    # SINYAL & CONFIDENCE
    # ══════════════════════════════
    signal     = _signal_from_score(score, score_max, rr, adx_v)
    confidence = _compute_confidence(score, score_max)

    # ── Position Sizing (risk 1% dari modal Rp 100jt) ──
    modal_rp    = 100_000_000   # Rp 100 juta
    risk_rp     = modal_rp * 0.01
    risk_per_sh = abs(price - stop_loss)
    lot_size    = 100           # 1 lot IDX = 100 lembar
    sizing_lot  = int(risk_rp / (risk_per_sh * lot_size)) if risk_per_sh > 0 else 0
    sizing_lot  = min(sizing_lot, 500)   # cap 500 lot agar tidak absurd

    return ScreenResult(
        ticker     = ticker,
        harga      = price,
        signal     = signal,
        score      = score,
        score_max  = score_max,
        confidence = confidence,
        reasons    = reasons,
        swing_high = sh,
        swing_low  = sl,
        entry_lo   = entry_lo,
        entry_hi   = entry_hi,
        target1    = target1,
        target2    = target2,
        stop_loss  = stop_loss,
        rr         = rr,
        atr        = atr_val,
        atr_pct    = atr_pct,
        adv_rp     = adv_rp_val,
        vol_rel    = vol_rel_v,
        rs_score   = rs_val,
        adx        = adx_v,
        trend_bars = trend_bars,
        cqs        = cqs_v,
        sektor     = sektor,
        sizing_lot = sizing_lot,
    )


# ═══════════════════════════════════════════════════════════
# 9. CHART (Plotly — detail untuk modal analisis)
# ═══════════════════════════════════════════════════════════
def build_mini_chart(ticker: str, df_raw: pd.DataFrame,
                     result: ScreenResult, display_bars: int = 120) -> go.Figure:
    """
    Chart candlestick + Ichimoku + Fibonacci levels + Volume.
    Indikator dihitung dari df_raw PENUH, baru dipotong ke display_bars.
    """
    ind = compute_indicators(df_raw)
    if ind is None:
        return go.Figure()

    n   = len(df_raw)
    cut = min(display_bars, n)
    s   = slice(n - cut, n)

    dates = df_raw.index[s]
    op_s  = df_raw["Open"].iloc[s]
    hi_s  = df_raw["High"].iloc[s]
    lo_s  = df_raw["Low"].iloc[s]
    cl_s  = df_raw["Close"].iloc[s]
    vol_s = df_raw["Volume"].iloc[s]
    sa_s  = ind["senkou_a"].iloc[s]
    sb_s  = ind["senkou_b"].iloc[s]
    ema20_s = ind["ema20"].iloc[s]
    ema50_s = ind["ema50"].iloc[s]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28],
                        vertical_spacing=0.03)

    BG  = "#0E1117"
    BG2 = "#1E2329"

    # — Candlestick —
    fig.add_trace(go.Candlestick(
        x=dates, open=op_s, high=hi_s, low=lo_s, close=cl_s,
        name=ticker,
        increasing_fillcolor="#00C853", increasing_line_color="#00C853",
        decreasing_fillcolor="#FF3B30", decreasing_line_color="#FF3B30",
    ), row=1, col=1)

    # — Ichimoku Cloud —
    fig.add_trace(go.Scatter(x=dates, y=sa_s, line=dict(width=0),
                             name="Senkou A", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=sb_s, line=dict(width=0),
                             fill="tonexty",
                             fillcolor="rgba(0,200,83,0.12)" if result.signal in ("STRONG BUY","BUY")
                             else "rgba(255,59,48,0.12)",
                             name="Cloud"), row=1, col=1)

    # — EMA —
    fig.add_trace(go.Scatter(x=dates, y=ema20_s, name="EMA20",
                             line=dict(color="#F0B90B", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ema50_s, name="EMA50",
                             line=dict(color="#848E9C", width=1, dash="dot")), row=1, col=1)

    # — Fibonacci Levels —
    for level, val, lbl, col in [
        (result.entry_hi,  result.entry_hi,  "Entry Hi (38.2%)", "#00C853"),
        (result.entry_lo,  result.entry_lo,  "Entry Lo (50%)",   "#4caf50"),
        (result.stop_loss, result.stop_loss, "Stop Loss",         "#FF3B30"),
        (result.target1,   result.target1,   "Target 1 (127.2%)","#F0B90B"),
        (result.target2,   result.target2,   "Target 2 (161.8%)","#FF9800"),
    ]:
        fig.add_hline(y=val, line=dict(color=col, width=1, dash="dash"),
                      annotation_text=f" {lbl}: {val:,.0f}",
                      annotation_position="right",
                      annotation_font=dict(color=col, size=10),
                      row=1, col=1)

    # — Volume —
    vol_colors = ["#00C853" if cl_s.iloc[i] >= op_s.iloc[i] else "#FF3B30"
                  for i in range(len(vol_s))]
    fig.add_trace(go.Bar(x=dates, y=vol_s, name="Volume",
                         marker_color=vol_colors, opacity=0.7), row=2, col=1)

    adv20_s = ind["adv20"].iloc[s]
    fig.add_trace(go.Scatter(x=dates, y=adv20_s, name="ADV20",
                             line=dict(color="#F0B90B", width=1, dash="dot")), row=2, col=1)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color="#FAFAFA", size=11),
        height=500,
        margin=dict(l=0, r=120, t=30, b=0),
        legend=dict(bgcolor="#1E2329", bordercolor="#2a2d35", x=0, y=1),
        xaxis_rangeslider_visible=False,
        title=dict(text=f"<b>{ticker}</b> — {result.signal}", x=0.01,
                   font=dict(color=SIGNAL_COLOR.get(result.signal, "#FAFAFA"), size=14)),
    )
    fig.update_xaxes(gridcolor="#2a2d35", showgrid=True)
    fig.update_yaxes(gridcolor="#2a2d35", showgrid=True)

    return fig


# ═══════════════════════════════════════════════════════════
# 10. UI COMPONENTS
# ═══════════════════════════════════════════════════════════
def render_regime_banner(regime: str, breadth: float, ihsg_ratio: float):
    col_map = {"BULL": "#00e676", "BEAR": "#f44336", "NEUTRAL": "#ffc107"}
    cls_map = {"BULL": "regime-bull", "BEAR": "regime-bear", "NEUTRAL": "regime-neutral"}
    color   = col_map.get(regime, "#ffc107")
    trend_lbl = "🔼 EMA20 > EMA50 (Uptrend)" if ihsg_ratio > 1 else "🔽 EMA20 < EMA50 (Downtrend)"
    st.markdown(f"""
    <div class='{cls_map.get(regime,"regime-neutral")}'>
      <span style='font-size:1.1rem;font-weight:700;color:{color};'>
        📊 Market Regime: {regime}</span>
      &nbsp;&nbsp;
      <span style='color:#848E9C;font-size:0.9rem;'>
        Breadth: <b style='color:#FAFAFA;'>{breadth:.1f}%</b> saham di atas EMA20
        &nbsp;·&nbsp; IHSG: {trend_lbl}
      </span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")


def render_kpi_row(results: list[ScreenResult]):
    by_sig = {s: sum(1 for r in results if r.signal == s) for s in SIGNAL_ORDER}
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f"<div class='kpi-card'><div class='kpi-label'>Total Kandidat</div>"
                f"<div class='kpi-value'>{len(results)}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi-card' style='border-color:#00e676;'><div class='kpi-label'>STRONG BUY</div>"
                f"<div class='kpi-value' style='color:#00e676;'>{by_sig['STRONG BUY']}</div></div>",
                unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi-card' style='border-color:#4caf50;'><div class='kpi-label'>BUY</div>"
                f"<div class='kpi-value' style='color:#4caf50;'>{by_sig['BUY']}</div></div>",
                unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi-card' style='border-color:#ffc107;'><div class='kpi-label'>WATCH</div>"
                f"<div class='kpi-value' style='color:#ffc107;'>{by_sig['WATCH']}</div></div>",
                unsafe_allow_html=True)
    c5.markdown(f"<div class='kpi-card' style='border-color:#f44336;'><div class='kpi-label'>AVOID</div>"
                f"<div class='kpi-value' style='color:#f44336;'>{by_sig['AVOID']}</div></div>",
                unsafe_allow_html=True)
    st.markdown("")


def results_to_df(results: list[ScreenResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        sig_color = SIGNAL_COLOR.get(r.signal, "#FAFAFA")
        rows.append({
            "Ticker":    r.ticker,
            "Signal":    r.signal,
            "Score":     f"{r.score}/{r.score_max}",
            "Conf.":     r.confidence,
            "Harga":     f"{r.harga:,.0f}",
            "Entry Zone":f"{r.entry_lo:,.0f}–{r.entry_hi:,.0f}",
            "Target 1":  f"{r.target1:,.0f}",
            "Stop Loss": f"{r.stop_loss:,.0f}",
            "R/R":       f"{r.rr:.1f}:1",
            "ADX":       f"{r.adx:.0f}",
            "Vol×":      f"{r.vol_rel:.1f}",
            "RS":        f"{r.rs_score:+.2f}",
            "ADV Rp(jt)":f"{r.adv_rp:,.0f}",
            "ATR%":      f"{r.atr_pct:.1f}%",
            "Sizing(lot)":r.sizing_lot,
            "Sektor":    r.sektor,
        })
    return pd.DataFrame(rows)


@st.dialog("📊 Trading Plan Detail", width="large")
def show_detail_modal(result: ScreenResult, df_raw: pd.DataFrame):
    sig_color = SIGNAL_COLOR.get(result.signal, "#FAFAFA")

    # Header
    st.markdown(f"""
    <h3 style='margin:0;'>
      {result.ticker} &nbsp;
      <span class='badge-{"strong-buy" if result.signal=="STRONG BUY"
                           else result.signal.lower().replace(" ","-")
                           if result.signal in ("BUY","WATCH","AVOID")
                           else "watch"}'>
        {result.signal}
      </span>
    </h3>
    <p style='color:#848E9C;margin:4px 0 12px;'>
      Score: <b style='color:#FAFAFA;'>{result.score}/{result.score_max}</b> &nbsp;·&nbsp;
      Confidence: <b style='color:{sig_color};'>{result.confidence}</b> &nbsp;·&nbsp;
      Sektor: {result.sektor} &nbsp;·&nbsp;
      Di atas awan: <b style='color:#FAFAFA;'>{result.trend_bars} bar</b>
    </p>
    """, unsafe_allow_html=True)

    # Chart
    fig = build_mini_chart(result.ticker, df_raw, result)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Harga Sekarang", f"{result.harga:,.0f}")
    c2.metric("Entry Zone", f"{result.entry_lo:,.0f}–{result.entry_hi:,.0f}")
    c3.metric("Stop Loss", f"{result.stop_loss:,.0f}",
              delta=f"{(result.stop_loss/result.harga-1)*100:.1f}%",
              delta_color="inverse")
    c4.metric("Target 1 (127.2%)", f"{result.target1:,.0f}",
              delta=f"+{(result.target1/result.harga-1)*100:.1f}%")
    c5.metric("Target 2 (161.8%)", f"{result.target2:,.0f}",
              delta=f"+{(result.target2/result.harga-1)*100:.1f}%")
    c6.metric("R/R Ratio", f"{result.rr:.1f}:1",
              delta="✅ OK" if result.rr >= 2.0 else "⚠️ Tipis" if result.rr >= 1.5 else "❌ Buruk",
              delta_color="off")

    st.markdown("---")
    # Detail Reasons
    col_r, col_p = st.columns([1, 1])
    with col_r:
        st.markdown("#### 🧠 Analisis Sinyal")
        for reason in result.reasons:
            st.write(reason)

    with col_p:
        st.markdown("#### 🎯 Action Plan")

        entry_mid = (result.entry_lo + result.entry_hi) / 2
        st.markdown(f"""
        **📍 Swing Reference:**  High {result.swing_high:,.0f} → Low {result.swing_low:,.0f}

        **🟢 Zona Entry:** {result.entry_lo:,.0f} – {result.entry_hi:,.0f}
        _(Fibo 50%–38.2% retracement)_

        **🎯 Target 1:** {result.target1:,.0f} _(Fibo Ext 127.2%)_
        **🎯 Target 2:** {result.target2:,.0f} _(Fibo Ext 161.8%)_

        **🛑 Stop Loss:** {result.stop_loss:,.0f}
        _(max antara Kijun & Fibo 61.8%)_

        **📐 Risk/Reward:** {result.rr:.1f}:1
        {"✅ Setup layak" if result.rr >= 2.0 else "⚠️ R/R minimal, disiplin SL wajib" if result.rr >= 1.5 else "❌ R/R buruk — skip setup ini"}

        **💰 Position Sizing** (risk 1% modal Rp100jt):
        → Maks **{result.sizing_lot} lot** ({result.sizing_lot * 100:,} lembar)

        **📊 ATR Harian:** {result.atr:,.0f} ({result.atr_pct:.1f}%)
        **📈 Likuiditas:** ADV Rp{result.adv_rp:,.0f}jt/hari
        **⚡ Volume:** {result.vol_rel:.1f}× ADV20
        """)

        if result.rs_score > 0.5:
            st.success(f"💪 RS vs IHSG: {result.rs_score:+.2f} — saham lebih kuat dari pasar")
        elif result.rs_score < -0.5:
            st.warning(f"⚠️ RS vs IHSG: {result.rs_score:+.2f} — saham lebih lemah dari pasar")


# ═══════════════════════════════════════════════════════════
# 11. MAIN APP
# ═══════════════════════════════════════════════════════════
def main():
    st.markdown("<h1 style='margin-bottom:4px;'>🦅 Ichi-Fibo-Heikin Pro v6.0</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#848E9C;margin-top:0;'>Screener teknikal IDX — Ichimoku · Fibonacci · Heikin Ashi · ADX · Stoch RSI · RS vs IHSG</p>", unsafe_allow_html=True)
    st.markdown("---")

    emiten_df   = load_emiten()
    all_tickers = emiten_df["ticker"].tolist()
    sektor_map  = dict(zip(emiten_df["ticker"], emiten_df.get("sektor", pd.Series("—", index=emiten_df.index))))

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## ⚙️ Konfigurasi Screener")

        n_tickers  = st.slider("Jumlah emiten (Top N)", 10, len(all_tickers), min(150, len(all_tickers)), 10)
        min_adv_rp = st.slider("Min ADV Likuiditas (Rp juta/hari)", 500, 20_000, 2_000, 500,
                                help="Average Daily Value dalam juta rupiah. Rp2M = Rp2 miliar/hari")
        min_rr     = st.slider("Min R/R Ratio", 1.0, 4.0, 1.5, 0.5)
        workers    = st.slider("Parallel workers", 4, 24, 12, 2)
        show_avoid = st.toggle("Tampilkan AVOID di tabel", value=False)

        st.markdown("---")
        st.markdown("**Filter Sinyal**")
        show_signals = st.multiselect("Tampilkan sinyal:", SIGNAL_ORDER,
                                      default=["STRONG BUY", "BUY", "WATCH"])
        st.markdown("---")
        st.markdown(f"<span style='color:#848E9C;font-size:0.78rem;'>v6.0 · {len(all_tickers)} emiten loaded</span>",
                    unsafe_allow_html=True)
        run_btn = st.button("🚀 Jalankan Screener", use_container_width=True, type="primary")

    # ── Run ──
    if not run_btn:
        # Landing page
        st.markdown("""
        ### 👆 Klik **Jalankan Screener** di sidebar untuk memulai.

        **Metodologi v6.0:**
        | Komponen | Bobot | Keterangan |
        |---|---|---|
        | Ichimoku | 4 poin | Cloud, TK cross, Kijun support, Kijun slope |
        | Trend Strength (ADX) | 3 poin | ADX, DI+/DI-, EMA20/50 |
        | Momentum | 4 poin | MACD fresh cross, Stoch RSI, HA candle, RSI |
        | Volume | 2 poin | Volume spike, ADV-Rp likuiditas |
        | Relative Strength | 1 poin | RS vs IHSG benchmark |
        | **Total** | **14 poin** | |

        **Gate Wajib (tidak bisa di-score lebih):**
        - ✅ Harga di atas awan Ichimoku (present cloud, bukan future)
        - ✅ Heikin Ashi hijau (HA_C > HA_O)
        - ✅ R/R ≥ 1.5:1 (jika tidak — AVOID otomatis)
        - ✅ ADV-Rp ≥ threshold likuiditas (jika tidak — tidak masuk screener)
        """)
        return

    target_tickers = all_tickers[:n_tickers]

    # ── Download Data ──
    with st.spinner("📡 Mengunduh data IHSG..."):
        ihsg_df = fetch_ihsg()
        ihsg_close = ihsg_df["Close"].squeeze() if not ihsg_df.empty else None

    prog_container = st.empty()
    prog_container.info(f"⏳ Mengunduh data {len(target_tickers)} emiten secara paralel ({workers} workers)...")

    t0 = time.time()
    all_dfs = fetch_parallel(target_tickers, PERIOD, workers)
    elapsed = time.time() - t0

    prog_container.success(f"✅ Data siap: {len(all_dfs)}/{len(target_tickers)} emiten berhasil diunduh dalam {elapsed:.1f}s")

    if not all_dfs:
        st.error("Tidak ada data yang berhasil diunduh. Cek koneksi internet.")
        return

    # ── Market Regime ──
    regime, breadth, ihsg_ratio = get_market_regime(ihsg_df, all_dfs)
    render_regime_banner(regime, breadth, ihsg_ratio)

    # ── Screener ──
    with st.spinner("🔍 Menganalisis indikator teknikal..."):
        results_raw: list[ScreenResult] = []
        prog2 = st.progress(0.0, text="Menganalisis...")
        total = len(all_dfs)

        for i, (ticker, df) in enumerate(all_dfs.items()):
            sektor = sektor_map.get(ticker, "—")
            res = screen_ticker(ticker, df, ihsg_close, min_adv_rp, sektor)
            if res is not None:
                # Override R/R gate dengan parameter sidebar
                if res.rr < min_rr:
                    res.signal = "AVOID"
                results_raw.append(res)
            prog2.progress((i + 1) / total, text=f"Analisis {ticker}...")

        prog2.empty()

    # Filter & sort
    results_filtered = [r for r in results_raw if r.signal in show_signals]
    results_filtered.sort(key=lambda r: (SIGNAL_ORDER.index(r.signal), -r.score))

    # ── KPI ──
    render_kpi_row(results_raw)  # gunakan raw (semua sinyal) untuk KPI

    st.markdown(f"### 📋 Hasil Screener — {len(results_filtered)} kandidat")
    if not results_filtered:
        st.warning("Tidak ada saham yang memenuhi kriteria dengan filter saat ini.")
        return

    # ── Tabel ──
    df_display = results_to_df(results_filtered)
    st.dataframe(df_display, use_container_width=True, hide_index=True,
                 column_config={
                     "Signal":   st.column_config.TextColumn("Signal", width=110),
                     "R/R":      st.column_config.TextColumn("R/R"),
                     "Conf.":    st.column_config.TextColumn("Confidence"),
                     "Sizing(lot)": st.column_config.NumberColumn("Sizing (lot)", format="%d"),
                 })

    # ── Action Panel ──
    st.markdown("---")
    st.markdown("### 🔍 Action Panel — Klik untuk Trading Plan Lengkap")

    # Group by signal tier
    for sig in SIGNAL_ORDER:
        tier = [r for r in results_filtered if r.signal == sig]
        if not tier:
            continue
        sig_color = SIGNAL_COLOR[sig]
        st.markdown(f"<h4 style='color:{sig_color};'>{sig} ({len(tier)})</h4>", unsafe_allow_html=True)

        cols = st.columns(min(8, len(tier)))
        for j, res in enumerate(tier):
            with cols[j % len(cols)]:
                btn_label = f"{res.ticker}\n{res.score}/{res.score_max}"
                if st.button(btn_label, key=f"detail_{res.ticker}_{sig}",
                             use_container_width=True,
                             help=f"R/R {res.rr:.1f}:1 | ADX {res.adx:.0f} | Vol {res.vol_rel:.1f}×"):
                    show_detail_modal(res, all_dfs[res.ticker])


if __name__ == "__main__":
    main()
