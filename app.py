"""
Screener Ichi-Fibo-Heikin Pro  •  v9.0  (True Quant System)
════════════════════════════════════════════════════════════════════
TRUE QUANT UPGRADES (v8.0 -> v9.0):
  1. Continuous Win Prob (Sigmoid): math.tanh menggantikan if/else heuristik.
  2. Friction Model: R/R dikurangi slippage (0.2%) & komisi (0.15%).
  3. Regime-Adjusted Kelly: Kelly = 0 jika market bearish, maksimal jika bull.
  4. Volatility Kelly Scalar: Semakin tinggi ATR%, semakin kecil lot size.
  5. Portfolio Heat Cap: Batas total risk portofolio (mis. max 6% modal).
"""

from __future__ import annotations

import hashlib
import html
import logging
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange

# ═══════════════════════════════════════════════════════
# LOGGING & CONFIG
# ═══════════════════════════════════════════════════════
logger = logging.getLogger("ifh_quant_v9")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

st.set_page_config(page_title="IFH Quant v9.0", layout="wide", initial_sidebar_state="expanded", page_icon="🦅")

st.markdown("""
<style>
  .stApp { background:#0E1117; color:#FAFAFA; }
  .block-container { padding-top:1.2rem; }
  .kpi { background:#1E2329; border-radius:10px; padding:12px 16px; border-left:4px solid #F0B90B; margin-bottom:8px; }
  .kpi-lbl { font-size:.78rem; color:#848E9C; margin-bottom:2px; }
  .kpi-val { font-size:1.3rem; font-weight:700; color:#FAFAFA; }
  .regime-bull { background:#0a2f1a; border-left:5px solid #00e676; padding:10px 16px; border-radius:8px; margin-bottom:12px; }
  .regime-bear { background:#2f0a0a; border-left:5px solid #f44336; padding:10px 16px; border-radius:8px; margin-bottom:12px; }
  .regime-neutral { background:#2a2a0a; border-left:5px solid #ffc107; padding:10px 16px; border-radius:8px; margin-bottom:12px; }
  hr { border-color:#2a2d35; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════
IHSG_TICKER = "^JKSE"
PERIOD      = "2y"
MIN_BARS    = 90
CHUNK_SIZE  = 20
MAX_RETRY   = 3
MAX_LOT_CAP = 1000
DISPLAY_BARS = 120
SHARES_PER_LOT = 100

SIGNAL_ORDER = ["STRONG BUY", "BUY", "WATCH", "AVOID"]
SIG_COLOR    = {"STRONG BUY": "#00e676", "BUY": "#4caf50", "WATCH": "#ffc107", "AVOID": "#f44336"}

# Quant Thresholds
SCORE_MAX        = 15
DEFAULT_MIN_RR   = 1.5
VOL_CAP_MAX      = 0.15
VOL_CAP_MIN      = 0.015
KELLY_FRACTION   = 0.25  # Max Kelly fraction in Bull market

# Friction Model (Biaya Transaksi IDX)
SLIPPAGE_PCT = 0.002  # 0.2% slippage
COMMISSION_PCT = 0.0015 # 0.15% komisi broker

# Ichimoku
ICHI_W1, ICHI_W2, ICHI_W3 = 9, 26, 52

class SessionKeys:
    RUN_DATA = "ifh_run"
    PORTFOLIO_HEAT = "portfolio_heat"
    MANUAL_RESULT = "manual_result"
    MANUAL_ERROR = "manual_error"
    MANUAL_INPUT = "manual_ticker_input"
    APP_MODE = "app_mode"

def esc(text: str) -> str:
    return html.escape(str(text))

# ═══════════════════════════════════════════════════════
# DATACLASS
# ═══════════════════════════════════════════════════════
@dataclass
class QuantResult:
    ticker:     str
    harga:      float
    signal:     str
    score:      int
    score_max:  int
    conf:       str
    reasons:    list = field(default_factory=list)

    sh:         float = 0.0
    sl_fib:     float = 0.0
    entry_hi:   float = 0.0
    entry_lo:   float = 0.0
    target1:    float = 0.0
    target2:    float = 0.0
    stop_loss:  float = 0.0
    rr:         float = 0.0      # Gross R/R
    adj_rr:     float = 0.0      # Friction-adjusted R/R

    # Quant Metrics
    win_prob:   float = 0.0
    ev_r:       float = 0.0      # EV in R (using adj_rr)
    kelly_pct:  float = 0.0
    
    # Technicals
    atr:        float = 0.0
    atr_pct:    float = 0.0
    adv_rp:     float = 0.0
    vol_rel:    float = 0.0
    rs:         float = 0.0
    adx:        float = 0.0
    trend_bars: int   = 0
    chikou_bull: bool = False
    gate_passed: bool = True
    above_ema20: bool = False
    ma200:       float = float("nan")
    above_ma200: Optional[bool] = None
    fib_mode:    str = "bullish"
    sektor:     str   = "—"
    
    # Sizing
    lot:        int   = 0
    posisi_rp:  float = 0.0
    risk_rp:    float = 0.0
    portfolio_breach: bool = False

    _chart_data: Optional[dict] = field(default=None, repr=False, compare=False)

def effective_signal(r: QuantResult, min_rr: float) -> str:
    if r.adj_rr < min_rr or r.ev_r < 0 or r.lot == 0:
        return "AVOID"
    return r.signal

# ═══════════════════════════════════════════════════════
# DOWNLOAD LAYER
# ═══════════════════════════════════════════════════════
def _safe_download(ticker: str, period: str) -> Optional[pd.DataFrame]:
    for attempt in range(MAX_RETRY):
        try:
            if attempt > 0: time.sleep((2 ** attempt) + random.uniform(0.05, 0.4))
            df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True, actions=False)
            if df is None or df.empty: continue
            df.columns = [str(c).strip().title() for c in df.columns]
            req = {"Open", "High", "Low", "Close", "Volume"}
            if not req.issubset(df.columns): continue
            df = df[sorted(req)].dropna(subset=["Close"])
            return df if len(df) >= MIN_BARS else None
        except Exception as e:
            logger.warning(f"Download {ticker} attempt {attempt+1}: {e}")
    return None

def _batch_chunk(tickers: list, period: str) -> dict:
    results: dict = {}
    if not tickers: return results
    if len(tickers) == 1:
        df = _safe_download(tickers[0], period)
        if df is not None: results[tickers[0]] = df
        return results

    try:
        raw = yf.download(tickers, period=period, interval="1d", group_by="ticker", auto_adjust=True, progress=False, threads=False)
        if raw is None or raw.empty: raise ValueError("empty")
        if not isinstance(raw.columns, pd.MultiIndex): raise ValueError("flat cols")
        
        lvl0 = raw.columns.get_level_values(0).unique().tolist()
        lvl1 = raw.columns.get_level_values(1).unique().tolist()

        for t in tickers:
            try:
                df = raw[t].copy() if t in lvl0 else (raw.xs(t, axis=1, level=1).copy() if t in lvl1 else None)
                if df is None: continue
                df.columns = [str(c).strip().title() for c in df.columns]
                req = {"Open", "High", "Low", "Close", "Volume"}
                if not req.issubset(df.columns): continue
                df = df[sorted(req)].dropna(subset=["Close"])
                if len(df) >= MIN_BARS: results[t] = df
            except: continue
    except:
        for t in tickers:
            df = _safe_download(t, period)
            if df is not None: results[t] = df
    return results

def fetch_all(tickers: list, period: str = PERIOD, max_workers: int = 6, chunk_size: int = CHUNK_SIZE) -> dict:
    chunks = [tickers[i: i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    results: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_batch_chunk, ch, period): ch for ch in chunks}
        for fut in as_completed(futs):
            try:
                data = fut.result()
                if data: results.update(data)
            except: pass
    missing = [t for t in tickers if t not in results]
    if missing:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs2 = {ex.submit(_safe_download, t, period): t for t in missing}
            for fut in as_completed(futs2):
                t = futs2[fut]
                try:
                    df = fut.result()
                    if df is not None: results[t] = df
                except: pass
    return results

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ihsg(period: str = PERIOD) -> pd.DataFrame:
    df = _safe_download(IHSG_TICKER, period)
    return df if (df is not None and len(df) >= MIN_BARS) else pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_emiten() -> pd.DataFrame:
    try:
        df = pd.read_csv("emiten.csv")
        df.columns = [c.strip().lower() for c in df.columns]
        if "ticker" not in df.columns: raise ValueError
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        df["ticker"] = df["ticker"].apply(lambda x: x if x.endswith(".JK") else x + ".JK")
        if "sektor" not in df.columns: df["sektor"] = "—"
        return df.dropna(subset=["ticker"])
    except:
        base = ["BBCA","BMRI","BBRI","BREN","AMMN","TLKM","ASII","GOTO","MDKA","ICBP","INDF","SMGR","ADRO","UNVR","KLBF","PTBA","ANTM","EXCL","MIKA","CPIN","ITMG","BUMI","MEDC","INCO","SIDO","MTEL","ISAT","GGRM","HMSP","TOWR","WIFI","DSSA"]
        return pd.DataFrame({"ticker": [t + ".JK" for t in base], "sektor": "—"})

# ═══════════════════════════════════════════════════════
# INDICATORS & MATH
# ═══════════════════════════════════════════════════════
def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha_c = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_o = np.empty(len(df))
    ha_o[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2
    for i in range(1, len(df)): ha_o[i] = (ha_o[i-1] + ha_c.iloc[i-1]) / 2
    ha_o_s = pd.Series(ha_o, index=df.index)
    return pd.DataFrame({
        "HA_O": ha_o_s, "HA_C": ha_c, 
        "HA_H": pd.concat([ha_o_s, ha_c, df["High"]], axis=1).max(axis=1), 
        "HA_L": pd.concat([ha_o_s, ha_c, df["Low"]], axis=1).min(axis=1)
    }, index=df.index)

def stoch_rsi(close: pd.Series, rsi_p=14, stoch_p=14, k_s=3, d_s=3):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/rsi_p, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/rsi_p, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / (loss + 1e-9))
    raw_k = (rsi - rsi.rolling(stoch_p).min()) / (rsi.rolling(stoch_p).max() - rsi.rolling(stoch_p).min() + 1e-9) * 100
    k = raw_k.rolling(k_s).mean()
    return k, k.rolling(d_s).mean(), rsi

def ichimoku_manual(high, low, close, w1=ICHI_W1, w2=ICHI_W2, w3=ICHI_W3):
    tenkan = (high.rolling(w1).max() + low.rolling(w1).min()) / 2
    kijun  = (high.rolling(w2).max() + low.rolling(w2).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(w2)
    senkou_b = ((high.rolling(w3).max() + low.rolling(w3).min()) / 2).shift(w2)
    chikou_chart = close.shift(-w2)
    return dict(tenkan=tenkan, kijun=kijun, senkou_a=senkou_a, senkou_b=senkou_b, chikou_chart=chikou_chart)

def _df_hash(df: pd.DataFrame) -> int:
    try: return int(hashlib.md5(pd.util.hash_pandas_object(df[["Close", "Volume"]]).values.tobytes()).hexdigest()[:12], 16)
    except: return hash(str(df.shape))

@st.cache_data(ttl=900, show_spinner=False, max_entries=500)
def compute_indicators_cached(_hash: int, df_raw: pd.DataFrame) -> Optional[dict]:
    if len(df_raw) < MIN_BARS: return None
    c, h, lo, o, v = df_raw["Close"], df_raw["High"], df_raw["Low"], df_raw["Open"], df_raw["Volume"]
    ichi = ichimoku_manual(h, lo, c)
    ha = heikin_ashi(df_raw)  # FIX: df -> df_raw
    macd_hist = MACD(c).macd_diff()
    adx_obj = ADXIndicator(h, lo, c, window=14)
    srsi_k, srsi_d, rsi14 = stoch_rsi(c)
    
    return dict(
        close=c, high=h, low=lo, open=o, volume=v,
        tenkan=ichi["tenkan"], kijun=ichi["kijun"], senkou_a=ichi["senkou_a"], senkou_b=ichi["senkou_b"], chikou_chart=ichi["chikou_chart"],
        ha=ha, ema20=EMAIndicator(c, window=20).ema_indicator(), ema50=EMAIndicator(c, window=50).ema_indicator(), ma200=c.rolling(200).mean(),
        macd_hist=macd_hist, adx=adx_obj.adx(), di_plus=adx_obj.adx_pos(), di_minus=adx_obj.adx_neg(),
        atr=AverageTrueRange(h, lo, c, window=14).average_true_range(),
        srsi_k=srsi_k, srsi_d=srsi_d, rsi14=rsi14,
        adv20=v.rolling(20).mean(), vol_rel=v/v.rolling(20).mean().replace(0, np.nan), adv_rp=(c*v).rolling(20).mean()/1_000_000,
        cqs=((c-lo)/(h-lo).replace(0, np.nan)).fillna(0.5)
    )

def compute_indicators(df_raw: pd.DataFrame) -> Optional[dict]:
    return compute_indicators_cached(_df_hash(df_raw), df_raw)

# ═══════════════════════════════════════════════════════
# FIBONACCI & SWINGS
# ═══════════════════════════════════════════════════════
def find_swing(high: pd.Series, low: pd.Series, lookback: int = 90) -> tuple:
    n = min(lookback, len(high))
    hw, lw = high.iloc[-n:].reset_index(drop=True), low.iloc[-n:].reset_index(drop=True)
    sh_pos = int(hw.values.argmax())
    sh = float(hw.iloc[sh_pos])
    sl = float(lw.iloc[:max(sh_pos+1, min(n, sh_pos+5))].min()) if sh_pos > 0 else float(lw.min())
    if sh <= sl: sh, sl = float(hw.max()), float(lw.min())
    return sh, sl

def fib_levels(sh, sl):
    d = sh - sl
    return dict(sh=sh, sl=sl, f382=sh-0.382*d, f500=sh-0.500*d, f618=sh-0.618*d, e1272=sl+1.272*d, e1618=sl+1.618*d)

def find_swing_bearish(high, low, lookback=90):
    n = min(lookback, len(high))
    hw, lw = high.iloc[-n:].reset_index(drop=True), low.iloc[-n:].reset_index(drop=True)
    sl_pos = int(lw.values.argmin())
    sl = float(lw.iloc[sl_pos])
    sh = float(hw.iloc[:max(sl_pos+1, min(n, sl_pos+5))].max()) if sl_pos > 0 else float(hw.max())
    if sh <= sl: sh, sl = float(hw.max()), float(lw.min())
    return sh, sl

def fib_levels_bearish(sh, sl):
    d = sh - sl
    return dict(sh=sh, sl=sl, r382=sl+0.382*d, r500=sl+0.500*d, r618=sl+0.618*d, d1272=sh-1.272*d, d1618=sh-1.618*d)

def adaptive_lookback(atr_pct):
    return int(np.clip(90 * (2.0 / max(atr_pct, 0.3)), 60, 150))

# ═══════════════════════════════════════════════════════
# MARKET REGIME & RS
# ═══════════════════════════════════════════════════════
def ihsg_trend(ihsg_df):
    if ihsg_df.empty or len(ihsg_df) <= 52: return False, 1.0
    e20, e50 = ihsg_df["Close"].ewm(span=20, adjust=False).mean(), ihsg_df["Close"].ewm(span=50, adjust=False).mean()
    return float(e20.iloc[-1]) > float(e50.iloc[-1]), float(e20.iloc[-1]) / max(float(e50.iloc[-1]), 1.0)

def decide_regime(ihsg_bull, breadth):
    if ihsg_bull and breadth > 55: return "BULL"
    if not ihsg_bull and breadth < 45: return "BEAR"
    return "NEUTRAL"

def rs_score(close, ihsg_ret, window=20):
    if ihsg_ret is None: return 0.0
    common = close.index.intersection(ihsg_ret.index)
    if len(common) < window + 5: return 0.0
    rel = close.loc[common].pct_change() - ihsg_ret.loc[common]
    mu, std = rel.iloc[-window:].mean(), rel.iloc[-window:].std()
    return float(mu / std) if std > 0 and not np.isnan(std) else 0.0

# ═══════════════════════════════════════════════════════
# TRUE QUANT ENGINE
# ═══════════════════════════════════════════════════════
def _conf(s): 
    p = s / SCORE_MAX
    return "VERY HIGH" if p >= 0.8 else "HIGH" if p >= 0.6 else "MEDIUM" if p >= 0.4 else "LOW"

def _vectorized_trend_bars(close, senkou_a, senkou_b, max_bars=60):
    n = min(max_bars, len(close))
    c_rec, sa_rec, sb_rec = close.iloc[-n:], senkou_a.iloc[-n:], senkou_b.iloc[-n:]
    kumo_top = pd.concat([sa_rec, sb_rec], axis=1).max(axis=1)
    above = (c_rec > kumo_top).iloc[::-1].values
    count = 0
    for val in above:
        if val and not pd.isna(sa_rec.iloc[-1 - count]): count += 1
        else: break
    return count

def _calc_continuous_win_prob(adx, trend_bars, above_ma200, chikou_bull, rs):
    """Sigmoid-based continuous probability mapping using math.tanh."""
    p = 0.50
    p += math.tanh((adx - 20) / 4) * 0.15   # ADX smooth scaling
    p += math.tanh((trend_bars - 5) / 3) * 0.10 # Trend bars smooth scaling
    p += 0.05 if above_ma200 else -0.05
    p += 0.05 if chikou_bull else -0.05
    p += math.tanh(rs / 0.5) * 0.05        # RS smooth scaling
    return np.clip(p, 0.05, 0.90)

def _calc_friction_adj_rr(price, target, stop_loss):
    """Calculates R/R after slippage and commission friction."""
    adj_entry = price * (1 + SLIPPAGE_PCT + COMMISSION_PCT)
    adj_target = target * (1 - SLIPPAGE_PCT - COMMISSION_PCT)
    adj_sl = stop_loss * (1 + SLIPPAGE_PCT) # Assume SL fills worse
    
    risk = adj_entry - adj_sl
    reward = adj_target - adj_entry
    if risk <= 0: return 0.0
    return reward / risk

def _calc_regime_kelly(win_prob, adj_rr, atr_pct, ihsg_bull):
    """Kelly Criterion scaled by Volatility and Market Regime."""
    if adj_rr <= 0: return 0.0
    base_kelly = win_prob - ((1 - win_prob) / adj_rr)
    if base_kelly <= 0: return 0.0
    
    # 1. Volatility Scalar (Max 1.0 if ATR 1%, drops to 0.2 if ATR 10%)
    vol_scalar = np.clip(2.0 / max(atr_pct, 0.5), 0.2, 1.0)
    
    # 2. Regime Scalar (Stop new buys in Bear, full size in Bull)
    regime_scalar = 1.0 if ihsg_bull else 0.0
    
    return max(0.0, base_kelly * KELLY_FRACTION * vol_scalar * regime_scalar)

def screen_one(ticker, df_raw, ihsg_ret, min_adv_rp, ihsg_bull, sektor="—", modal_rp=100_000_000, risk_pct=0.01, max_exposure_pct=0.25, require_gate=True) -> Optional[QuantResult]:
    ind = compute_indicators(df_raw)
    if ind is None: return None

    price = float(ind["close"].iloc[-1])
    if price <= 0: return None

    adv_rp_v = float(ind["adv_rp"].iloc[-1])
    if np.isnan(adv_rp_v) or (require_gate and adv_rp_v < min_adv_rp): return None

    atr_v = float(ind["atr"].iloc[-1])
    atr_pct = atr_v / price * 100
    sa_v, sb_v = float(ind["senkou_a"].iloc[-1]), float(ind["senkou_b"].iloc[-1])
    if pd.isna(sa_v) or pd.isna(sb_v): return None

    kijun_v = float(ind["kijun"].iloc[-1])
    ha_bull = float(ind["ha"].iloc[-1]["HA_C"]) > float(ind["ha"].iloc[-1]["HA_O"])
    above_cloud = price > max(sa_v, sb_v)
    gate_passed = above_cloud and ha_bull

    if require_gate and not gate_passed: return None

    score = 0
    reasons = []
    
    # A. ICHIMOKU (5)
    if sa_v > sb_v: score += 1; reasons.append("🟢 Bullish Cloud")
    if float(ind["tenkan"].iloc[-1]) > kijun_v: score += 1; reasons.append("🟢 TK Cross Bullish")
    if price > kijun_v: score += 1; reasons.append("🟢 Price > Kijun")
    if kijun_v > float(ind["kijun"].iloc[-6]): score += 1; reasons.append("🟢 Kijun Slope Up")
    
    chikou_bull = len(ind["close"]) > ICHI_W2 and float(ind["close"].iloc[-1]) > float(ind["close"].iloc[-(ICHI_W2+1)])
    if chikou_bull: score += 1; reasons.append("🟢 Chikou Clear")
    
    trend_bars = _vectorized_trend_bars(ind["close"], ind["senkou_a"], ind["senkou_b"])

    # B. TREND (3)
    adx_v = float(ind["adx"].iloc[-1])
    if adx_v >= 25: score += 1; reasons.append(f"🟢 ADX Strong ({adx_v:.0f})")
    elif adx_v >= 20: score += 1; reasons.append(f"🟡 ADX Building ({adx_v:.0f})")
    
    if float(ind["di_plus"].iloc[-1]) > float(ind["di_minus"].iloc[-1]): score += 1; reasons.append("🟢 DI+ > DI-")
    if float(ind["ema20"].iloc[-1]) > float(ind["ema50"].iloc[-1]): score += 1; reasons.append("🟢 EMA20 > EMA50")

    # C. MOMENTUM (4)
    macd_now, macd_prev = float(ind["macd_hist"].iloc[-1]), float(ind["macd_hist"].iloc[-2])
    if macd_now > 0 and macd_prev <= 0: score += 2; reasons.append("🔥 MACD Fresh Cross")
    elif macd_now > 0 and macd_now > macd_prev: score += 1; reasons.append("🟢 MACD Rising")
    
    sk_v, sd_v = float(ind["srsi_k"].iloc[-1]), float(ind["srsi_d"].iloc[-1])
    if sk_v > sd_v and sk_v < 50: score += 1; reasons.append("🟢 StochRSI Cross")
    if (float(ind["ha"].iloc[-1]["HA_O"]) - float(ind["ha"].iloc[-1]["HA_L"])) < atr_v * 0.3: score += 1; reasons.append("🔥 HA No Lower Wick")

    # D. VOLUME (2)
    vol_r = float(ind["vol_rel"].iloc[-1])
    if vol_r >= 2.0: score += 1; reasons.append(f"🔥 Volume Spike {vol_r:.1f}x")
    elif vol_r >= 1.2: score += 1; reasons.append(f"🟢 Volume Above Avg {vol_r:.1f}x")
    if adv_rp_v >= 10_000: score += 1; reasons.append("🟢 High Liquidity")

    # E. RS (1)
    rs_v = rs_score(ind["close"], ihsg_ret)
    if rs_v > 0.5: score += 1; reasons.append(f"🟢 Outperform IHSG ({rs_v:+.2f})")

    # MA200 Context
    ma200_v = float(ind["ma200"].iloc[-1])
    above_ma200 = None if pd.isna(ma200_v) else price > ma200_v
    if above_ma200: reasons.append("🟢 Above MA200")
    elif above_ma200 is False: reasons.append("🔴 Below MA200")

    if not gate_passed:
        reasons.insert(0, "🔴 GATE FAILED: Not above cloud / HA Red")
        signal = "WATCH"
        fib_res = _compute_fib_bearish(ind, price, atr_v)
    else:
        fib_res = _compute_fib_bullish(ind, price, atr_v, kijun_v)
        signal = "AVOID" if (adx_v < 20 and _conf(score) == "LOW") else {"VERY HIGH": "STRONG BUY", "HIGH": "BUY", "MEDIUM": "WATCH"}.get(_conf(score), "AVOID")

    # ── TRUE QUANT METRICS ──
    win_prob = _calc_continuous_win_prob(adx_v, trend_bars, above_ma200, chikou_bull, rs_v)
    
    # Friction-Adjusted R/R
    adj_rr = _calc_friction_adj_rr(price, fib_res["target1"], fib_res["stop_loss"])
    
    # EV uses Adj R/R
    ev_r = (win_prob * adj_rr) - ((1 - win_prob) * 1.0)
    
    # Regime-Adjusted Kelly
    kelly_pct = _calc_regime_kelly(win_prob, adj_rr, atr_pct, ihsg_bull)
    
    # Strict Invalidation Volatility Cap
    sl_dist = abs(price - fib_res["stop_loss"]) / price
    if sl_dist > VOL_CAP_MAX or sl_dist < VOL_CAP_MIN:
        signal = "AVOID"
        reasons.append(f"⚠️ Volatility Cap Failed (SL dist {sl_dist*100:.1f}%)")
        kelly_pct = 0.0

    if ev_r < 0:
        signal = "AVOID"
        reasons.append(f"❌ Negative Expected Value ({ev_r:.2f}R Net)")
        
    if kelly_pct == 0 and ihsg_bull:
        signal = "AVOID"
        reasons.append("❌ Kelly Criterion = 0% (Math says do not trade)")

    # ── POSITION SIZING ──
    dynamic_risk = min(risk_pct, kelly_pct) if kelly_pct > 0 else 0.0
    risk_rp = modal_rp * dynamic_risk
    risk_per_sh = abs(price * (1 + SLIPPAGE_PCT) - fib_res["stop_loss"] * (1 + SLIPPAGE_PCT))
    lot_by_risk = int(risk_rp / (risk_per_sh * SHARES_PER_LOT)) if risk_per_sh > 0 else 0
    lot_by_expo = int((modal_rp * max_exposure_pct) / (price * SHARES_PER_LOT))
    lot = max(0, min(lot_by_risk, lot_by_expo, MAX_LOT_CAP))
    posisi_rp = lot * SHARES_PER_LOT * price

    # Chart Data Slice
    n = len(df_raw)
    s = slice(n - min(DISPLAY_BARS, n), n)
    chart_data = {k: v.iloc[s] if hasattr(v, 'iloc') else v for k, v in {
        "dt": df_raw.index[s], "open": df_raw["Open"].iloc[s], "close": df_raw["Close"].iloc[s],
        "high": df_raw["High"].iloc[s], "low": df_raw["Low"].iloc[s], "volume": df_raw["Volume"].iloc[s],
        "senkou_a": ind["senkou_a"].iloc[s], "senkou_b": ind["senkou_b"].iloc[s],
        "ema20": ind["ema20"].iloc[s], "ema50": ind["ema50"].iloc[s], "ma200": ind["ma200"].iloc[s],
        "adv20": ind["adv20"].iloc[s], "chikou": ind["chikou_chart"].iloc[s]
    }.items()}

    return QuantResult(
        ticker=ticker, harga=price, signal=signal, score=score, score_max=SCORE_MAX, conf=_conf(score), reasons=reasons,
        sh=fib_res["sh"], sl_fib=fib_res["sl_fib"], entry_hi=fib_res["entry_hi"], entry_lo=fib_res["entry_lo"],
        target1=fib_res["target1"], target2=fib_res["target2"], stop_loss=fib_res["stop_loss"], rr=fib_res["rr"],
        adj_rr=adj_rr, win_prob=win_prob, ev_r=ev_r, kelly_pct=kelly_pct,
        atr=atr_v, atr_pct=atr_pct, adv_rp=adv_rp_v, vol_rel=vol_r, rs=rs_v, adx=adx_v, trend_bars=trend_bars,
        chikou_bull=chikou_bull, gate_passed=gate_passed, above_ema20=price > float(ind["ema20"].iloc[-1]),
        ma200=ma200_v if not pd.isna(ma200_v) else float("nan"), above_ma200=above_ma200, fib_mode=fib_res["fib_mode"],
        sektor=sektor, lot=lot, posisi_rp=posisi_rp, risk_rp=risk_rp, _chart_data=chart_data
    )

def _compute_fib_bullish(ind, price, atr_v, kijun_v):
    sh, sl = find_swing(ind["high"], ind["low"], adaptive_lookback(atr_v/price*100))
    fib = fib_levels(sh, sl)
    e20 = float(ind["ema20"].iloc[-1])
    entry_lo, entry_hi = fib["f500"], fib["f382"]
    if entry_lo <= e20 <= entry_hi:
        entry_lo = max(entry_lo, e20 * 0.99)
        entry_hi = min(entry_hi, e20 * 1.01)
    stop_loss = max(kijun_v, fib["f618"])
    target1, target2 = fib["e1272"], fib["e1618"]
    entry_ref = max(price, (entry_lo + entry_hi) / 2)
    rr = (target1 - entry_ref) / max(entry_ref - stop_loss, 1.0)
    return dict(sh=sh, sl_fib=sl, entry_hi=entry_hi, entry_lo=entry_lo, target1=target1, target2=target2, stop_loss=stop_loss, rr=rr, fib_mode="bullish")

def _compute_fib_bearish(ind, price, atr_v):
    sh, sl = find_swing_bearish(ind["high"], ind["low"], adaptive_lookback(atr_v/price*100))
    fibb = fib_levels_bearish(sh, sl)
    stop_loss = fibb["r618"]
    target1, target2 = fibb["d1272"], fibb["d1618"]
    rr = max(sh - stop_loss, 0.0) / max(stop_loss - sl, 1.0)
    return dict(sh=sh, sl_fib=sl, entry_hi=fibb["r500"], entry_lo=fibb["r382"], target1=target1, target2=target2, stop_loss=stop_loss, rr=rr, fib_mode="bearish")

# ═══════════════════════════════════════════════════════
# UI RENDERING
# ═══════════════════════════════════════════════════════
def build_chart(res: QuantResult) -> go.Figure:
    cd = res._chart_data
    if cd is None: return go.Figure()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=cd["dt"], open=cd["open"], high=cd["high"], low=cd["low"], close=cd["close"], name=esc(res.ticker), increasing_fillcolor="#00C853", decreasing_fillcolor="#FF3B30"), row=1, col=1)
    fig.add_trace(go.Scatter(x=cd["dt"], y=cd["senkou_a"], line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=cd["dt"], y=cd["senkou_b"], line=dict(width=0), fill="tonexty", fillcolor="rgba(0,200,83,0.13)" if res.signal in ("STRONG BUY", "BUY") else "rgba(255,59,48,0.13)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=cd["dt"], y=cd["ema20"], line=dict(color="#F0B90B", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=cd["dt"], y=cd["ema50"], line=dict(color="#848E9C", width=1, dash="dot")), row=1, col=1)
    if cd["ma200"].notna().any(): fig.add_trace(go.Scatter(x=cd["dt"], y=cd["ma200"], line=dict(color="#FF6EC7", width=1.4, dash="dashdot")), row=1, col=1)
    
    for val, col, lbl in [(res.entry_hi, "#00C853", f"Entry: {res.entry_hi:,.0f}"), (res.stop_loss, "#FF3B30", f"SL: {res.stop_loss:,.0f}"), (res.target1, "#F0B90B", f"T1: {res.target1:,.0f}")]:
        fig.add_hline(y=val, row=1, col=1, line=dict(color=col, dash="dash"), annotation_text=f" {lbl}", annotation_font=dict(color=col, size=9))

    vcol = ["#00C853" if float(cd["close"].iloc[i]) >= float(cd["open"].iloc[i]) else "#FF3B30" for i in range(len(cd["volume"]))]
    fig.add_trace(go.Bar(x=cd["dt"], y=cd["volume"], marker_color=vcol, opacity=0.65), row=2, col=1)
    
    sc = SIG_COLOR.get(res.signal, "#FAFAFA")
    fig.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#1E2329", font=dict(color="#FAFAFA", size=11), height=520, margin=dict(l=0, r=140, t=36, b=0), xaxis_rangeslider_visible=False,
                      title=dict(text=f"<b>{esc(res.ticker)}</b> {res.signal} | Net R/R {res.adj_rr:.1f} | EV {res.ev_r:.2f}R | P_win {res.win_prob*100:.0f}% | Lot {res.lot}", x=0.01, font=dict(color=sc, size=13)))
    return fig

def render_analysis_content(res: QuantResult, modal_rp=100_000_000, max_expo=0.25):
    if not res.gate_passed:
        st.warning("⚠️ **Gate Gagal.** Analisis di bawah adalah Peta Breakdown, bukan sinyal beli aktif.")

    sc = SIG_COLOR.get(res.signal, "#FAFAFA")
    st.markdown(f"<h3 style='margin:0;'>{esc(res.ticker)} <span style='color:{sc};font-size:.95rem;'>{res.signal}</span></h3><p style='color:#848E9C;margin:4px 0 10px;'>Score <b>{res.score}/{res.score_max}</b> · P_win <b>{res.win_prob*100:.0f}%</b> · Net EV <b>{res.ev_r:+.2f}R</b> · Adj R/R <b>{res.adj_rr:.1f}:1</b></p>", unsafe_allow_html=True)
    
    st.plotly_chart(build_chart(res), use_container_width=True, config={"displayModeBar": False})

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Harga", f"{res.harga:,.0f}")
    c2.metric("Entry Zone", f"{res.entry_lo:,.0f}–{res.entry_hi:,.0f}")
    c3.metric("Stop Loss", f"{res.stop_loss:,.0f}", delta=f"{(res.stop_loss/res.harga-1)*100:.1f}%", delta_color="inverse")
    c4.metric("Target 1", f"{res.target1:,.0f}", delta=f"+{(res.target1/res.harga-1)*100:.1f}%")
    c5.metric("Target 2", f"{res.target2:,.0f}", delta=f"+{(res.target2/res.harga-1)*100:.1f}%")

    st.markdown("---")
    col_r, col_p = st.columns(2)
    with col_r:
        st.markdown("#### 🧠 Analisis Kuantitatif")
        for r in res.reasons: st.write(r)
    with col_p:
        st.markdown("#### 🎯 Quant Trading Plan")
        st.markdown(f"""
**💰 Position Sizing (Regime-Kelly):**
- Kelly Frac: `{res.kelly_pct*100:.1f}%`
- Risk/Capital: `{res.risk_rp/modal_rp*100:.2f}%`
- **Lot Size: {res.lot} lot** ({res.lot * SHARES_PER_LOT:,} lembar)
- Eksposur: Rp{res.posisi_rp/1_000_000:,.1f}jt
- Risk Rp: Rp{res.risk_rp/1_000_000:,.1f}jt

**📊 Risk Metrics (Friction-Adjusted):**
- Gross R/R: {res.rr:.1f}:1
- Net R/R: {res.adj_rr:.1f}:1 *(setelah slippage 0.2% & komisi 0.15%)*
- ATR Harian: {res.atr:,.0f} ({res.atr_pct:.1f}%)
- ADX: {res.adx:.1f}
- RS vs IHSG: {res.rs:+.2f}
""")

@st.dialog("📊 Trading Plan Detail", width="large")
def show_modal(res: QuantResult, modal_rp=100_000_000, max_expo=0.25):
    render_analysis_content(res, modal_rp, max_expo)

def render_regime(regime: str, breadth: float, ratio: float):
    cls_map = {"BULL": "regime-bull", "BEAR": "regime-bear"}
    col_map = {"BULL": "#00e676", "BEAR": "#f44336"}
    cls = cls_map.get(regime, "regime-neutral")
    color = col_map.get(regime, "#ffc107")
    trend = "🔼 EMA20 > EMA50" if ratio > 1 else "🔽 EMA20 < EMA50"
    st.markdown(f"""
    <div class='{cls}'>
      <span style='font-size:1.05rem;font-weight:700;color:{color};'>
        📊 Market Regime: {regime}
      </span>&nbsp;&nbsp;
      <span style='color:#848E9C;font-size:.88rem;'>
        Breadth: <b style='color:#FAFAFA;'>{breadth:.1f}%</b> saham di atas EMA20
        &nbsp;·&nbsp; IHSG: {trend}
      </span>
    </div>
    """, unsafe_allow_html=True)

def render_kpi(results: list, min_rr: float):
    by = {s: 0 for s in SIGNAL_ORDER}
    for r in results:
        by[effective_signal(r, min_rr)] += 1
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, lbl, val, bc in [
        (c1, "Total Lolos Gate",  str(len(results)),         "#F0B90B"),
        (c2, "STRONG BUY",        str(by["STRONG BUY"]),     "#00e676"),
        (c3, "BUY",               str(by["BUY"]),             "#4caf50"),
        (c4, "WATCH",             str(by["WATCH"]),           "#ffc107"),
        (c5, "AVOID",             str(by["AVOID"]),           "#f44336"),
    ]:
        col.markdown(f"<div class='kpi' style='border-color:{bc};'><div class='kpi-lbl'>{lbl}</div><div class='kpi-val' style='color:{bc};'>{val}</div></div>", unsafe_allow_html=True)

def results_to_df(results_with_sig: list) -> pd.DataFrame:
    rows = []
    for r, eff_sig in results_with_sig:
        rows.append({
            "Ticker": r.ticker, "Signal": eff_sig, "Score": f"{r.score}/{r.score_max}",
            "P_win": f"{r.win_prob*100:.0f}%", "Net EV (R)": f"{r.ev_r:+.2f}",
            "Net R/R": f"{r.adj_rr:.1f}", "Harga": f"{r.harga:,.0f}", 
            "Target 1": f"{r.target1:,.0f}", "Stop Loss": f"{r.stop_loss:,.0f}",
            "Lot": r.lot, "Risk Rp(jt)": f"{r.risk_rp/1_000_000:,.1f}",
            "Sektor": r.sektor,
        })
    return pd.DataFrame(rows)

def render_manual_analysis(sek_map: dict, modal_rp: float, risk_pct: float, max_expo: float):
    st.markdown("### 🔍 Analisa Manual — Cek Saham Spesifik")
    st.markdown("<p style='color:#848E9C;'>Masukkan kode ticker APAPUN. Gate wajib tidak diberlakukan di sini.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    with col1:
        ticker_raw = st.text_input("Kode Ticker", placeholder="Contoh: BBCA, TLKM", key=SessionKeys.MANUAL_INPUT, label_visibility="collapsed")
    with col2:
        analyze_btn = st.button("🔎 Analisa", use_container_width=True, type="primary")

    if analyze_btn and ticker_raw and ticker_raw.strip():
        ticker = ticker_raw.strip().upper()
        if not ticker.endswith(".JK"): ticker += ".JK"

        with st.spinner(f"📡 Mengunduh & menganalisis {esc(ticker)}..."):
            df = _safe_download(ticker, PERIOD)
            ihsg_df = fetch_ihsg()
            ihsg_bull, _ = ihsg_trend(ihsg_df)
            if df is None:
                st.session_state[SessionKeys.MANUAL_RESULT] = None
                st.session_state[SessionKeys.MANUAL_ERROR] = f"❌ Data untuk **{esc(ticker)}** tidak ditemukan."
            else:
                ihsg_close = ihsg_df["Close"] if not ihsg_df.empty else None
                ihsg_ret = ihsg_close.pct_change() if ihsg_close is not None else None
                sektor = sek_map.get(ticker, "—")

                r = screen_one(ticker, df, ihsg_ret, min_adv_rp=0, ihsg_bull=ihsg_bull, sektor=sektor,
                               modal_rp=modal_rp, risk_pct=risk_pct, max_exposure_pct=max_expo, require_gate=False)
                if r is None:
                    st.session_state[SessionKeys.MANUAL_RESULT] = None
                    st.session_state[SessionKeys.MANUAL_ERROR] = f"❌ Data **{esc(ticker)}** tidak cukup untuk dianalisis."
                else:
                    st.session_state[SessionKeys.MANUAL_RESULT] = r
                    st.session_state[SessionKeys.MANUAL_ERROR] = None

    if st.session_state.get(SessionKeys.MANUAL_ERROR):
        st.error(st.session_state[SessionKeys.MANUAL_ERROR])

    res = st.session_state.get(SessionKeys.MANUAL_RESULT)
    if res is not None:
        st.markdown("---")
        render_analysis_content(res, modal_rp, max_expo)
    elif not st.session_state.get(SessionKeys.MANUAL_ERROR):
        st.info("💡 Masukkan kode ticker di atas dan klik **Analisa** untuk memulai.")

def render_sidebar(all_tickers: list) -> dict:
    with st.sidebar:
        st.markdown("## ⚙️ Konfigurasi Quant v9.0")
        mode = st.radio("Mode", ["🚀 Screener Massal", "🔍 Analisa Manual"], key=SessionKeys.APP_MODE)
        st.markdown("---")

        config = {"mode": mode}

        if mode == "🚀 Screener Massal":
            config["n_tickers"] = st.slider("Jumlah emiten (Top N)", 10, len(all_tickers), min(200, len(all_tickers)), 10)
            config["min_adv_rp"] = st.slider("Min Likuiditas (Rp jt/hari)", 500, 20_000, 2_000, 500)
            config["min_rr"] = st.slider("Min Net R/R Ratio", 0.5, 4.0, DEFAULT_MIN_RR, 0.5)
            config["workers"] = st.slider("Parallel workers", 2, 12, 6, 1)
            config["chunk"] = st.slider("Ticker per request", 5, 50, CHUNK_SIZE, 5)
            st.markdown("---")
        else:
            st.caption("🔍 **Mode Analisa Manual** — masukkan ticker di halaman utama.")
            st.markdown("---")

        st.markdown("**Position Sizing**")
        config["modal_rp"] = st.number_input("Modal (Rp)", value=100_000_000, step=10_000_000, format="%d")
        config["risk_pct"] = st.slider("Max Risk per trade (%)", 0.5, 3.0, 1.0, 0.5) / 100
        config["max_expo"] = st.slider("Max eksposur per saham (%)", 10, 50, 25, 5) / 100
        
        # Portfolio Heat Cap
        config["max_port_risk"] = st.slider("Max Portfolio Heat (%)", 2.0, 15.0, 6.0, 0.5) / 100

        if mode == "🚀 Screener Massal":
            st.markdown("---")
            st.markdown("**Filter Tampilan**")
            config["show_sigs"] = st.multiselect("Tampilkan sinyal:", SIGNAL_ORDER, default=["STRONG BUY", "BUY", "WATCH"])
            st.markdown("---")

        st.markdown(f"<span style='color:#848E9C;font-size:.78rem;'>v9.0 True Quant · {len(all_tickers)} emiten loaded</span>", unsafe_allow_html=True)

        if mode == "🚀 Screener Massal":
            config["run_btn"] = st.button("🚀 Jalankan Screener", use_container_width=True, type="primary")
        else:
            config["run_btn"] = False

    return config

def run_screener_pipeline(target: list, sek_map: dict, config: dict) -> Optional[dict]:
    with st.spinner("📡 Mengunduh data IHSG..."):
        ihsg_df = fetch_ihsg()
        ihsg_close = ihsg_df["Close"] if not ihsg_df.empty else None
        ihsg_ret = ihsg_close.pct_change() if ihsg_close is not None else None
        ihsg_bull, _ = ihsg_trend(ihsg_df)

    if ihsg_df.empty:
        st.warning("⚠️ IHSG tidak berhasil diunduh — RS score tidak tersedia.")

    info = st.empty()
    info.info(f"⏳ Mengunduh {len(target)} emiten ({config['chunk']} ticker/request, {config['workers']} workers)...")

    t0 = time.time()
    all_dfs = fetch_all(target, PERIOD, max_workers=config["workers"], chunk_size=config["chunk"])
    elapsed = time.time() - t0
    n_ok = len(all_dfs)

    if n_ok == 0:
        info.error("❌ Tidak ada data berhasil diunduh. Cek koneksi atau rate-limit yfinance.")
        return None

    info.success(f"✅ {n_ok}/{len(target)} emiten berhasil diunduh dalam {elapsed:.1f}s")

    def _worker(args):
        ticker, df = args
        sektor = sek_map.get(ticker, "—")
        try:
            r = screen_one(ticker, df, ihsg_ret, config["min_adv_rp"], ihsg_bull, sektor,
                           modal_rp=config["modal_rp"], risk_pct=config["risk_pct"], max_exposure_pct=config["max_expo"])
        except Exception as e:
            logger.error(f"screen_one {ticker}: {e}")
            r = None
        above_ema = r.above_ema20 if r is not None else False
        return r, above_ema

    prog = st.progress(0.0, text="Analisis paralel...")
    raw_results: list = []
    above_count = total_breadth = 0
    items = list(all_dfs.items())

    with ThreadPoolExecutor(max_workers=config["workers"]) as ex:
        futs = {ex.submit(_worker, item): item[0] for item in items}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                r, above_ema = fut.result()
                total_breadth += 1
                if above_ema: above_count += 1
                if r is not None: raw_results.append(r)
            except Exception as e:
                logger.warning(f"Worker error: {e}")
            prog.progress(done / len(futs), text=f"Analisis {done}/{len(futs)}...")
    prog.empty()

    breadth_inline = (above_count / total_breadth * 100) if total_breadth > 0 else 50.0
    ihsg_bull, ihsg_ratio = ihsg_trend(ihsg_df)
    regime = decide_regime(ihsg_bull, breadth_inline)

    # ── PORTFOLIO HEAT CAP CALCULATION ──
    # Sort by EV descending to prioritize best trades
    raw_results.sort(key=lambda x: x.ev_r, reverse=True)
    
    max_port_risk_rp = config["modal_rp"] * config["max_port_risk"]
    current_port_risk = 0.0
    
    for r in raw_results:
        if r.signal in ("STRONG BUY", "BUY", "WATCH") and r.risk_rp > 0:
            if current_port_risk + r.risk_rp > max_port_risk_rp:
                # Truncate or zero out position
                remaining_risk = max_port_risk_rp - current_port_risk
                if remaining_risk > 0:
                    # Resize lot to fit remaining portfolio heat
                    r.lot = int(remaining_risk / (abs(r.harga - r.stop_loss) * SHARES_PER_LOT))
                    r.risk_rp = r.lot * abs(r.harga - r.stop_loss) * SHARES_PER_LOT
                    r.posisi_rp = r.lot * r.harga * SHARES_PER_LOT
                    current_port_risk += r.risk_rp
                    r.portfolio_breach = True
                    r.reasons.append(f"⚠️ Portfolio Heat Cap Active: Lot dipangkas ke {r.lot} (sisa risk portofolio).")
                else:
                    # No risk left
                    r.lot = 0
                    r.posisi_rp = 0.0
                    r.risk_rp = 0.0
                    r.portfolio_breach = True
                    if r.signal in ("STRONG BUY", "BUY"): 
                        r.signal = "AVOID"
                        r.reasons.append("❌ Portfolio Heat Cap Penuh: Tidak ada alokasi risiko tersisa.")
            else:
                current_port_risk += r.risk_rp

    passed_tickers = {r.ticker for r in raw_results if r.lot > 0 or not r.gate_passed}
    slim_dfs = {t: all_dfs[t] for t in passed_tickers if t in all_dfs}

    return dict(raw_results=raw_results, all_dfs=slim_dfs, regime=regime, breadth=breadth_inline,
                ihsg_ratio=ihsg_ratio, n_ok=n_ok, n_target=len(target), elapsed=elapsed, total_risk=current_port_risk)

def render_results(run_data: dict, config: dict):
    raw_results = run_data["raw_results"]
    all_dfs = run_data["all_dfs"]
    min_rr = config["min_rr"]
    show_sigs = config["show_sigs"]
    modal_rp = config["modal_rp"]
    max_expo = config["max_expo"]

    render_regime(run_data["regime"], run_data["breadth"], run_data["ihsg_ratio"])

    results_eff = [(r, effective_signal(r, min_rr)) for r in raw_results]
    filtered = [(r, eff) for r, eff in results_eff if eff in show_sigs]
    filtered.sort(key=lambda pair: (SIGNAL_ORDER.index(pair[1]), -pair[0].score))

    st.markdown("")
    render_kpi(raw_results, min_rr)
    
    total_risk_rp = run_data.get("total_risk", 0.0)
    st.markdown(f"### 📋 Hasil — **{len(filtered)}** kandidat tampil  <span style='color:#848E9C;font-size:.85rem;'>({len(raw_results)} lolos gate dari {run_data['n_ok']} diunduh) | Total Portfolio Risk: **Rp{total_risk_rp/1_000_000:.1f}jt** ({total_risk_rp/modal_rp*100:.1f}%)</span>", unsafe_allow_html=True)

    if not filtered:
        st.warning("Tidak ada saham yang memenuhi kriteria. Coba turunkan filter.")
        return

    df_disp = results_to_df(filtered)
    st.dataframe(df_disp, use_container_width=True, hide_index=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = df_disp.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download hasil (.csv)", csv, file_name=f"ifh_quant_v9_{ts}.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("### 🔍 Action Panel — Klik ticker untuk Trading Plan lengkap")

    for sig in SIGNAL_ORDER:
        tier = [r for r, eff in filtered if eff == sig]
        if not tier: continue
        avg_score = sum(r.score for r in tier) / len(tier)
        exp_label = f"{sig}  ({len(tier)} saham)  —  avg score {avg_score:.1f}/{tier[0].score_max}"
        
        with st.expander(exp_label, expanded=(sig in ("STRONG BUY", "BUY"))):
            NCOLS = 5
            rows = [tier[i: i + NCOLS] for i in range(0, len(tier), NCOLS)]
            for row in rows:
                cols = st.columns(NCOLS)
                for j, r in enumerate(row):
                    with cols[j]:
                        btn_label = f"**{esc(r.ticker)}**\nRp{r.harga:,.0f} | Lot: {r.lot}\nNet EV {r.ev_r:+.2f}R"
                        if r.portfolio_breach:
                            btn_label = "⚠️ CAP TRUNCATED\n" + btn_label
                            
                        if st.button(btn_label, key=f"btn_{r.ticker}_{sig}", use_container_width=True,
                                     help=f"Net R/R: {r.adj_rr:.1f} | P_win: {r.win_prob*100:.0f}% | ADX {r.adx:.0f}"):
                            show_modal(r, modal_rp, max_expo)

# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def main():
    st.markdown("<h1 style='margin-bottom:2px;'>🦅 Ichi-Fibo-Heikin Pro v9.0 (True Quant)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#848E9C;margin-top:0;'>Quant Trading System · Continuous Probabilities · Friction Model · Portfolio Heat · Kelly Criterion</p>", unsafe_allow_html=True)
    st.markdown("---")

    emiten_df = load_emiten()
    all_tickers = emiten_df["ticker"].tolist()
    sek_col = emiten_df["sektor"] if "sektor" in emiten_df.columns else pd.Series("—", index=emiten_df.index)
    sek_map = dict(zip(emiten_df["ticker"], sek_col.fillna("—")))

    config = render_sidebar(all_tickers)

    if config["mode"] == "🔍 Analisa Manual":
        render_manual_analysis(sek_map, config["modal_rp"], config["risk_pct"], config["max_expo"])
        return

    if config["run_btn"]:
        target = all_tickers[:config["n_tickers"]]
        run_data = run_screener_pipeline(target, sek_map, config)
        if run_data is not None:
            st.session_state[SessionKeys.RUN_DATA] = run_data

    run_data = st.session_state.get(SessionKeys.RUN_DATA)
    if run_data is None:
        st.markdown("### 👆 Klik **Jalankan Screener** di sidebar untuk memulai.")
        return

    render_results(run_data, config)

if __name__ == "__main__":
    main()
