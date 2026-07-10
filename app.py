"""
Screener Ichi-Fibo-Heikin Pro  •  v6.6  (Kalkulator Average Down)
════════════════════════════════════════════════════════════════════
FITUR BARU v6.6:
  "💰 Kalkulator Average Down" — expander opt-in di render_analysis_content(),
  jadi otomatis tersedia di KEDUA tempat (modal Action Panel Screener
  Massal, DAN halaman Analisa Manual) — termasuk utk saham yang TIDAK
  lolos gate teknikal (fib_mode="bearish"), sesuai permintaan: "scan
  manual atau meskipun saham itu tidak lolos skrining".

  Input: lot yang sudah dipegang + harga rata-rata beli (user isi sendiri,
  bukan dari data — average price historis user tidak ada di data pasar).
  Output: floating P/L saat ini, average price baru & breakeven % kalau
  nambah N lot di harga sekarang (slider interaktif), potensi return ke
  Target 1 (mode bullish) atau ke zona Resistance (mode bearish), cross-
  check total eksposur terhadap batas % modal yang di-set di sidebar
  (Position Sizing), dan catatan kontekstual dari gate_passed/fib_mode
  (TIDAK memaksakan rekomendasi "harus/jangan" — murni kalkulator +
  konteks, keputusan tetap di tangan user, sesuai <legal_and_financial_advice>).

  render_analysis_content()/show_modal() sekarang menerima modal_rp &
  max_expo (sudah ada di sidebar Position Sizing sejak v6.4/v6.5, kini
  di-thread ke modal juga, sebelumnya cuma dipakai di screen_one()).

Semua fix v6.1–v6.5 tetap berlaku tanpa perubahan: Ichimoku shift
terverifikasi empiris, session_state architecture fix, find_swing
peak-then-prior-low, stop-loss ATR-minimum, position sizing exposure
cap, RSI Wilder's EMA, HA rekursif, mode Analisa Manual, MA200 context,
Peta Breakdown (Fibo bearish), dsb.
"""


from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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
# PAGE CONFIG — wajib paling atas sebelum st lain
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="IFH Pro v6.6",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🦅",
)

# ═══════════════════════════════════════════════════════
# CSS DARK MODE
# ═══════════════════════════════════════════════════════
st.markdown("""
<style>
  .stApp { background:#0E1117; color:#FAFAFA; }
  .block-container { padding-top:1.2rem; }
  .kpi {
    background:#1E2329; border-radius:10px; padding:12px 16px;
    border-left:4px solid #F0B90B; margin-bottom:8px;
  }
  .kpi-lbl { font-size:.78rem; color:#848E9C; margin-bottom:2px; }
  .kpi-val { font-size:1.3rem; font-weight:700; color:#FAFAFA; }
  .regime-bull {
    background:#0a2f1a; border-left:5px solid #00e676;
    padding:10px 16px; border-radius:8px; margin-bottom:12px;
  }
  .regime-bear {
    background:#2f0a0a; border-left:5px solid #f44336;
    padding:10px 16px; border-radius:8px; margin-bottom:12px;
  }
  .regime-neutral {
    background:#2a2a0a; border-left:5px solid #ffc107;
    padding:10px 16px; border-radius:8px; margin-bottom:12px;
  }
  hr { border-color:#2a2d35; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# KONSTANTA
# ═══════════════════════════════════════════════════════
IHSG_TICKER = "^JKSE"
PERIOD      = "2y"      # ~500 bar — dinaikkan dari 1y agar MA200 (butuh 200 bar) punya buffer aman
MIN_BARS    = 90        # minimum utk analisis jalan; MA200 sendiri butuh 200 bar (ditangani terpisah, graceful)
CHUNK_SIZE  = 20
MAX_RETRY   = 3

SIGNAL_ORDER = ["STRONG BUY", "BUY", "WATCH", "AVOID"]
SIG_COLOR    = {
    "STRONG BUY": "#00e676",
    "BUY":        "#4caf50",
    "WATCH":      "#ffc107",
    "AVOID":      "#f44336",
}


# ═══════════════════════════════════════════════════════
# DATACLASS
# ═══════════════════════════════════════════════════════
@dataclass
class SR:
    ticker:     str
    harga:      float
    signal:     str          # sinyal ASLI hasil scoring (immutable, tidak dimutasi UI)
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
    rr:         float = 0.0

    atr:        float = 0.0
    atr_pct:    float = 0.0
    adv_rp:     float = 0.0
    vol_rel:    float = 0.0
    rs:         float = 0.0
    adx:        float = 0.0
    trend_bars: int   = 0
    cqs:        float = 0.0
    chikou_bull: bool = False
    gate_passed: bool = True   # False jika lolos via require_gate=False (mode Analisa Manual)
    ma200:       float = float("nan")           # NaN jika history <200 bar
    above_ma200: Optional[bool] = None          # None = belum bisa ditentukan (data kurang)
    fib_mode:    str = "bullish"                # "bullish" (default) atau "bearish" (breakdown map)
    sektor:     str   = "—"
    lot:        int   = 0
    posisi_rp:  float = 0.0   # nilai eksposur (Rp) pada sizing yang direkomendasikan

    _ind: Optional[dict] = field(default=None, repr=False, compare=False)


def effective_signal(r: SR, min_rr: float) -> str:
    """
    Sinyal EFEKTIF untuk tampilan/filter: turunkan ke AVOID jika R/R di
    bawah threshold yang dipilih user di sidebar SAAT INI — dihitung
    fresh setiap render (TIDAK memutasi r.signal), sehingga slider bisa
    diubah bolak-balik tanpa re-run screener dan tanpa histeresis aneh.
    """
    return "AVOID" if r.rr < min_rr else r.signal


# ═══════════════════════════════════════════════════════
# DOWNLOAD LAYER
# ═══════════════════════════════════════════════════════
def _safe_download(ticker: str, period: str) -> Optional[pd.DataFrame]:
    """
    Download SATU ticker via yf.Ticker().history() — SELALU flat columns
    di semua versi yfinance (root-cause fix utk bug "0/953 emiten").
    Retry 3x dengan exponential backoff + jitter utk rate-limit transient.
    """
    for attempt in range(MAX_RETRY):
        try:
            if attempt > 0:
                time.sleep((2 ** attempt) + random.uniform(0.05, 0.4))

            df = yf.Ticker(ticker).history(
                period=period, interval="1d",
                auto_adjust=True, actions=False,
            )
            if df is None or df.empty:
                continue

            df.columns = [str(c).strip().title() for c in df.columns]
            required = {"Open", "High", "Low", "Close", "Volume"}
            if not required.issubset(df.columns):
                continue

            df = df[sorted(required)].dropna(subset=["Close"])
            if len(df) < MIN_BARS:
                return None
            return df

        except Exception:
            pass

    return None


def _batch_chunk(tickers: list, period: str) -> dict:
    """
    Download sekelompok ticker dalam SATU yf.download() call (efisiensi).

    DATA-INTEGRITY FIX: sebelumnya, jika raw.columns TIDAK MultiIndex
    padahal ticker>1 (edge case tapi nyata), kode lama meng-copy SATU
    dataframe yang sama ke SEMUA ticker → salah atribusi harga secara
    diam-diam. Sekarang kasus itu di-raise sebagai kegagalan batch,
    sehingga jatuh ke fallback individual per ticker (aman, walau lebih
    lambat) daripada mempercayai data yang ambigu.
    """
    results: dict = {}
    if not tickers:
        return results

    if len(tickers) == 1:
        df = _safe_download(tickers[0], period)
        if df is not None:
            results[tickers[0]] = df
        return results

    try:
        raw = yf.download(
            tickers, period=period, interval="1d",
            group_by="ticker", auto_adjust=True,
            progress=False, threads=False,
        )
        if raw is None or raw.empty:
            raise ValueError("empty batch result")

        if not isinstance(raw.columns, pd.MultiIndex):
            # FIX: multi-ticker request TAPI hasil flat columns — tidak
            # bisa dipastikan data ini milik ticker mana. Jangan dipercaya.
            raise ValueError("unexpected flat columns for multi-ticker batch")

        lvl0 = raw.columns.get_level_values(0).unique().tolist()
        lvl1 = raw.columns.get_level_values(1).unique().tolist()

        for t in tickers:
            try:
                if t in lvl0:
                    df = raw[t].copy()
                elif t in lvl1:
                    df = raw.xs(t, axis=1, level=1).copy()
                else:
                    continue

                df.columns = [str(c).strip().title() for c in df.columns]
                required = {"Open", "High", "Low", "Close", "Volume"}
                if not required.issubset(df.columns):
                    continue

                df = df[sorted(required)].dropna(subset=["Close"])
                if len(df) >= MIN_BARS:
                    results[t] = df
            except Exception:
                continue

    except Exception:
        for t in tickers:
            df = _safe_download(t, period)
            if df is not None:
                results[t] = df

    return results


def fetch_all(tickers: list, period: str = PERIOD,
              max_workers: int = 6, chunk_size: int = CHUNK_SIZE) -> dict:
    """Download semua ticker paralel dalam chunks + fallback individual utk yang gagal."""
    chunks  = [tickers[i: i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    results: dict = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_batch_chunk, ch, period): ch for ch in chunks}
        for fut in as_completed(futs):
            try:
                data = fut.result()
                if data:
                    results.update(data)
            except Exception:
                pass

    missing = [t for t in tickers if t not in results]
    if missing:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs2 = {ex.submit(_safe_download, t, period): t for t in missing}
            for fut in as_completed(futs2):
                t = futs2[fut]
                try:
                    df = fut.result()
                    if df is not None:
                        results[t] = df
                except Exception:
                    pass

    return results


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ihsg(period: str = PERIOD) -> pd.DataFrame:
    df = _safe_download(IHSG_TICKER, period)
    return df if (df is not None and len(df) >= MIN_BARS) else pd.DataFrame()


# ═══════════════════════════════════════════════════════
# EMITEN LIST
# ═══════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def load_emiten() -> pd.DataFrame:
    try:
        df = pd.read_csv("emiten.csv")
        df.columns = [c.strip().lower() for c in df.columns]
        if "ticker" not in df.columns:
            raise ValueError
        df["ticker"] = df["ticker"].str.strip().str.upper()
        df["ticker"] = df["ticker"].apply(lambda x: x if x.endswith(".JK") else x + ".JK")
        if "sektor" not in df.columns:
            df["sektor"] = "—"
        return df.dropna(subset=["ticker"])
    except FileNotFoundError:
        base = [
            "BBCA","BMRI","BBRI","BREN","AMMN","TLKM","ASII","GOTO",
            "MDKA","ICBP","INDF","SMGR","ADRO","UNVR","KLBF","PTBA",
            "ANTM","EXCL","MIKA","CPIN","ITMG","BUMI","MEDC","INCO",
            "SIDO","MTEL","ISAT","GGRM","HMSP","TOWR","WIFI","DSSA",
        ]
        return pd.DataFrame({"ticker": [t + ".JK" for t in base], "sektor": "—"})


# ═══════════════════════════════════════════════════════
# HEIKIN ASHI — formula rekursif yang benar
# ═══════════════════════════════════════════════════════
def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    HA_C[i] = (O+H+L+C)/4
    HA_O[i] = (HA_O[i-1] + HA_C[i-1]) / 2  — rekursif murni, tidak vectorizable
    HA_H[i] = max(H, HA_O[i], HA_C[i]);  HA_L[i] = min(L, HA_O[i], HA_C[i])
    """
    ha_c = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_c_arr = ha_c.values

    ha_o = np.empty(len(df))
    ha_o[0] = (float(df["Open"].iloc[0]) + float(df["Close"].iloc[0])) / 2
    for i in range(1, len(df)):
        ha_o[i] = (ha_o[i - 1] + ha_c_arr[i - 1]) / 2

    ha_o_s = pd.Series(ha_o, index=df.index)
    ha_h   = pd.concat([ha_o_s, ha_c, df["High"]], axis=1).max(axis=1)
    ha_l   = pd.concat([ha_o_s, ha_c, df["Low"]],  axis=1).min(axis=1)

    return pd.DataFrame(
        {"HA_O": ha_o_s, "HA_C": ha_c, "HA_H": ha_h, "HA_L": ha_l},
        index=df.index,
    )


# ═══════════════════════════════════════════════════════
# STOCHASTIC RSI — Wilder's EMA, stochastic dari nilai RSI
# ═══════════════════════════════════════════════════════
def stoch_rsi(close: pd.Series,
              rsi_p: int = 14, stoch_p: int = 14,
              k_s: int = 3, d_s: int = 3):
    """RSI pakai Wilder's EMA (alpha=1/period) — identik MetaTrader/TradingView."""
    delta = close.diff()
    alpha = 1.0 / rsi_p
    gain  = delta.clip(lower=0).ewm(alpha=alpha, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=alpha, adjust=False).mean()
    rsi   = 100 - 100 / (1 + gain / (loss + 1e-9))

    rsi_lo = rsi.rolling(stoch_p).min()
    rsi_hi = rsi.rolling(stoch_p).max()
    raw_k  = (rsi - rsi_lo) / (rsi_hi - rsi_lo + 1e-9) * 100

    k = raw_k.rolling(k_s).mean()
    d = k.rolling(d_s).mean()
    return k, d, rsi


# ═══════════════════════════════════════════════════════
# ICHIMOKU — implementasi manual, shift eksplisit & terverifikasi
# ═══════════════════════════════════════════════════════
def ichimoku_manual(high: pd.Series, low: pd.Series, close: pd.Series,
                    w1: int = 9, w2: int = 26, w3: int = 52):
    """
    Dihitung manual (bukan pakai ta.trend.IchimokuIndicator) supaya shift
    sepenuhnya eksplisit dan tidak bergantung pada asumsi perilaku
    internal library — TERVERIFIKASI EMPIRIS bahwa ta.IchimokuIndicator
    dengan visual=False (default) TIDAK melakukan shift apapun.

    Tenkan/Kijun: nilai di bar t dihitung dari window berakhir di t
                  (tidak di-shift — memang diplot di posisi t).
    Senkou A/B  : dihitung dari Tenkan/Kijun/High/Low di bar t, TAPI
                  secara konvensi Ichimoku standar nilai ini diplot 26
                  bar KE DEPAN (leading span). Supaya "cloud yang
                  terlihat hari ini" (present cloud) bisa dibandingkan
                  dengan harga hari ini, kita ambil nilai yang DIHITUNG
                  26 bar LALU — yaitu raw_senkou.shift(26).
    Chikou      : close hari ini, secara konvensi diplot 26 bar ke
                  BELAKANG. Untuk chart, ini close.shift(-26). Untuk
                  keperluan konfirmasi sinyal, "Chikou di atas harga
                  26 bar lalu" cukup dicek sbg close[t] > close[t-26].
    """
    tenkan = (high.rolling(w1).max() + low.rolling(w1).min()) / 2
    kijun  = (high.rolling(w2).max() + low.rolling(w2).min()) / 2

    senkou_a_raw = (tenkan + kijun) / 2
    senkou_b_raw = (high.rolling(w3).max() + low.rolling(w3).min()) / 2

    senkou_a = senkou_a_raw.shift(w2)   # present cloud (FIX utama)
    senkou_b = senkou_b_raw.shift(w2)   # present cloud (FIX utama)
    chikou_chart = close.shift(-w2)     # untuk referensi visual di chart

    return dict(
        tenkan=tenkan, kijun=kijun,
        senkou_a=senkou_a, senkou_b=senkou_b,
        chikou_chart=chikou_chart,
    )


# ═══════════════════════════════════════════════════════
# COMPUTE INDICATORS
# ═══════════════════════════════════════════════════════
def compute_indicators(df_raw: pd.DataFrame) -> Optional[dict]:
    """
    Hitung semua indikator dari df_raw PENUH (jangan tail/potong dulu).
    Selalu bekerja pada salinan internal agar tidak mutate df di cache.
    """
    if len(df_raw) < MIN_BARS:
        return None

    df = df_raw.copy()
    c, h, lo, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]

    ichi = ichimoku_manual(h, lo, c)

    ha    = heikin_ashi(df)
    ema20 = EMAIndicator(c, window=20).ema_indicator()
    ema50 = EMAIndicator(c, window=50).ema_indicator()
    ma200 = c.rolling(200).mean()   # NaN kalau history <200 bar — ditangani graceful di screen_one
    macd_hist = MACD(c).macd_diff()

    adx_obj  = ADXIndicator(h, lo, c, window=14)
    adx_val  = adx_obj.adx()
    di_plus  = adx_obj.adx_pos()
    di_minus = adx_obj.adx_neg()

    atr = AverageTrueRange(h, lo, c, window=14).average_true_range()
    srsi_k, srsi_d, rsi14 = stoch_rsi(c)

    adv20   = v.rolling(20).mean()
    vol_rel = v / adv20.replace(0, np.nan)
    adv_rp  = (c * v).rolling(20).mean() / 1_000_000

    cqs = ((c - lo) / (h - lo).replace(0, np.nan)).fillna(0.5)

    return dict(
        close=c, high=h, low=lo, open=o, volume=v,
        tenkan=ichi["tenkan"], kijun=ichi["kijun"],
        senkou_a=ichi["senkou_a"], senkou_b=ichi["senkou_b"],
        chikou_chart=ichi["chikou_chart"],
        ha=ha, ema20=ema20, ema50=ema50, ma200=ma200,
        macd_hist=macd_hist,
        adx=adx_val, di_plus=di_plus, di_minus=di_minus,
        atr=atr, srsi_k=srsi_k, srsi_d=srsi_d, rsi14=rsi14,
        adv20=adv20, vol_rel=vol_rel, adv_rp=adv_rp, cqs=cqs,
    )


# ═══════════════════════════════════════════════════════
# FIBONACCI — swing high (puncak) lalu swing low SEBELUM puncak
# ═══════════════════════════════════════════════════════
def find_swing(high: pd.Series, low: pd.Series, lookback: int = 90) -> tuple:
    """
    Swing high = titik tertinggi dalam window (puncak impulse leg).
    Swing low  = titik terendah SEBELUM puncak itu (awal impulse leg).

    FIX: versi sebelumnya mencari pivot high & pivot low TERBARU secara
    independen — bisa menangkap swing low dari PULLBACK SETELAH puncak
    (mengukur down-leg, bukan up-leg yang sedang di-retrace). Diverifikasi
    dengan data sintetis: versi lama mengukur range 32, versi ini 71.7
    (swing yang benar) pada pola naik-puncak-turun-naik yang sama.
    """
    n  = min(lookback, len(high))
    hw = high.iloc[-n:].reset_index(drop=True)
    lw = low.iloc[-n:].reset_index(drop=True)

    sh_pos = int(hw.values.argmax())
    sh     = float(hw.iloc[sh_pos])

    if sh_pos > 0:
        sl = float(lw.iloc[: sh_pos + 1].min())
    else:
        sl = float(lw.min())

    if sh <= sl:
        sh, sl = float(hw.max()), float(lw.min())
    return sh, sl


def fib_levels(sh: float, sl: float) -> dict:
    d = sh - sl
    return dict(
        sh=sh, sl=sl,
        f236=sh - 0.236 * d,
        f382=sh - 0.382 * d,
        f500=sh - 0.500 * d,
        f618=sh - 0.618 * d,
        e1272=sl + 1.272 * d,
        e1618=sl + 1.618 * d,
        e2000=sl + 2.000 * d,
    )


def find_swing_bearish(high: pd.Series, low: pd.Series, lookback: int = 90) -> tuple:
    """
    Mirror dari find_swing() untuk konteks BREAKDOWN (saham gagal gate
    bullish — di bawah awan / HA merah). Dipakai HANYA di mode Analisa
    Manual (require_gate=False); Screener Massal tidak pernah sampai ke
    fungsi ini karena sudah di-reject duluan oleh gate.

    Swing low  = titik TERENDAH dalam window (dasar dari down-leg).
    Swing high = titik TERTINGGI SEBELUM low itu (puncak sebelum longsor,
                 titik awal dari down-leg yang sedang diukur).
    """
    n  = min(lookback, len(high))
    hw = high.iloc[-n:].reset_index(drop=True)
    lw = low.iloc[-n:].reset_index(drop=True)

    sl_pos = int(lw.values.argmin())
    sl     = float(lw.iloc[sl_pos])

    if sl_pos > 0:
        sh = float(hw.iloc[: sl_pos + 1].max())
    else:
        sh = float(hw.max())

    if sh <= sl:
        sh, sl = float(hw.max()), float(lw.min())
    return sh, sl


def fib_levels_bearish(sh: float, sl: float) -> dict:
    """
    Level Fibonacci utk PETA BREAKDOWN (bukan sinyal beli):
    - r382/r500/r618 : zona RESISTANCE — retracement NAIK dari swing low
                       ke arah swing high. Ini area dimana rally/bounce
                       kemungkinan mentok.
    - r618 dobel-fungsi sbg level INVALIDASI: kalau harga closing di atas
      ini, anggap tekanan jual mulai habis / breakdown mulai diragukan.
    - d1272/d1618    : proyeksi DOWNSIDE — ekstensi ke BAWAH swing low,
                       seberapa jauh longsor bisa lanjut kalau breakdown
                       berlanjut.
    """
    d = sh - sl
    return dict(
        sh=sh, sl=sl,
        r382=sl + 0.382 * d,
        r500=sl + 0.500 * d,
        r618=sl + 0.618 * d,     # invalidation level
        d1272=sh - 1.272 * d,
        d1618=sh - 1.618 * d,
    )


def adaptive_lookback(atr_pct: float) -> int:
    """
    Lookback Fibonacci menyesuaikan volatilitas: saham stabil (ATR%
    rendah) butuh window lebih panjang utk menemukan swing berarti;
    saham volatile (ATR% tinggi) membentuk swing lebih cepat.

    FIX: formula lama (30/atr_pct, floor 60) diverifikasi numerik HAMPIR
    SELALU jatuh ke floor 60 utk rentang ATR% realistis saham IDX
    (1-4%), gagal benar-benar "adaptif". Diskalakan ulang dari baseline
    ATR%=2.0 → lookback=90, dengan rentang efektif 60-150.
    """
    baseline = 2.0
    return int(np.clip(90 * (baseline / max(atr_pct, 0.3)), 60, 150))


# ═══════════════════════════════════════════════════════
# MARKET REGIME — dipecah jadi 2 fungsi murni & jelas
# ═══════════════════════════════════════════════════════
def ihsg_trend(ihsg_df: pd.DataFrame) -> tuple:
    """Return (ihsg_bull: bool, ihsg_ratio: float) dari EMA20 vs EMA50 IHSG."""
    if ihsg_df.empty or len(ihsg_df) <= 52:
        return False, 1.0
    ic  = ihsg_df["Close"].squeeze()
    e20 = ic.ewm(span=20, adjust=False).mean()
    e50 = ic.ewm(span=50, adjust=False).mean()
    bull  = float(e20.iloc[-1]) > float(e50.iloc[-1])
    ratio = float(e20.iloc[-1]) / max(float(e50.iloc[-1]), 1.0)
    return bull, ratio


def decide_regime(ihsg_bull: bool, breadth: float) -> str:
    """Gabungkan trend IHSG + market breadth jadi satu label regime."""
    if ihsg_bull and breadth > 55:
        return "BULL"
    if not ihsg_bull and breadth < 45:
        return "BEAR"
    return "NEUTRAL"


# ═══════════════════════════════════════════════════════
# RELATIVE STRENGTH vs IHSG
# ═══════════════════════════════════════════════════════
def rs_score(close: pd.Series, ihsg_ret: Optional[pd.Series], window: int = 20) -> float:
    """
    PERF-FIX: menerima ihsg_ret (pct_change IHSG) yang SUDAH dihitung
    sekali di caller, bukan menghitung ulang .pct_change() setiap
    dipanggil (sebelumnya terjadi 953x per screening run — mahal & sia-sia
    karena hasilnya selalu sama).
    """
    if ihsg_ret is None:
        return 0.0
    common = close.index.intersection(ihsg_ret.index)
    if len(common) < window + 5:
        return 0.0
    rel    = close.loc[common].pct_change() - ihsg_ret.loc[common]
    recent = rel.iloc[-window:]
    mu, std = recent.mean(), recent.std()
    return float(mu / std) if std > 0 and not np.isnan(std) else 0.0


# ═══════════════════════════════════════════════════════
# SCREENER ENGINE
# ═══════════════════════════════════════════════════════
def _conf(s: int, mx: int) -> str:
    p = s / mx if mx > 0 else 0
    if p >= 0.80: return "VERY HIGH"
    if p >= 0.60: return "HIGH"
    if p >= 0.40: return "MEDIUM"
    return "LOW"


def _signal(s: int, mx: int, rr: float, adx: float) -> str:
    if rr < 1.5:
        return "AVOID"
    if adx < 20 and _conf(s, mx) == "LOW":
        return "AVOID"
    c = _conf(s, mx)
    if c == "VERY HIGH": return "STRONG BUY"
    if c == "HIGH":      return "BUY"
    if c == "MEDIUM":    return "WATCH"
    return "AVOID"


def screen_one(ticker: str, df_raw: pd.DataFrame,
               ihsg_ret: Optional[pd.Series],
               min_adv_rp: float, sektor: str = "—",
               modal_rp: float = 100_000_000,
               risk_pct: float = 0.01,
               max_exposure_pct: float = 0.25,
               require_gate: bool = True) -> Optional[SR]:
    """
    Analisis teknikal satu ticker. Scoring max = 15 poin:
      Ichimoku (5, termasuk Chikou) + ADX/Trend (3) + Momentum (4)
      + Volume (2) + RS (1)

    require_gate=True  (default, dipakai Screener Massal — TIDAK berubah
                        dari v6.3): jika harga di bawah cloud atau HA
                        merah atau ADV di bawah min_adv_rp → return None.
    require_gate=False (dipakai mode Analisa Manual): gate di atas TIDAK
                        menggugurkan hasil — semua saham selalu dianalisis
                        penuh. Tapi supaya tidak menyesatkan, jika gate
                        gagal, signal dipaksa maksimal WATCH (tidak
                        pernah BUY/STRONG BUY) dan alasannya ditulis
                        eksplisit di baris pertama reasons.
    """
    ind = compute_indicators(df_raw)
    if ind is None:
        return None

    price = float(ind["close"].iloc[-1])
    if price <= 0:
        return None

    adv_rp_v = float(ind["adv_rp"].iloc[-1])
    if np.isnan(adv_rp_v):
        return None   # data benar-benar tidak ada, bukan soal threshold
    if require_gate and adv_rp_v < min_adv_rp:
        return None

    atr_v   = float(ind["atr"].iloc[-1])
    atr_pct = atr_v / price * 100

    def f(key, idx=-1):
        return float(ind[key].iloc[idx])

    tenkan_v  = f("tenkan")
    kijun_v   = f("kijun")
    sa_v      = f("senkou_a")
    sb_v      = f("senkou_b")
    ema20_v   = f("ema20")
    ema50_v   = f("ema50")
    macd_now  = f("macd_hist")
    macd_prev = f("macd_hist", -2)
    adx_v     = f("adx")
    dip_v     = f("di_plus")
    dim_v     = f("di_minus")
    sk_v      = f("srsi_k")
    sd_v      = f("srsi_d")
    sk_p      = f("srsi_k", -2)
    sd_p      = f("srsi_d", -2)
    vol_r     = f("vol_rel")

    ha_now   = ind["ha"].iloc[-1]
    ha_bull  = float(ha_now["HA_C"]) > float(ha_now["HA_O"])
    ha_noshd = (float(ha_now["HA_O"]) - float(ha_now["HA_L"])) < atr_v * 0.3

    # FIX: validitas cloud pakai isna() eksplisit (bukan sentinel -1 lama)
    if pd.isna(sa_v) or pd.isna(sb_v):
        return None

    kumo_top    = max(sa_v, sb_v)
    above_cloud = price > kumo_top
    gate_passed = above_cloud and ha_bull

    if require_gate and not gate_passed:
        return None

    # Chikou confirmation: close hari ini > close 26 bar lalu
    close_s = ind["close"]
    chikou_bull = (len(close_s) > 26) and (float(close_s.iloc[-1]) > float(close_s.iloc[-27]))

    # Trend duration (pakai cloud yang SUDAH benar timing-nya)
    trend_bars = 0
    for i in range(1, min(60, len(close_s))):
        sa_i = float(ind["senkou_a"].iloc[-i])
        sb_i = float(ind["senkou_b"].iloc[-i])
        if pd.isna(sa_i) or pd.isna(sb_i):
            break
        if float(close_s.iloc[-i]) > max(sa_i, sb_i):
            trend_bars += 1
        else:
            break

    # ── SCORING max 15 ──
    score = 0
    MX    = 15
    reasons: list = []

    # Jika dianalisis via mode Analisa Manual (require_gate=False) dan gate
    # utama gagal, catat eksplisit di baris PALING ATAS supaya user paham
    # kenapa ini bukan sinyal beli aktif — meski tetap dapat analisis penuh.
    if not gate_passed:
        if not above_cloud:
            reasons.append("🔴 GATE UTAMA GAGAL: harga di BAWAH awan Ichimoku — bukan setup bullish yang valid")
        if not ha_bull:
            reasons.append("🔴 GATE UTAMA GAGAL: Heikin Ashi MERAH — momentum turun, bukan setup bullish yang valid")

    # Peringatan likuiditas informasional (tidak memengaruhi skor) — relevan
    # terutama di mode Analisa Manual dimana min_adv_rp tidak digerbang.
    if adv_rp_v < 500:
        reasons.append(f"⚠️ Likuiditas SANGAT RENDAH (ADV Rp{adv_rp_v:,.0f}jt/hari) — risiko slippage tinggi")

    # A. ICHIMOKU (max 5)
    if sa_v > sb_v:
        score += 1
        reasons.append("🟢 Awan HIJAU (SA>SB) — bullish cloud structure")
    else:
        reasons.append("🟡 Awan merah — cloud bearish, perlu konfirmasi lebih")

    if tenkan_v > kijun_v:
        score += 1
        reasons.append("🟢 Tenkan > Kijun — momentum jangka pendek bullish")
    else:
        reasons.append("🔴 Tenkan < Kijun — momentum jangka pendek lemah")

    if price > kijun_v:
        score += 1
        reasons.append(f"🟢 Harga ({price:,.0f}) > Kijun ({kijun_v:,.0f})")

    kijun_5d = float(ind["kijun"].iloc[-6]) if len(ind["kijun"]) > 5 else kijun_v
    if kijun_v > kijun_5d:
        score += 1
        reasons.append("🟢 Kijun slope naik — tren dasar menguat")

    if chikou_bull:
        score += 1
        reasons.append("🟢 Chikou Span di atas harga 26 hari lalu — konfirmasi historis bullish")
    else:
        reasons.append("🔴 Chikou Span di bawah/setara harga 26 hari lalu — konfirmasi lemah")

    # B. TREND / ADX (max 3)
    if adx_v >= 25:
        score += 1
        reasons.append(f"🟢 ADX kuat {adx_v:.1f} — tren tegas")
    elif adx_v >= 20:
        score += 1
        reasons.append(f"🟡 ADX moderat {adx_v:.1f} — tren sedang membangun")
    else:
        reasons.append(f"🔴 ADX lemah {adx_v:.1f} — sideways/choppy")

    if dip_v > dim_v:
        score += 1
        reasons.append(f"🟢 DI+ ({dip_v:.1f}) > DI- ({dim_v:.1f}) — beli dominan")
    else:
        reasons.append("🔴 DI- > DI+ — jual dominan")

    if ema20_v > ema50_v:
        score += 1
        reasons.append("🟢 EMA20 > EMA50 — uptrend menengah")
    else:
        reasons.append("🔴 EMA20 < EMA50 — downtrend menengah")

    # C. MOMENTUM (max 4)
    macd_fresh = macd_now > 0 and macd_prev <= 0
    macd_up    = macd_now > 0 and macd_now > macd_prev
    if macd_fresh:
        score += 2
        reasons.append("🔥 MACD FRESH CROSS — histogram baru jadi positif, timing ideal")
    elif macd_up:
        score += 1
        reasons.append("🟢 MACD menguat — momentum upside membangun")
    else:
        reasons.append("🔴 MACD negatif/melemah")

    srsi_cross   = (sk_v > sd_v) and (sk_p <= sd_p)
    srsi_from_os = (sk_v > 20) and (sk_p <= 20)
    if srsi_cross and sk_v < 50:
        score += 1
        reasons.append(f"🟢 StochRSI fresh cross ({sk_v:.0f}) — momentum awal")
    elif srsi_from_os:
        score += 1
        reasons.append(f"🔥 StochRSI keluar oversold ({sk_v:.0f}) — reversal")
    elif sk_v > 80:
        reasons.append(f"🟡 StochRSI overbought ({sk_v:.0f}) — hati-hati")

    if ha_noshd:
        score += 1
        reasons.append("🔥 HA kuat tanpa ekor bawah — bullish pressure dominan")

    # D. VOLUME (max 2)
    if vol_r >= 2.0:
        score += 1
        reasons.append(f"🔥 Volume SPIKE {vol_r:.1f}xADV — smart money masuk")
    elif vol_r >= 1.2:
        score += 1
        reasons.append(f"🟢 Volume {vol_r:.1f}xADV — di atas normal")
    else:
        reasons.append(f"🟡 Volume {vol_r:.1f}xADV — lemah")

    if adv_rp_v >= 10_000:
        score += 1
        reasons.append(f"🟢 Likuiditas tinggi (ADV Rp{adv_rp_v:,.0f}jt/hari)")

    # E. RS vs IHSG (max 1)
    rs_v = rs_score(close_s, ihsg_ret)
    if rs_v > 0.5:
        score += 1
        reasons.append(f"🟢 RS {rs_v:+.2f} — outperform IHSG")
    elif rs_v < -0.5:
        reasons.append(f"🔴 RS {rs_v:+.2f} — underperform IHSG")

    # ── MA200: konteks trend jangka panjang (INFORMASIONAL — tidak memengaruhi skor) ──
    # Independen dari gate Ichimoku (yang menilai kondisi jangka pendek/menengah):
    # saham bisa saja gate_passed=True (setup jangka pendek bullish) TAPI masih
    # di bawah MA200 (gambaran besar jangka panjang masih bearish), atau sebaliknya.
    ma200_v = f("ma200")
    if pd.isna(ma200_v):
        above_ma200 = None
        reasons.append("⚪ MA200 belum tersedia (data historis <200 hari) — trend jangka panjang tidak terukur")
    else:
        above_ma200 = price > ma200_v
        if above_ma200:
            reasons.append(f"🟢 Harga di atas MA200 ({ma200_v:,.0f}) — trend jangka panjang masih sehat")
        else:
            reasons.append(f"🔴 Harga di BAWAH MA200 ({ma200_v:,.0f}) — trend jangka panjang bearish, ekstra hati-hati")

    # ══════════════════════════════════════════════════
    # FIBONACCI — cabang BULLISH (default) vs BEARISH (breakdown map)
    # ══════════════════════════════════════════════════
    # Bullish: dipakai kalau gate_passed=True (SELALU True di Screener Massal
    #          karena require_gate=True sudah reject duluan kalau gate gagal;
    #          logika & angka di cabang ini 100% SAMA seperti v6.4, tidak berubah).
    # Bearish: HANYA bisa tercapai kalau require_gate=False (mode Analisa Manual)
    #          DAN gate_passed=False — memetakan level breakdown (resistance,
    #          invalidasi, proyeksi downside) alih-alih entry/target beli.
    lookback = adaptive_lookback(atr_pct)

    if gate_passed:
        fib_mode = "bullish"
        sh, sl   = find_swing(ind["high"], ind["low"], lookback)
        fib      = fib_levels(sh, sl)

        entry_hi  = fib["f382"]
        entry_lo  = fib["f500"]
        entry_mid = (entry_hi + entry_lo) / 2

        tech_sl     = max(kijun_v, fib["f618"])
        min_dist_sl = price - atr_v * 1.2
        stop_loss   = min(tech_sl, min_dist_sl)
        stop_loss   = max(stop_loss, price * 0.85)
        stop_loss   = min(stop_loss, price * 0.995)

        target1 = fib["e1272"]
        target2 = fib["e1618"]

        entry_ref = max(price, entry_mid)
        risk      = max(entry_ref - stop_loss, 1.0)
        reward    = target1 - entry_ref
        rr        = reward / risk

    else:
        fib_mode = "bearish"
        sh, sl   = find_swing_bearish(ind["high"], ind["low"], lookback)
        fibb     = fib_levels_bearish(sh, sl)

        # Field direuse dgn makna beda (dijelaskan ulang di render_analysis_content):
        # entry_hi/entry_lo -> zona resistance; stop_loss -> level invalidasi;
        # target1/target2   -> proyeksi downside.
        entry_hi  = fibb["r500"]
        entry_lo  = fibb["r382"]
        stop_loss = fibb["r618"]     # level invalidasi breakdown
        target1   = fibb["d1272"]
        target2   = fibb["d1618"]

        # R/R diinterpretasi sbg "kalau breakdown invalid (reclaim r618),
        # potensi ke swing high lama vs risiko balik ke swing low."
        risk   = max(stop_loss - sl, 1.0)
        reward = max(sh - stop_loss, 0.0)
        rr     = reward / risk

    conf   = _conf(score, MX)
    signal = _signal(score, MX, rr, adx_v)

    # FIX konsistensi: kalau gate utama gagal (hanya bisa terjadi saat
    # require_gate=False, mode Analisa Manual), signal TIDAK BOLEH tampil
    # sebagai BUY/STRONG BUY — itu akan kontradiksi dengan gate wajib yang
    # dipakai Screener Massal. Turunkan paksa ke maksimal WATCH.
    if not gate_passed and signal in ("STRONG BUY", "BUY"):
        signal = "WATCH"

    # ── POSITION SIZING: risk-based DENGAN cap eksposur maksimal ──
    # Catatan: di fib_mode bearish, "stop_loss" adalah level invalidasi
    # (di ATAS harga saat ini), jadi risk_per_sh di sini tetap dihitung
    # jarak absolut — sizing tetap masuk akal sbg "seberapa besar posisi
    # kalau nanti entry di level reclaim tsb", bukan entry sekarang.
    risk_rp     = modal_rp * risk_pct
    risk_per_sh = abs(price - stop_loss)
    lot_by_risk = int(risk_rp / (risk_per_sh * 100)) if risk_per_sh > 0 else 0
    lot_by_expo = int((modal_rp * max_exposure_pct) / (price * 100))
    lot = max(0, min(lot_by_risk, lot_by_expo, 500))
    posisi_rp = lot * 100 * price

    return SR(
        ticker=ticker, harga=price, signal=signal,
        score=score, score_max=MX, conf=conf, reasons=reasons,
        sh=sh, sl_fib=sl, entry_hi=entry_hi, entry_lo=entry_lo,
        target1=target1, target2=target2, stop_loss=stop_loss, rr=rr,
        atr=atr_v, atr_pct=atr_pct, adv_rp=adv_rp_v,
        vol_rel=vol_r, rs=rs_v, adx=adx_v,
        trend_bars=trend_bars, cqs=float(ind["cqs"].iloc[-1]),
        chikou_bull=chikou_bull, gate_passed=gate_passed,
        ma200=ma200_v if not pd.isna(ma200_v) else float("nan"),
        above_ma200=above_ma200, fib_mode=fib_mode,
        sektor=sektor, lot=lot, posisi_rp=posisi_rp,
        _ind=ind,
    )


# ═══════════════════════════════════════════════════════
# CHART
# ═══════════════════════════════════════════════════════
def build_chart(ticker: str, df_raw: pd.DataFrame,
                res: SR, display_bars: int = 120) -> go.Figure:
    """Indikator dari df_raw PENUH (warm-up cukup), dipotong ke display_bars utk tampilan."""
    ind = res._ind if res._ind is not None else compute_indicators(df_raw)
    if ind is None:
        return go.Figure()

    n   = len(df_raw)
    cut = min(display_bars, n)
    s   = slice(n - cut, n)
    dt  = df_raw.index[s]

    op  = df_raw["Open"].iloc[s]
    cl  = df_raw["Close"].iloc[s]
    hi  = df_raw["High"].iloc[s]
    lo  = df_raw["Low"].iloc[s]
    vo  = df_raw["Volume"].iloc[s]
    sa  = ind["senkou_a"].iloc[s]
    sb  = ind["senkou_b"].iloc[s]
    e20 = ind["ema20"].iloc[s]
    e50 = ind["ema50"].iloc[s]
    ma200_s = ind["ma200"].iloc[s]   # bisa penuh NaN kalau history <200 bar — plotly skip otomatis
    adv = ind["adv20"].iloc[s]
    chikou = ind["chikou_chart"].iloc[s]

    BG, BG2 = "#0E1117", "#1E2329"
    bull = res.signal in ("STRONG BUY", "BUY")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=dt, open=op, high=hi, low=lo, close=cl, name=ticker,
        increasing_fillcolor="#00C853", increasing_line_color="#00C853",
        decreasing_fillcolor="#FF3B30", decreasing_line_color="#FF3B30",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=dt, y=sa, line=dict(width=0),
                             showlegend=False, name="SenkouA"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dt, y=sb, line=dict(width=0), name="Cloud", fill="tonexty",
        fillcolor="rgba(0,200,83,0.13)" if bull else "rgba(255,59,48,0.13)",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=dt, y=e20, name="EMA20",
                             line=dict(color="#F0B90B", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dt, y=e50, name="EMA50",
                             line=dict(color="#848E9C", width=1, dash="dot")), row=1, col=1)
    # MA200: hanya render kalau ada nilai valid (bukan seluruhnya NaN) — hindari
    # legend entry kosong untuk saham dgn history <200 hari
    if ma200_s.notna().any():
        fig.add_trace(go.Scatter(x=dt, y=ma200_s, name="MA200",
                                 line=dict(color="#FF6EC7", width=1.4, dash="dashdot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dt, y=chikou, name="Chikou",
                             line=dict(color="#BA68C8", width=1, dash="dot")), row=1, col=1)

    if res.fib_mode == "bearish":
        level_defs = [
            (res.entry_hi,  "#ffc107", f"Resistance 50%: {res.entry_hi:,.0f}"),
            (res.entry_lo,  "#ff9800", f"Resistance 38.2%: {res.entry_lo:,.0f}"),
            (res.stop_loss, "#00C853", f"Invalidasi 61.8%: {res.stop_loss:,.0f}"),
            (res.target1,   "#FF3B30", f"Downside 1 (127.2%): {res.target1:,.0f}"),
            (res.target2,   "#b71c1c", f"Downside 2 (161.8%): {res.target2:,.0f}"),
        ]
    else:
        level_defs = [
            (res.entry_hi,  "#00C853", f"Entry Hi 38.2%: {res.entry_hi:,.0f}"),
            (res.entry_lo,  "#4caf50", f"Entry Lo 50%: {res.entry_lo:,.0f}"),
            (res.stop_loss, "#FF3B30", f"Stop Loss: {res.stop_loss:,.0f}"),
            (res.target1,   "#F0B90B", f"T1 127.2%: {res.target1:,.0f}"),
            (res.target2,   "#FF9800", f"T2 161.8%: {res.target2:,.0f}"),
        ]

    for val, col, lbl in level_defs:
        fig.add_hline(y=val, row=1, col=1,
                      line=dict(color=col, width=1, dash="dash"),
                      annotation_text=f" {lbl}", annotation_position="right",
                      annotation_font=dict(color=col, size=9))

    vcol = ["#00C853" if float(cl.iloc[i]) >= float(op.iloc[i]) else "#FF3B30"
            for i in range(len(vo))]
    fig.add_trace(go.Bar(x=dt, y=vo, name="Vol", marker_color=vcol, opacity=0.65), row=2, col=1)
    fig.add_trace(go.Scatter(x=dt, y=adv, name="ADV20",
                             line=dict(color="#F0B90B", width=1, dash="dot")), row=2, col=1)

    sc = SIG_COLOR.get(res.signal, "#FAFAFA")
    title_suffix = " · 🗺️ PETA BREAKDOWN" if res.fib_mode == "bearish" else ""
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color="#FAFAFA", size=11), height=520,
        margin=dict(l=0, r=140, t=36, b=0),
        legend=dict(bgcolor="#1E2329", bordercolor="#2a2d35", x=0, y=1, font=dict(size=10)),
        xaxis_rangeslider_visible=False,
        title=dict(
            text=(f"<b>{ticker}</b>  {res.signal}  |  Score {res.score}/{res.score_max}  |  "
                  f"R/R {res.rr:.1f}:1  |  {res.trend_bars} bar di atas awan{title_suffix}"),
            x=0.01, font=dict(color=sc, size=13),
        ),
    )
    fig.update_xaxes(gridcolor="#2a2d35")
    fig.update_yaxes(gridcolor="#2a2d35")
    return fig


# ═══════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════
def render_regime(regime: str, breadth: float, ratio: float):
    cls_map = {"BULL": "regime-bull", "BEAR": "regime-bear"}
    col_map = {"BULL": "#00e676", "BEAR": "#f44336"}
    cls   = cls_map.get(regime, "regime-neutral")
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
        col.markdown(
            f"<div class='kpi' style='border-color:{bc};'>"
            f"<div class='kpi-lbl'>{lbl}</div>"
            f"<div class='kpi-val' style='color:{bc};'>{val}</div></div>",
            unsafe_allow_html=True,
        )


def results_to_df(results_with_sig: list) -> pd.DataFrame:
    rows = []
    for r, eff_sig in results_with_sig:
        rows.append({
            "Ticker":     r.ticker,
            "Signal":     eff_sig,
            "Score":      f"{r.score}/{r.score_max}",
            "Conf":       r.conf,
            "Harga":      f"{r.harga:,.0f}",
            "Entry Zone": f"{r.entry_lo:,.0f}–{r.entry_hi:,.0f}",
            "Target 1":   f"{r.target1:,.0f}",
            "Target 2":   f"{r.target2:,.0f}",
            "Stop Loss":  f"{r.stop_loss:,.0f}",
            "R/R":        f"{r.rr:.1f}:1",
            "ADX":        f"{r.adx:.0f}",
            "Vol×":       f"{r.vol_rel:.1f}",
            "RS":         f"{r.rs:+.2f}",
            "ADV(Rpjt)":  f"{r.adv_rp:,.0f}",
            "ATR%":       f"{r.atr_pct:.1f}%",
            "Lot Sizing": r.lot,
            "Eksposur":   f"Rp{r.posisi_rp/1_000_000:,.1f}jt",
            "Sektor":     r.sektor,
        })
    return pd.DataFrame(rows)


def render_analysis_content(res: SR, df_raw: pd.DataFrame,
                            modal_rp: float = 100_000_000, max_expo: float = 0.25):
    """
    Konten analisis lengkap (chart + metrics + reasons + action plan +
    kalkulator average down). Dipakai BERSAMA oleh show_modal() (Screener
    Massal) dan render_manual_analysis() (Analisa Manual) — satu sumber
    kebenaran, supaya perilaku & tampilan selalu konsisten di kedua mode.

    Label & wording metric/action-plan BERBEDA tergantung res.fib_mode:
    - "bullish" (default, satu-satunya mode yg dipakai Screener Massal):
      Entry Zone / Target 1&2 / Stop Loss — persis seperti v6.4, TIDAK berubah.
    - "bearish" (hanya tercapai di Analisa Manual saat gate gagal): field
      yang SAMA direinterpretasi sbg Resistance Zone / Proyeksi Downside /
      Level Invalidasi — peta breakdown, BUKAN rekomendasi entry beli.
    """
    sc = SIG_COLOR.get(res.signal, "#FAFAFA")
    is_bearish_map = (res.fib_mode == "bearish")

    if not res.gate_passed:
        st.warning(
            "⚠️ **Saham ini TIDAK memenuhi gate teknikal wajib** "
            "(harga di atas awan Ichimoku + Heikin Ashi hijau). "
            "Analisis di bawah tetap dihitung penuh untuk referensi, tapi "
            "**bukan sinyal beli aktif**. Level di bawah ini adalah **peta "
            "breakdown** (resistance & proyeksi downside), bukan zona entry."
        )

    # ── MA200 badge — konteks jangka panjang, independen dari gate Ichimoku ──
    if res.above_ma200 is None:
        ma200_badge = "⚪ MA200: data historis belum cukup (<200 hari)"
        ma200_color = "#848E9C"
    elif res.above_ma200:
        ma200_badge = f"🟢 Di atas MA200 ({res.ma200:,.0f})"
        ma200_color = "#4caf50"
    else:
        ma200_badge = f"🔴 Di BAWAH MA200 ({res.ma200:,.0f})"
        ma200_color = "#f44336"

    st.markdown(
        f"<h3 style='margin:0;'>{res.ticker} &nbsp;"
        f"<span style='color:{sc};font-size:.95rem;'>{res.signal}</span></h3>"
        f"<p style='color:#848E9C;margin:4px 0 10px;'>"
        f"Score <b style='color:#fff;'>{res.score}/{res.score_max}</b> · "
        f"Conf <b style='color:{sc};'>{res.conf}</b> · "
        f"Di atas awan <b style='color:#fff;'>{res.trend_bars} bar</b> · "
        f"Sektor: {res.sektor} &nbsp;·&nbsp; "
        f"<span style='color:{ma200_color};'>{ma200_badge}</span></p>",
        unsafe_allow_html=True,
    )

    fig = build_chart(res.ticker, df_raw, res)
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Harga", f"{res.harga:,.0f}")
    if is_bearish_map:
        c2.metric("Zona Resistance", f"{res.entry_lo:,.0f}–{res.entry_hi:,.0f}")
        c3.metric("Level Invalidasi", f"{res.stop_loss:,.0f}",
                  delta=f"+{(res.stop_loss/res.harga-1)*100:.1f}%", delta_color="off")
        c4.metric("Proyeksi Downside 1", f"{res.target1:,.0f}",
                  delta=f"{(res.target1/res.harga-1)*100:.1f}%")
        c5.metric("Proyeksi Downside 2", f"{res.target2:,.0f}",
                  delta=f"{(res.target2/res.harga-1)*100:.1f}%")
        c6.metric("R/R (jika reclaim)", f"{res.rr:.1f}:1", delta_color="off")
    else:
        c2.metric("Entry Zone", f"{res.entry_lo:,.0f}–{res.entry_hi:,.0f}")
        c3.metric("Stop Loss", f"{res.stop_loss:,.0f}",
                  delta=f"{(res.stop_loss/res.harga-1)*100:.1f}%", delta_color="inverse")
        c4.metric("Target 1", f"{res.target1:,.0f}",
                  delta=f"+{(res.target1/res.harga-1)*100:.1f}%")
        c5.metric("Target 2", f"{res.target2:,.0f}",
                  delta=f"+{(res.target2/res.harga-1)*100:.1f}%")
        rr_lbl = "✅ Bagus" if res.rr >= 2.0 else "⚠️ Cukup" if res.rr >= 1.5 else "❌ Buruk"
        c6.metric("R/R", f"{res.rr:.1f}:1", delta=rr_lbl, delta_color="off")

    st.markdown("---")
    col_r, col_p = st.columns(2)

    with col_r:
        st.markdown("#### 🧠 Analisis Sinyal")
        for reason in res.reasons:
            st.write(reason)

    with col_p:
        if is_bearish_map:
            st.markdown("#### 🗺️ Peta Breakdown")
            st.markdown(f"""
**📍 Swing Ref:** SH {res.sh:,.0f} *(puncak sebelum longsor)* → SL {res.sl_fib:,.0f} *(dasar saat ini)*

**🟡 Zona Resistance:** {res.entry_lo:,.0f} – {res.entry_hi:,.0f}
*(Fibo 38.2%–50% dari swing low — area rally/bounce kemungkinan mentok)*

**🚧 Level Invalidasi:** {res.stop_loss:,.0f} *(Fibo 61.8%)*
*Kalau closing di ATAS ini, anggap tekanan jual mulai habis — breakdown mulai diragukan, awasi reversal.*

**📉 Proyeksi Downside 1:** {res.target1:,.0f} *(ekstensi 127.2%)*
**📉 Proyeksi Downside 2:** {res.target2:,.0f} *(ekstensi 161.8%)*
*Seberapa jauh longsor bisa lanjut KALAU breakdown berlanjut — bukan target jual, ini peringatan risiko.*

**📐 R/R jika reclaim:** {res.rr:.1f}:1
*(kalau harga berhasil balik ke atas level invalidasi, potensi ke swing high lama vs risiko balik ke swing low)*

**💰 Sizing referensi** *(kalau nanti entry di level reclaim, risk 1% modal):*
→ Maks **{res.lot} lot** ({res.lot * 100:,} lembar) ≈ Rp{res.posisi_rp/1_000_000:,.1f}jt

**📊 ATR Harian:** {res.atr:,.0f} ({res.atr_pct:.1f}%)
**💧 Likuiditas:** ADV Rp{res.adv_rp:,.0f}jt/hari
**⚡ Volume:** {res.vol_rel:.1f}× ADV20
**📈 ADX:** {res.adx:.1f}  |  **RS vs IHSG:** {res.rs:+.2f}
**👁️ Chikou:** {"✅ Konfirmasi bullish" if res.chikou_bull else "⚠️ Belum konfirmasi"}
""")
        else:
            st.markdown("#### 🎯 Action Plan")
            rr_note = (
                "✅ Setup layak trading"         if res.rr >= 2.0 else
                "⚠️ R/R minimal, SL wajib ketat" if res.rr >= 1.5 else
                "❌ R/R buruk — pertimbangkan skip"
            )
            st.markdown(f"""
**📍 Swing Ref:** SH {res.sh:,.0f} → SL {res.sl_fib:,.0f}

**🟢 Zona Entry:** {res.entry_lo:,.0f} – {res.entry_hi:,.0f}
*(Fibo 50%–38.2% retracement, tunggu pullback)*

**🎯 Target 1:** {res.target1:,.0f} *(Fibo Ext 127.2%)*
**🎯 Target 2:** {res.target2:,.0f} *(Fibo Ext 161.8%)*

**🛑 Stop Loss:** {res.stop_loss:,.0f}
*(support teknikal, minimal ±1.2×ATR dari harga)*

**📐 R/R:** {res.rr:.1f}:1 — {rr_note}

**💰 Position Sizing** *(risk 1% modal, cap eksposur 25%):*
→ Maks **{res.lot} lot** ({res.lot * 100:,} lembar) ≈ Rp{res.posisi_rp/1_000_000:,.1f}jt

**📊 ATR Harian:** {res.atr:,.0f} ({res.atr_pct:.1f}%)
**💧 Likuiditas:** ADV Rp{res.adv_rp:,.0f}jt/hari
**⚡ Volume:** {res.vol_rel:.1f}× ADV20
**📈 ADX:** {res.adx:.1f}  |  **RS vs IHSG:** {res.rs:+.2f}
**👁️ Chikou:** {"✅ Konfirmasi bullish" if res.chikou_bull else "⚠️ Belum konfirmasi"}
""")
        if res.rs > 0.5:
            st.success(f"💪 Outperform IHSG (RS {res.rs:+.2f})")
        elif res.rs < -0.5:
            st.warning(f"⚠️ Underperform IHSG (RS {res.rs:+.2f})")

    # ══════════════════════════════════════════════════
    # KALKULATOR AVERAGE DOWN — opt-in (expander), berlaku utk saham
    # APAPUN (lolos gate atau tidak, Screener Massal atau Analisa Manual).
    # Ini kalkulator angka, BUKAN rekomendasi "harus/jangan average down"
    # — keputusan tetap di tangan user, tapi konteks teknikal (gate_passed,
    # fib_mode, level invalidasi/target) ditampilkan supaya keputusan itu
    # terinformasi.
    # ══════════════════════════════════════════════════
    st.markdown("---")
    with st.expander("💰 Kalkulator Average Down (kalau kamu sudah pegang saham ini)"):
        st.caption(
            "Isi posisi yang sudah kamu pegang, lihat average price baru & "
            "breakeven kalau nambah di harga sekarang. Ini murni kalkulator "
            "angka — keputusan nambah atau tidak tetap di tangan kamu."
        )

        c1, c2 = st.columns(2)
        existing_lot = c1.number_input(
            "Lot yang sudah dipegang", min_value=1, value=10, step=1,
            key=f"ad_lot_{res.ticker}",
        )
        existing_avg = c2.number_input(
            "Harga rata-rata beli (Rp)", min_value=1.0,
            value=float(round(res.harga * 1.1, 0)), step=1.0,
            key=f"ad_avg_{res.ticker}",
        )

        unrealized_pct = (res.harga - existing_avg) / existing_avg * 100
        unrealized_rp  = (res.harga - existing_avg) * existing_lot * 100
        pl_color = "normal" if unrealized_pct >= 0 else "inverse"
        st.metric("Floating P/L saat ini", f"{unrealized_pct:+.1f}%",
                  f"Rp{unrealized_rp:+,.0f}", delta_color=pl_color)

        # FIX: batas slider sebelumnya cuma existing_lot*3 — kalau existing_lot
        # kecil (mis. 1 lot), range jadi cuma 0-20, nggak cukup buat simulasi
        # nambah banyak. Sekarang ikut mempertimbangkan berapa lot yang bisa
        # dibeli dari modal yang di-configure di sidebar, supaya range slider
        # tetap masuk akal berapa pun ukuran posisi awal.
        max_by_modal = int(modal_rp / (100 * res.harga)) if res.harga > 0 else 20
        max_tambah   = max(existing_lot * 3, max_by_modal, 20)
        tambah_lot = st.slider(
            "Mau nambah berapa lot di harga sekarang?", 0, int(max_tambah),
            value=int(existing_lot), key=f"ad_add_{res.ticker}",
        )

        if tambah_lot > 0:
            new_avg = (existing_lot * existing_avg + tambah_lot * res.harga) / (existing_lot + tambah_lot)
            breakeven_pct  = (new_avg / res.harga - 1) * 100
            modal_tambahan = tambah_lot * 100 * res.harga

            d1, d2, d3 = st.columns(3)
            d1.metric("Average baru", f"{new_avg:,.0f}",
                      f"{(new_avg/existing_avg-1)*100:+.1f}% dari avg lama")
            d2.metric("Harga perlu naik", f"{breakeven_pct:+.1f}%",
                      "buat breakeven dari average baru", delta_color="off")
            d3.metric("Modal tambahan", f"Rp{modal_tambahan/1_000_000:,.1f}jt")

            # ── Cross-check ke target/resistance teknikal yang sudah dihitung ──
            if res.fib_mode == "bullish":
                potensi_t1 = (res.target1 / new_avg - 1) * 100
                st.caption(
                    f"📐 Kalau Target 1 ({res.target1:,.0f}) tercapai, potensi return dari "
                    f"average baru: **{potensi_t1:+.1f}%**"
                )
            else:
                potensi_r = (res.entry_hi / new_avg - 1) * 100
                st.caption(
                    f"📐 Kalau harga rally ke zona Resistance ({res.entry_hi:,.0f}), potensi "
                    f"return dari average baru: **{potensi_r:+.1f}%** — resistance ini belum "
                    f"tentu tercapai, bukan target pasti."
                )

            # ── Cross-check eksposur ke config Position Sizing di sidebar ──
            total_lot   = existing_lot + tambah_lot
            total_value = total_lot * 100 * res.harga
            pct_modal   = total_value / modal_rp * 100 if modal_rp > 0 else 0
            if pct_modal > max_expo * 100:
                st.warning(
                    f"⚠️ Total posisi setelah nambah: Rp{total_value/1_000_000:,.1f}jt "
                    f"= **{pct_modal:.0f}% dari modal** — di atas batas eksposur "
                    f"{max_expo*100:.0f}% yang kamu set di sidebar. Pertimbangkan kurangi jumlah."
                )
            else:
                st.caption(
                    f"Total posisi setelah nambah: Rp{total_value/1_000_000:,.1f}jt "
                    f"({pct_modal:.0f}% dari modal, masih dalam batas {max_expo*100:.0f}%)"
                )

            # ── Konteks teknikal — bukan rekomendasi, cuma info posisi relatif ──
            if res.fib_mode == "bearish":
                jarak_invalidasi = (res.stop_loss / res.harga - 1) * 100
                st.warning(
                    f"⚠️ Saham ini belum menunjukkan tanda reversal — level invalidasi "
                    f"breakdown ({res.stop_loss:,.0f}) masih **{jarak_invalidasi:+.1f}%** di "
                    f"atas harga sekarang. Average down di kondisi ini menaikkan risiko "
                    f"konsentrasi di saham yang trend-nya masih turun — pastikan alasanmu "
                    f"masih valid sebelum nambah."
                )
            elif res.gate_passed:
                st.success(
                    f"✅ Struktur teknikal saat ini bullish (di atas awan Ichimoku + HA "
                    f"hijau). Average down di sini lebih dekat ke 'beli pullback dalam "
                    f"uptrend' — tetap perhatikan stop loss di {res.stop_loss:,.0f}."
                )
            else:
                st.info(
                    "ℹ️ Gate teknikal belum lolos penuh — cek ulang alasan sinyal di "
                    "atas sebelum menambah posisi."
                )


@st.dialog("📊 Trading Plan Detail", width="large")
def show_modal(res: SR, df_raw: pd.DataFrame,
               modal_rp: float = 100_000_000, max_expo: float = 0.25):
    render_analysis_content(res, df_raw, modal_rp, max_expo)


def show_landing_page():
    st.markdown("""
### 👆 Klik **Jalankan Screener** di sidebar untuk memulai.

**Metodologi v6.6:**
| Komponen | Poin | Detail |
|---|---|---|
| Ichimoku | 5 | Cloud color, TK cross, Kijun support, Kijun slope, **Chikou confirm** |
| Trend/ADX | 3 | ADX strength, DI+/DI−, EMA20 vs EMA50 |
| Momentum | 4 | MACD fresh cross (+2), Stoch RSI, HA candle quality |
| Volume | 2 | Volume spike ×ADV20, ADV-Rp likuiditas |
| RS vs IHSG | 1 | Excess return z-score 20 hari |
| **Total** | **15** | |

**Gate wajib (gagal = tidak masuk screener):**
- ✅ Harga di atas **present cloud** Ichimoku (dihitung manual, shift 26 terverifikasi empiris)
- ✅ Heikin Ashi hijau (formula rekursif benar)
- ✅ R/R ≥ min R/R sidebar (default 1.5:1)
- ✅ ADV-Rp ≥ threshold likuiditas sidebar

**Temuan & fix utama deep-dive v6.3** *(tetap berlaku di v6.4)*:
- 🔧 **[Kritis]** UI tidak lagi kolaps ke landing page saat klik ticker/filter — hasil disimpan di session_state
- 🔧 **[Kritis]** Ichimoku cloud: shift(26) diverifikasi EMPIRIS langsung terhadap library `ta` (bukan asumsi)
- 🔧 find_swing(): swing high (puncak) → swing low SEBELUM puncak (bukan pivot independen)
- 🔧 Stop loss: wajib minimal ±1.2×ATR (hindari SL kelewat ketat/whipsaw)
- 🔧 Position sizing: tambah cap eksposur 25% modal (hindari over-concentration)
- 🔧 Lookback Fibonacci adaptif diskalakan ulang (terbukti numerik lama selalu floor 60)
- 🔧 Chikou Span ditambahkan sebagai konfirmasi + referensi chart

**Baru di v6.4:** mode **🔍 Analisa Manual** di sidebar — cek satu ticker
spesifik (bebas, tidak harus ada di emiten.csv), gate wajib tidak
digerbang, tapi signal BUY/STRONG BUY tetap ditahan ke maksimal WATCH
jika gate teknikal gagal (supaya tidak kontradiksi dengan Screener Massal).
    """)


def render_manual_analysis(sek_map: dict, modal_rp: float, risk_pct: float, max_expo: float):
    """
    Mode 'Analisa Manual' — input ticker bebas, dapat trading plan lengkap
    yang sama seperti modal Action Panel Screener Massal, TANPA perlu
    ticker itu lolos gate wajib maupun ada di emiten.csv.

    Hasil disimpan di st.session_state (namespace 'manual_*', terpisah
    dari 'ifh_run' milik Screener Massal) — konsisten dengan pola fix
    arsitektur v6.3: render selalu baca dari session_state, bukan dari
    status tombol yang ephemeral, supaya interaksi lain (mis. pindah tab)
    tidak menghilangkan hasil analisis yang sudah didapat.
    """
    st.markdown("### 🔍 Analisa Manual — Cek Saham Spesifik")
    st.markdown(
        "<p style='color:#848E9C;'>Masukkan kode ticker APAPUN — tidak harus ada di "
        "daftar emiten.csv. Gate wajib (di atas awan + HA hijau) <b>tidak digerbang</b> "
        "di sini, jadi kamu bisa cek saham yang sedang bearish atau yang sudah kamu pegang.</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([4, 1])
    with col1:
        ticker_raw = st.text_input(
            "Kode Ticker", placeholder="Contoh: BBCA, TLKM, BREN, atau BBCA.JK",
            key="manual_ticker_input", label_visibility="collapsed",
        )
    with col2:
        analyze_btn = st.button("🔎 Analisa", width='stretch', type="primary")

    # Quick-pick beberapa ticker populer dari emiten.csv (kalau ada)
    quick_picks = list(sek_map.keys())[:8]
    if quick_picks:
        st.markdown(
            "<span style='color:#848E9C;font-size:.78rem;'>Cepat pilih:</span>",
            unsafe_allow_html=True,
        )
        qcols = st.columns(len(quick_picks))
        for i, qt in enumerate(quick_picks):
            with qcols[i]:
                if st.button(qt.replace(".JK", ""), key=f"quickpick_{qt}", width='stretch'):
                    st.session_state["manual_ticker_input"] = qt.replace(".JK", "")
                    ticker_raw = qt.replace(".JK", "")
                    analyze_btn = True

    if analyze_btn and ticker_raw and ticker_raw.strip():
        ticker = ticker_raw.strip().upper()
        if not ticker.endswith(".JK"):
            ticker += ".JK"

        with st.spinner(f"📡 Mengunduh & menganalisis {ticker}..."):
            df = _safe_download(ticker, PERIOD)

            if df is None:
                st.session_state["manual_result"] = None
                st.session_state["manual_error"] = (
                    f"❌ Data untuk **{ticker}** tidak ditemukan atau tidak cukup "
                    f"(minimal {MIN_BARS} hari trading). Cek kembali kode ticker-nya "
                    f"— pastikan ini kode saham IDX yang valid."
                )
            else:
                ihsg_df    = fetch_ihsg()
                ihsg_close = ihsg_df["Close"].squeeze() if not ihsg_df.empty else None
                ihsg_ret   = ihsg_close.pct_change() if ihsg_close is not None else None
                sektor     = sek_map.get(ticker, "—")

                r = screen_one(
                    ticker, df, ihsg_ret, min_adv_rp=0, sektor=sektor,
                    modal_rp=modal_rp, risk_pct=risk_pct, max_exposure_pct=max_expo,
                    require_gate=False,   # FIX inti fitur: jangan gerbang di mode manual
                )
                if r is None:
                    st.session_state["manual_result"] = None
                    st.session_state["manual_error"] = (
                        f"❌ Data **{ticker}** tidak cukup untuk dianalisis "
                        f"(kemungkinan data harga tidak lengkap)."
                    )
                else:
                    st.session_state["manual_result"] = r
                    st.session_state["manual_df"]     = df
                    st.session_state["manual_error"]  = None

    # ── Render selalu dari session_state (pola sama seperti fix v6.3) ──
    if st.session_state.get("manual_error"):
        st.error(st.session_state["manual_error"])

    res = st.session_state.get("manual_result")
    if res is not None:
        st.markdown("---")
        render_analysis_content(res, st.session_state["manual_df"], modal_rp, max_expo)
    elif not st.session_state.get("manual_error"):
        st.info("💡 Masukkan kode ticker di atas dan klik **Analisa** untuk memulai.")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def main():
    st.markdown(
        "<h1 style='margin-bottom:2px;'>🦅 Ichi-Fibo-Heikin Pro v6.6</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#848E9C;margin-top:0;'>"
        "Screener teknikal IDX · Ichimoku · Fibonacci · Heikin Ashi · "
        "ADX · Stoch RSI · RS vs IHSG · Position Sizing</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    emiten_df   = load_emiten()
    all_tickers = emiten_df["ticker"].tolist()
    sek_col = (emiten_df["sektor"] if "sektor" in emiten_df.columns
               else pd.Series("—", index=emiten_df.index))
    sek_map = dict(zip(emiten_df["ticker"], sek_col.fillna("—")))

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("## ⚙️ Konfigurasi")

        mode = st.radio(
            "Mode", ["🚀 Screener Massal", "🔍 Analisa Manual"],
            key="app_mode",
            help="Screener Massal: scan banyak saham sekaligus, wajib lolos gate teknikal. "
                 "Analisa Manual: cek satu ticker spesifik, gate tidak wajib.",
        )
        st.markdown("---")

        if mode == "🚀 Screener Massal":
            n_tickers = st.slider("Jumlah emiten (Top N)", 10, len(all_tickers),
                                  min(200, len(all_tickers)), 10)
            min_adv_rp = st.slider(
                "Min Likuiditas (Rp juta/hari)", 500, 20_000, 2_000, 500,
                help="ADV-Rp: Average Daily Value dalam juta rupiah. 2000 = Rp2 miliar/hari.",
            )
            min_rr = st.slider(
                "Min R/R Ratio", 1.0, 4.0, 1.5, 0.5,
                help="Bisa diubah kapan saja SETELAH screening — sinyal AVOID/BUY "
                     "dihitung ulang otomatis dari hasil yang sudah ada, tanpa perlu "
                     "download ulang.",
            )
            workers = st.slider("Parallel workers", 2, 12, 6, 1,
                                help="Thread paralel untuk download & analisis. Default 6 aman.")
            chunk = st.slider("Ticker per request (chunk)", 5, 50, CHUNK_SIZE, 5,
                              help="Chunk 20: 953 ticker → 48 request.")
            st.markdown("---")
        else:
            st.caption(
                "🔍 **Mode Analisa Manual** — masukkan ticker di halaman utama. "
                "Gate teknikal (di atas awan + HA hijau) TIDAK wajib di sini, jadi "
                "kamu bisa cek saham apapun, termasuk yang sedang bearish atau yang "
                "sudah kamu pegang."
            )
            st.markdown("---")

        st.markdown("**Position Sizing**")
        modal_rp = st.number_input("Modal (Rp)", value=100_000_000, step=10_000_000,
                                   format="%d")
        risk_pct = st.slider("Risk per trade (%)", 0.5, 3.0, 1.0, 0.5) / 100
        max_expo = st.slider("Max eksposur per saham (%)", 10, 50, 25, 5) / 100

        if mode == "🚀 Screener Massal":
            st.markdown("---")
            st.markdown("**Filter Tampilan**")
            show_sigs = st.multiselect("Tampilkan sinyal:", SIGNAL_ORDER,
                                       default=["STRONG BUY", "BUY", "WATCH"])

        st.markdown("---")
        st.markdown(
            f"<span style='color:#848E9C;font-size:.78rem;'>"
            f"v6.6 · {len(all_tickers)} emiten loaded</span>",
            unsafe_allow_html=True,
        )

        if mode == "🚀 Screener Massal":
            run_btn = st.button("🚀 Jalankan Screener", width='stretch', type="primary")
        else:
            run_btn = False

    # ══════════════════════════════════════════════════
    # BRANCH: Analisa Manual dirender terpisah, TIDAK menyentuh
    # pipeline/session_state milik Screener Massal sama sekali.
    # ══════════════════════════════════════════════════
    if mode == "🔍 Analisa Manual":
        render_manual_analysis(sek_map, modal_rp, risk_pct, max_expo)
        return

    # ══════════════════════════════════════════════════
    # PIPELINE — HANYA jalan saat run_btn diklik.
    # Hasil disimpan ke session_state; TIDAK menggerbang render di bawah.
    # ══════════════════════════════════════════════════
    if run_btn:
        target = all_tickers[:n_tickers]

        with st.spinner("📡 Mengunduh data IHSG..."):
            ihsg_df = fetch_ihsg()
            ihsg_close = ihsg_df["Close"].squeeze() if not ihsg_df.empty else None
            ihsg_ret = ihsg_close.pct_change() if ihsg_close is not None else None

        if ihsg_df.empty:
            st.warning("⚠️ IHSG tidak berhasil diunduh — RS score tidak tersedia, "
                       "market regime memakai breadth saja.")

        info = st.empty()
        info.info(f"⏳ Mengunduh {len(target)} emiten "
                 f"({chunk} ticker/request, {workers} parallel workers)...")

        t0      = time.time()
        all_dfs = fetch_all(target, PERIOD, max_workers=workers, chunk_size=chunk)
        elapsed = time.time() - t0
        n_ok    = len(all_dfs)

        if n_ok == 0:
            info.error(
                f"❌ Tidak ada data berhasil diunduh dalam {elapsed:.1f}s.\n\n"
                "Kemungkinan penyebab:\n"
                "1. Koneksi internet tidak aktif\n"
                "2. Yahoo Finance rate-limit — coba: kurangi workers, kecilkan chunk, tunggu 1 menit\n"
                "3. Ticker di emiten.csv tidak valid (pastikan ada suffix .JK)"
            )
            # TIDAK menghapus session_state lama — kalau ada hasil sebelumnya, tetap ditampilkan di bawah.
        else:
            info.success(f"✅ {n_ok}/{len(target)} emiten berhasil diunduh dalam {elapsed:.1f}s "
                        f"({n_ok/max(elapsed,1):.0f} ticker/s)")

            def _worker(args):
                ticker, df = args
                sektor = sek_map.get(ticker, "—")
                try:
                    ema20_last = float(df["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
                    above_ema  = float(df["Close"].iloc[-1]) > ema20_last
                except Exception:
                    above_ema = False
                try:
                    r = screen_one(ticker, df, ihsg_ret, min_adv_rp, sektor,
                                   modal_rp=modal_rp, risk_pct=risk_pct,
                                   max_exposure_pct=max_expo)
                except Exception:
                    r = None
                return r, above_ema

            prog = st.progress(0.0, text="Analisis paralel...")
            raw_results: list = []
            above_count = total_breadth = 0
            items = list(all_dfs.items())

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_worker, item): item[0] for item in items}
                done = 0
                for fut in as_completed(futs):
                    done += 1
                    try:
                        r, above_ema = fut.result()
                        total_breadth += 1
                        if above_ema:
                            above_count += 1
                        if r is not None:
                            raw_results.append(r)
                    except Exception:
                        pass
                    prog.progress(done / len(futs), text=f"Analisis {done}/{len(futs)}...")
            prog.empty()

            breadth_inline       = (above_count / total_breadth * 100) if total_breadth > 0 else 50.0
            ihsg_bull, ihsg_ratio = ihsg_trend(ihsg_df)
            regime = decide_regime(ihsg_bull, breadth_inline)

            # ── Simpan SEMUA yang dibutuhkan untuk render ke session_state ──
            st.session_state["ifh_run"] = dict(
                raw_results=raw_results, all_dfs=all_dfs,
                regime=regime, breadth=breadth_inline, ihsg_ratio=ihsg_ratio,
                n_ok=n_ok, n_target=len(target), elapsed=elapsed,
            )

    # ══════════════════════════════════════════════════
    # RENDER — SELALU dari session_state, TIDAK bergantung run_btn.
    # Ini yang membuat klik ticker / ubah filter / download CSV TIDAK
    # membuat hasil lenyap kembali ke landing page.
    # ══════════════════════════════════════════════════
    run_data = st.session_state.get("ifh_run")
    if run_data is None:
        show_landing_page()
        return

    raw_results = run_data["raw_results"]
    all_dfs     = run_data["all_dfs"]

    render_regime(run_data["regime"], run_data["breadth"], run_data["ihsg_ratio"])

    results_eff = [(r, effective_signal(r, min_rr)) for r in raw_results]
    filtered    = [(r, eff) for r, eff in results_eff if eff in show_sigs]
    filtered.sort(key=lambda pair: (SIGNAL_ORDER.index(pair[1]), -pair[0].score))

    st.markdown("")
    render_kpi(raw_results, min_rr)
    st.markdown(
        f"### 📋 Hasil — **{len(filtered)}** kandidat tampil  "
        f"<span style='color:#848E9C;font-size:.85rem;'>"
        f"({len(raw_results)} lolos gate dari {run_data['n_ok']} diunduh)</span>",
        unsafe_allow_html=True,
    )

    if not filtered:
        st.warning("Tidak ada saham yang memenuhi kriteria. "
                   "Coba: turunkan Min ADV, turunkan Min R/R, "
                   "atau tambahkan WATCH ke filter tampilan.")
        return

    df_disp = results_to_df(filtered)
    st.dataframe(
        df_disp, width='stretch', hide_index=True,
        column_config={
            "Signal":     st.column_config.TextColumn(width=110),
            "Lot Sizing": st.column_config.NumberColumn(format="%d"),
        },
    )

    csv = df_disp.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download hasil (.csv)", csv,
                       file_name="ifh_screener_results.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("### 🔍 Action Panel — Klik ticker untuk Trading Plan lengkap")

    for sig in SIGNAL_ORDER:
        tier = [r for r, eff in filtered if eff == sig]
        if not tier:
            continue
        exp_label = (f"{sig}  ({len(tier)} saham)  "
                    f"—  avg score {sum(r.score for r in tier)/len(tier):.1f}/{tier[0].score_max}")
        with st.expander(exp_label, expanded=(sig in ("STRONG BUY", "BUY"))):
            NCOLS = 5
            rows = [tier[i: i + NCOLS] for i in range(0, len(tier), NCOLS)]
            for row in rows:
                cols = st.columns(NCOLS)
                for j, r in enumerate(row):
                    with cols[j]:
                        if st.button(
                            f"**{r.ticker}**\n{r.score}/{r.score_max} · R/R {r.rr:.1f}",
                            key=f"btn_{r.ticker}_{sig}",
                            width='stretch',
                            help=(f"{r.conf} | ADX {r.adx:.0f} | Vol {r.vol_rel:.1f}× | "
                                 f"RS {r.rs:+.2f} | ADV Rp{r.adv_rp:,.0f}jt"),
                        ):
                            show_modal(r, all_dfs[r.ticker], modal_rp, max_expo)


if __name__ == "__main__":
    main()
