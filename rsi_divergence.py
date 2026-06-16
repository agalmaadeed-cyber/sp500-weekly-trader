"""
rsi_divergence.py
كاشف RSI Divergence على البيانات اليومية
"""

import pandas as pd
import numpy as np

RSI_PERIOD   = 14
SWING_WINDOW = 5
RSI_BULL_MAX = 40
RSI_BEAR_MIN = 60
ATR_PERIOD   = 14
STOP_ATR     = 1.5
T1_ATR       = 2.0
T2_ATR       = 4.0
MA200_PERIOD = 200


def compute_rsi(close: pd.Series, period=RSI_PERIOD) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(high, low, close, period=ATR_PERIOD) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def find_swing_lows(series: pd.Series, window=SWING_WINDOW) -> pd.Series:
    arr  = series.values
    mask = np.zeros(len(arr), dtype=bool)
    for i in range(window, len(arr) - window):
        if arr[i] == min(arr[i - window: i + window + 1]):
            mask[i] = True
    return pd.Series(mask, index=series.index)


def find_swing_highs(series: pd.Series, window=SWING_WINDOW) -> pd.Series:
    arr  = series.values
    mask = np.zeros(len(arr), dtype=bool)
    for i in range(window, len(arr) - window):
        if arr[i] == max(arr[i - window: i + window + 1]):
            mask[i] = True
    return pd.Series(mask, index=series.index)


def detect_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: أعمدة (open, high, low, close, volume)
    returns: DataFrame بالإشارات
    """
    df = df.copy()
    df["rsi"]  = compute_rsi(df["close"])
    df["atr"]  = compute_atr(df["high"], df["low"], df["close"])
    df["ma200"] = df["close"].rolling(MA200_PERIOD).mean()

    sl_mask = find_swing_lows(df["close"])
    sh_mask = find_swing_highs(df["close"])

    low_dates  = df.index[sl_mask].tolist()
    high_dates = df.index[sh_mask].tolist()

    signals = []

    # ── Bullish Divergence ────────────────────────────────────
    for j in range(1, len(low_dates)):
        d1, d2 = low_dates[j - 1], low_dates[j]
        p1, p2 = df.loc[d1, "close"], df.loc[d2, "close"]
        r1, r2 = df.loc[d1, "rsi"],   df.loc[d2, "rsi"]
        if pd.isna(r1) or pd.isna(r2):
            continue
        if p2 < p1 and r2 > r1 and r2 < RSI_BULL_MAX:
            atr = df.loc[d2, "atr"]
            if pd.isna(atr) or atr <= 0:
                continue
            entry = df.loc[d2, "close"]
            signals.append({
                "date":      d2,
                "direction": "long",
                "entry":     round(entry, 4),
                "stop":      round(entry - STOP_ATR * atr, 4),
                "target1":   round(entry + T1_ATR   * atr, 4),
                "target2":   round(entry + T2_ATR   * atr, 4),
                "rsi":       round(r2, 1),
                "atr":       round(atr, 4),
            })

    # ── Bearish Divergence (فلتر MA200) ───────────────────────
    for j in range(1, len(high_dates)):
        d1, d2 = high_dates[j - 1], high_dates[j]
        p1, p2 = df.loc[d1, "close"], df.loc[d2, "close"]
        r1, r2 = df.loc[d1, "rsi"],   df.loc[d2, "rsi"]
        if pd.isna(r1) or pd.isna(r2):
            continue
        if p2 > p1 and r2 < r1 and r2 > RSI_BEAR_MIN:
            # فلتر MA200: السعر يجب أن يكون تحت MA200
            ma200 = df.loc[d2, "ma200"]
            if pd.isna(ma200) or p2 >= ma200:
                continue
            atr = df.loc[d2, "atr"]
            if pd.isna(atr) or atr <= 0:
                continue
            entry = df.loc[d2, "close"]
            signals.append({
                "date":      d2,
                "direction": "short",
                "entry":     round(entry, 4),
                "stop":      round(entry + STOP_ATR * atr, 4),
                "target1":   round(entry - T1_ATR   * atr, 4),
                "target2":   round(entry - T2_ATR   * atr, 4),
                "rsi":       round(r2, 1),
                "atr":       round(atr, 4),
            })

    if not signals:
        return pd.DataFrame()

    out = pd.DataFrame(signals).set_index("date").sort_index()
    return out[~out.index.duplicated(keep="first")]
